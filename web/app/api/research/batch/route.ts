import { env } from 'cloudflare:workers';
import { persistResearchBatch, ResearchSessionConflictError } from '@/lib/db';
import { readBoundedJson, validateResearchPostHeaders } from '@/lib/research-http';
import {
  clearResearchSessionCookie,
  consumeResearchRateLimit,
  researchRateLimitKey,
  verifyResearchSessionToken,
} from '@/lib/research-security';
import { MAX_REQUEST_BYTES, parseResearchBatch } from '@/lib/validation';

const BATCHES_PER_SESSION_MINUTE = 40;
const BATCHES_PER_NETWORK_MINUTE = 300;
const MINUTE_MS = 60_000;

function json(body: unknown, status = 200, headers?: HeadersInit): Response {
  const responseHeaders = new Headers(headers);
  responseHeaders.set('Cache-Control', 'no-store');
  responseHeaders.set('X-Content-Type-Options', 'nosniff');
  return Response.json(body, {
    status,
    headers: responseHeaders,
  });
}

export async function POST(request: Request): Promise<Response> {
  const workerEnv = env as unknown as Env;
  const headerCheck = validateResearchPostHeaders(request, workerEnv.SITE_ORIGIN);
  if (!headerCheck.ok) return json({ error: headerCheck.error }, headerCheck.status);

  const database = workerEnv.DB;
  if (!database) return json({ error: 'The research database is not configured' }, 503);

  const body = await readBoundedJson(request, MAX_REQUEST_BYTES);
  if (!body.ok) return json({ error: body.error }, body.status);
  const parsed = parseResearchBatch(body.value);
  if (!parsed.ok) return json({ error: parsed.error }, 400);

  try {
    const verified = await verifyResearchSessionToken(
      database,
      request,
      parsed.value.session,
    );
    if (!verified.ok) {
      return json(
        { error: verified.error },
        401,
        { 'Set-Cookie': clearResearchSessionCookie() },
      );
    }

    const now = Date.now();
    const networkKey = await researchRateLimitKey(database, request, 'research-batch-network');
    const networkRate = await consumeResearchRateLimit(
      database,
      networkKey,
      BATCHES_PER_NETWORK_MINUTE,
      MINUTE_MS,
      now,
    );
    const sessionRate = await consumeResearchRateLimit(
      database,
      `research-batch-session:${verified.tokenHash}`,
      BATCHES_PER_SESSION_MINUTE,
      MINUTE_MS,
      now,
    );
    if (!networkRate.allowed || !sessionRate.allowed) {
      const resetAt = Math.max(networkRate.resetAt, sessionRate.resetAt);
      return json(
        { error: 'Too many submissions. Please try again later.' },
        429,
        { 'Retry-After': String(Math.max(1, Math.ceil((resetAt - now) / 1000))) },
      );
    }

    const result = await persistResearchBatch(database, parsed.value);
    return json(result);
  } catch (error) {
    if (error instanceof ResearchSessionConflictError) {
      return json({ error: error.message }, 409);
    }
    console.error('research batch persistence failed', error);
    return json({ error: 'Data could not be saved. The client will retry automatically.' }, 503);
  }
}
