import { env } from 'cloudflare:workers';
import { readBoundedJson, validateResearchPostHeaders } from '@/lib/research-http';
import {
  consumeResearchRateLimit,
  issueResearchSessionToken,
  purgeExpiredResearchSecurityRows,
  researchRateLimitKey,
} from '@/lib/research-security';
import { MAX_SESSION_REQUEST_BYTES, parseResearchSessionGrant } from '@/lib/validation';

const SESSION_GRANTS_PER_HOUR = 60;
const HOUR_MS = 60 * 60 * 1000;

function json(body: unknown, status = 200, headers?: HeadersInit): Response {
  const responseHeaders = new Headers(headers);
  responseHeaders.set('Cache-Control', 'no-store');
  responseHeaders.set('X-Content-Type-Options', 'nosniff');
  return Response.json(body, { status, headers: responseHeaders });
}

export async function POST(request: Request): Promise<Response> {
  const workerEnv = env as unknown as Env;
  const headerCheck = validateResearchPostHeaders(request, workerEnv.SITE_ORIGIN);
  if (!headerCheck.ok) return json({ error: headerCheck.error }, headerCheck.status);

  const database = workerEnv.DB;
  if (!database) return json({ error: 'The research database is not configured' }, 503);

  const body = await readBoundedJson(request, MAX_SESSION_REQUEST_BYTES);
  if (!body.ok) return json({ error: body.error }, body.status);
  const now = Date.now();
  const parsed = parseResearchSessionGrant(body.value, now);
  if (!parsed.ok) return json({ error: parsed.error }, 400);

  try {
    const networkKey = await researchRateLimitKey(database, request, 'session-grant');
    const rate = await consumeResearchRateLimit(
      database,
      networkKey,
      SESSION_GRANTS_PER_HOUR,
      HOUR_MS,
      now,
    );
    if (!rate.allowed) {
      return json(
        { error: 'Too many anonymous research sessions. Please try again later.' },
        429,
        { 'Retry-After': String(Math.max(1, Math.ceil((rate.resetAt - now) / 1000))) },
      );
    }

    const issued = await issueResearchSessionToken(database, parsed.value, now);
    await purgeExpiredResearchSecurityRows(database, now);
    return json(
      {
        ok: true,
        consentVersion: parsed.value.consentVersion,
        expiresAt: issued.expiresAt,
      },
      201,
      { 'Set-Cookie': issued.cookie },
    );
  } catch (error) {
    console.error('research session issuance failed', error);
    return json({ error: 'The anonymous research session could not be created. Please try again later.' }, 503);
  }
}
