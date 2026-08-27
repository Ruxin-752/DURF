import { env } from 'cloudflare:workers';
import { persistResearchBatch } from '@/lib/db';
import { allowAnonymousBatch } from '@/lib/rate-limit';
import { MAX_REQUEST_BYTES, parseResearchBatch } from '@/lib/validation';

function json(body: unknown, status = 200): Response {
  return Response.json(body, {
    status,
    headers: { 'Cache-Control': 'no-store' },
  });
}

export async function POST(request: Request): Promise<Response> {
  const contentLength = Number(request.headers.get('content-length') ?? 0);
  if (contentLength > MAX_REQUEST_BYTES) return json({ error: '请求过大' }, 413);

  const raw = await request.text();
  if (new TextEncoder().encode(raw).byteLength > MAX_REQUEST_BYTES) {
    return json({ error: '请求过大' }, 413);
  }

  let input: unknown;
  try {
    input = JSON.parse(raw);
  } catch {
    return json({ error: 'JSON 格式无效' }, 400);
  }

  const parsed = parseResearchBatch(input);
  if (!parsed.ok) return json({ error: parsed.error }, 400);
  if (!allowAnonymousBatch(parsed.value.session.anonymousUserId)) {
    return json({ error: '提交过于频繁，请稍后重试' }, 429);
  }

  const database = (env as unknown as Env).DB;
  if (!database) return json({ error: '研究数据库尚未绑定' }, 503);

  try {
    const result = await persistResearchBatch(database, parsed.value);
    return json({ ok: true, ...result });
  } catch (error) {
    console.error('research batch persistence failed', error);
    return json({ error: '数据暂时无法保存，客户端会自动重试' }, 503);
  }
}
