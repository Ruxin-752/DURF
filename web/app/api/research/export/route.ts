import { env } from 'cloudflare:workers';
import { exportTrainingJsonl } from '@/lib/db';

function tokensMatch(provided: string, expected: string): boolean {
  if (provided.length !== expected.length || expected.length < 16) return false;
  let mismatch = 0;
  for (let index = 0; index < provided.length; index += 1) {
    mismatch |= provided.charCodeAt(index) ^ expected.charCodeAt(index);
  }
  return mismatch === 0;
}

export async function GET(request: Request): Promise<Response> {
  const workerEnv = env as unknown as Env;
  const expected = workerEnv.ADMIN_EXPORT_TOKEN;
  if (!expected) {
    return Response.json({ error: 'ADMIN_EXPORT_TOKEN is not configured' }, { status: 503 });
  }

  const authorization = request.headers.get('authorization') ?? '';
  const provided = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
  if (!tokensMatch(provided, expected)) {
    return Response.json(
      { error: 'Unauthorized' },
      {
        status: 401,
        headers: { 'WWW-Authenticate': 'Bearer', 'Cache-Control': 'no-store' },
      },
    );
  }
  if (!workerEnv.DB) {
    return Response.json({ error: 'The research database is not configured' }, { status: 503 });
  }

  const jsonl = await exportTrainingJsonl(workerEnv.DB);
  const stamp = new Date().toISOString().slice(0, 10);
  return new Response(jsonl, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Content-Disposition': `attachment; filename="durf-training-${stamp}.jsonl"`,
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    },
  });
}
