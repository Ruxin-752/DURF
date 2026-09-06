import { readBoundedJson, validateResearchPostHeaders } from './research-http';
import { CLIENT_VERSION, SCHEMA_VERSION } from './research-types';

const DIAGNOSTIC_VERSION = 'durf-isolated-synthetic-roundtrip-v1';
const PROBE_KEY = 'synthetic_probe';
const PROBE_VALUE = 'durf_synthetic_roundtrip_v1';

type DiagnosticEnvironment = Pick<Env, 'ADMIN_EXPORT_TOKEN' | 'SITE_ORIGIN'> & {
  DB?: D1Database;
};

function json(body: unknown, status = 200, headers?: HeadersInit): Response {
  const responseHeaders = new Headers(headers);
  responseHeaders.set('Cache-Control', 'no-store');
  responseHeaders.set('X-Content-Type-Options', 'nosniff');
  return Response.json(body, { status, headers: responseHeaders });
}

function tokensMatch(provided: string, expected: string): boolean {
  if (provided.length !== expected.length || expected.length < 16) return false;
  let mismatch = 0;
  for (let index = 0; index < provided.length; index += 1) {
    mismatch |= provided.charCodeAt(index) ^ expected.charCodeAt(index);
  }
  return mismatch === 0;
}

interface ProbeRow {
  nonce: string;
  probe_key: string;
  probe_value: string;
}

/**
 * An administrative transport check, isolated from participant data and consent.
 * D1 batches execute these statements sequentially in a transaction: failure
 * rolls back the probe and successful execution removes it in the same batch.
 * A fresh nonce isolates concurrent calls; no client data is persisted.
 */
async function runSyntheticRoundtrip(database: D1Database): Promise<boolean> {
  const created = await database.prepare(`
    CREATE TABLE IF NOT EXISTS research_diagnostics (
      nonce TEXT PRIMARY KEY,
      probe_key TEXT NOT NULL CHECK (probe_key = 'synthetic_probe'),
      probe_value TEXT NOT NULL,
      created_at INTEGER NOT NULL
    )
  `).run();
  if (!created.success) return false;

  const nonce = crypto.randomUUID();
  const results = await database.batch([
    database.prepare(
      'INSERT INTO research_diagnostics (nonce, probe_key, probe_value, created_at) VALUES (?, ?, ?, ?)',
    ).bind(nonce, PROBE_KEY, PROBE_VALUE, Date.now()),
    database.prepare(
      'SELECT nonce, probe_key, probe_value FROM research_diagnostics WHERE nonce = ? AND probe_key = ?',
    ).bind(nonce, PROBE_KEY),
    database.prepare(
      'DELETE FROM research_diagnostics WHERE nonce = ? AND probe_key = ?',
    ).bind(nonce, PROBE_KEY),
    database.prepare(
      'SELECT nonce FROM research_diagnostics WHERE nonce = ? AND probe_key = ?',
    ).bind(nonce, PROBE_KEY),
  ]);
  if (results.length !== 4 || results.some((result) => !result.success)) return false;
  const readback = results[1].results as unknown as ProbeRow[];
  return (
    results[0].meta.changes === 1 &&
    readback.length === 1 &&
    readback[0].nonce === nonce &&
    readback[0].probe_key === PROBE_KEY &&
    readback[0].probe_value === PROBE_VALUE &&
    results[2].meta.changes === 1 &&
    results[3].results.length === 0
  );
}

export async function handleSyntheticDiagnosticRequest(
  request: Request,
  environment: DiagnosticEnvironment,
): Promise<Response> {
  if (request.method !== 'POST') {
    return json({ error: 'Method not allowed' }, 405, { Allow: 'POST' });
  }
  const expected = environment.ADMIN_EXPORT_TOKEN;
  if (!expected) return json({ error: 'ADMIN_EXPORT_TOKEN is not configured' }, 503);

  const authorization = request.headers.get('authorization') ?? '';
  const provided = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
  if (!tokensMatch(provided, expected)) {
    return json({ error: 'Unauthorized' }, 401, { 'WWW-Authenticate': 'Bearer' });
  }

  // Authentication precedes every database operation, including schema creation.
  const headerCheck = validateResearchPostHeaders(request, environment.SITE_ORIGIN);
  if (!headerCheck.ok) return json({ error: headerCheck.error }, headerCheck.status);
  if (new URL(request.url).search) return json({ error: 'Query parameters are not accepted' }, 400);
  const body = await readBoundedJson(request, 128);
  if (!body.ok) return json({ error: body.error }, body.status);
  if (
    body.value === null ||
    typeof body.value !== 'object' ||
    Array.isArray(body.value) ||
    Object.keys(body.value).length !== 0
  ) {
    return json({ error: 'The diagnostic body must be an empty JSON object' }, 400);
  }
  if (!environment.DB) return json({ error: 'The research database is not configured' }, 503);

  try {
    if (!(await runSyntheticRoundtrip(environment.DB))) throw new Error('Probe verification failed');
    return json({
      ok: true,
      diagnostic_version: DIAGNOSTIC_VERSION,
      schema_version: SCHEMA_VERSION,
      client_version: CLIENT_VERSION,
      no_research_rows_written: true,
      data_roundtrip: { inserted: true, read_back_matches: true, deleted: true },
      scope: 'Isolated synthetic database transport only; participant ingestion and training quality are not measured.',
    });
  } catch {
    // Never expose the admin token, random nonce, SQL errors, or database contents.
    return json({
      ok: false,
      error: 'Synthetic database diagnostic failed',
      no_research_rows_written: true,
      data_roundtrip: { status: 'failed' },
    }, 503);
  }
}
