import { describe, expect, it } from 'vitest';
import { handleSyntheticDiagnosticRequest } from '../lib/research-diagnostic';

const TOKEN = 'test-only-administrator-token';
const ORIGIN = 'https://kitchen.example';

function request(body = '{}', options: { token?: string; origin?: string; query?: string; contentType?: string } = {}) {
  return new Request(`${ORIGIN}/api/research/diagnostic${options.query ?? ''}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${options.token ?? TOKEN}`,
      Origin: options.origin ?? ORIGIN,
      'Content-Type': options.contentType ?? 'application/json',
    },
    body,
  });
}

function fakeDatabase(options: { failAt?: number; corruptReadback?: boolean } = {}) {
  const state = new Map<string, { nonce: string; probe_key: string; probe_value: string }>();
  const sql: string[] = [];
  const batches: Array<Array<{ sql: string; values: unknown[] }>> = [];
  function prepare(statement: string) {
    sql.push(statement);
    const prepared = {
      sql: statement,
      values: [] as unknown[],
      bind(...values: unknown[]) { this.values = values; return this; },
      async run() { return { success: true, results: [], meta: { changes: 0 } }; },
    };
    return prepared;
  }
  const db = {
    prepare,
    async batch(statements: Array<{ sql: string; values: unknown[] }>) {
      batches.push(statements);
      const snapshot = new Map(state);
      const results = [];
      try {
        for (let index = 0; index < statements.length; index += 1) {
          const statement = statements[index];
          if (options.failAt === index) throw new Error('secret-database-detail');
          const nonce = String(statement.values[0]);
          let rows: unknown[] = [];
          let changes = 0;
          if (statement.sql.startsWith('INSERT')) {
            if (state.has(nonce)) throw new Error('duplicate nonce');
            state.set(nonce, { nonce, probe_key: String(statement.values[1]), probe_value: String(statement.values[2]) });
            changes = 1;
          } else if (statement.sql.startsWith('DELETE')) {
            changes = Number(state.delete(nonce));
          } else if (statement.sql.startsWith('SELECT')) {
            const row = state.get(nonce);
            rows = row ? [{ ...row, ...(options.corruptReadback ? { probe_value: 'wrong' } : {}) }] : [];
          }
          results.push({ success: true, results: rows, meta: { changes } });
        }
      } catch (error) {
        state.clear();
        for (const [key, value] of snapshot) state.set(key, value);
        throw error;
      }
      return results;
    },
  };
  return { db: db as unknown as D1Database, state, sql, batches };
}

describe('isolated administrative synthetic diagnostic', () => {
  it('denies invalid credentials before any database or schema access', async () => {
    const database = fakeDatabase();
    const response = await handleSyntheticDiagnosticRequest(request('{}', { token: 'wrong' }), {
      DB: database.db, ADMIN_EXPORT_TOKEN: TOKEN,
    });
    expect(response.status).toBe(401);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
    expect(database.sql).toHaveLength(0);
  });

  it('rejects unconfigured and too-short secrets without database access', async () => {
    const database = fakeDatabase();
    expect((await handleSyntheticDiagnosticRequest(request(), { DB: database.db })).status).toBe(503);
    expect((await handleSyntheticDiagnosticRequest(request('{}', { token: 'short' }), {
      DB: database.db, ADMIN_EXPORT_TOKEN: 'short',
    })).status).toBe(401);
    expect(database.sql).toHaveLength(0);
  });

  it.each(['null', '[]', '{"text":"synthetic sentence"}', '{"sessionId":"never-write"}', '{bad json'])('rejects non-empty or malformed payload %s without writes', async (body) => {
    const database = fakeDatabase();
    const response = await handleSyntheticDiagnosticRequest(request(body), { DB: database.db, ADMIN_EXPORT_TOKEN: TOKEN });
    expect(response.status).toBe(400);
    expect(database.sql).toHaveLength(0);
  });

  it('enforces origin, content type, query and body-size restrictions before writes', async () => {
    for (const [probe, status] of [
      [request('{}', { origin: 'https://other.example' }), 403],
      [request('{}', { contentType: 'text/plain' }), 415],
      [request('{}', { query: '?action=custom' }), 400],
      [request(`{"x":"${'x'.repeat(150)}"}`), 413],
    ] as const) {
      const database = fakeDatabase();
      expect((await handleSyntheticDiagnosticRequest(probe, { DB: database.db, ADMIN_EXPORT_TOKEN: TOKEN })).status).toBe(status);
      expect(database.sql).toHaveLength(0);
    }
  });

  it('writes, verifies and removes its probe without touching research tables or exposing nonce', async () => {
    const database = fakeDatabase();
    const response = await handleSyntheticDiagnosticRequest(request(), { DB: database.db, ADMIN_EXPORT_TOKEN: TOKEN });
    expect(response.status).toBe(200);
    const text = await response.text();
    const body = JSON.parse(text);
    expect(body.no_research_rows_written).toBe(true);
    expect(body.data_roundtrip).toEqual({ inserted: true, read_back_matches: true, deleted: true });
    expect(database.state.size).toBe(0);
    expect(database.batches).toHaveLength(1);
    const nonce = String(database.batches[0][0].values[0]);
    expect(text).not.toContain(nonce);
    expect(text).not.toContain(TOKEN);
    expect(database.sql.every((sql) => sql.includes('research_diagnostics'))).toBe(true);
    expect(database.sql.join(' ')).not.toMatch(/\b(sessions|events|feedback|research_session_tokens|research_rate_limits|research_security_secrets)\b/);
  });

  it('uses different nonce keys for concurrent and repeated diagnostics', async () => {
    const database = fakeDatabase();
    const responses = await Promise.all(Array.from({ length: 3 }, () =>
      handleSyntheticDiagnosticRequest(request(), { DB: database.db, ADMIN_EXPORT_TOKEN: TOKEN })));
    expect(responses.map((response) => response.status)).toEqual([200, 200, 200]);
    expect(new Set(database.batches.map((batch) => batch[0].values[0])).size).toBe(3);
    expect(database.state.size).toBe(0);
  });

  it('reports corrupted readback as failure even if cleanup succeeded', async () => {
    const database = fakeDatabase({ corruptReadback: true });
    const response = await handleSyntheticDiagnosticRequest(request(), { DB: database.db, ADMIN_EXPORT_TOKEN: TOKEN });
    expect(response.status).toBe(503);
    expect(database.state.size).toBe(0);
  });

  it.each([1, 2, 3])('reports transaction failure at statement %s without leaking details or keeping probes', async (failAt) => {
    const database = fakeDatabase({ failAt });
    const response = await handleSyntheticDiagnosticRequest(request(), { DB: database.db, ADMIN_EXPORT_TOKEN: TOKEN });
    expect(response.status).toBe(503);
    const text = await response.text();
    expect(text).not.toContain('secret-database-detail');
    expect(text).not.toContain(TOKEN);
    expect(database.state.size).toBe(0);
  });
});
