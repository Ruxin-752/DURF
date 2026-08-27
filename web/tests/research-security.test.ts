import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  classifyExistingEvent,
  type EventRow,
  type FeedbackRow,
} from '../lib/db';
import { ResearchEventQueue } from '../lib/research-client';
import {
  buildResearchSessionCookie,
  consumeResearchRateLimit,
  issueResearchSessionToken,
  RESEARCH_SESSION_COOKIE,
  RESEARCH_SESSION_TTL_MS,
  researchRateLimitKey,
  sha256Hex,
  validateStoredResearchSession,
  verifyResearchSessionToken,
  type ResearchSessionTokenRow,
} from '../lib/research-security';
import { readBoundedJson, validateResearchPostHeaders } from '../lib/research-http';
import {
  CLIENT_VERSION,
  CONSENT_VERSION,
  SCHEMA_VERSION,
  type ResearchEvent,
} from '../lib/research-types';
import { parseResearchBatch, parseResearchSessionGrant } from '../lib/validation';

const SESSION_ID = '11111111-1111-4111-8111-111111111111';
const USER_ID = '22222222-2222-4222-8222-222222222222';
const EVENT_ID = '33333333-3333-4333-8333-333333333333';

function researchRequest(origin = 'https://study.example'): Request {
  return new Request(`${origin}/api/research/batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      Origin: origin,
      'Sec-Fetch-Site': 'same-origin',
    },
    body: '{}',
  });
}

function sampleEvent(now: number): ResearchEvent {
  return {
    eventId: EVENT_ID,
    sessionId: SESSION_ID,
    sequenceNumber: 0,
    eventType: 'move',
    occurredAt: now,
    payload: { y: 2, x: 1 },
    schemaVersion: SCHEMA_VERSION,
  };
}

function storedEvent(event: ResearchEvent): EventRow {
  return {
    event_id: event.eventId,
    session_id: event.sessionId,
    sequence_number: event.sequenceNumber,
    event_type: event.eventType,
    occurred_at: event.occurredAt,
    payload_json: '{"x":1,"y":2}',
    schema_version: event.schemaVersion,
    model_hash: null,
    route_trace: null,
    probabilities_json: null,
    created_at: event.occurredAt,
  };
}

class SecurityStatement {
  private arguments: unknown[] = [];

  constructor(
    private readonly owner: SecurityDatabase,
    private readonly sql: string,
  ) {}

  bind(...values: unknown[]): D1PreparedStatement {
    this.arguments = values;
    return this as unknown as D1PreparedStatement;
  }

  async run(): Promise<D1Result> {
    if (this.sql.includes('INSERT INTO research_security_secrets')) {
      this.owner.pepper ??= String(this.arguments[0]);
    } else if (this.sql.includes('INSERT INTO research_session_tokens')) {
      const [tokenHash, sessionId, anonymousUserId, consentVersion, consentedAt, issuedAt, expiresAt] =
        this.arguments;
      for (const [hash, row] of this.owner.tokens) {
        if (row.session_id === sessionId) this.owner.tokens.delete(hash);
      }
      this.owner.tokens.set(String(tokenHash), {
        token_hash: String(tokenHash),
        session_id: String(sessionId),
        anonymous_user_id: String(anonymousUserId),
        consent_version: String(consentVersion),
        consented_at: Number(consentedAt),
        issued_at: Number(issuedAt),
        expires_at: Number(expiresAt),
        revoked_at: null,
      });
    }
    return { success: true, meta: {} } as D1Result;
  }

  async first<T>(): Promise<T | null> {
    if (this.sql.includes('SELECT secret_value FROM research_security_secrets')) {
      return (this.owner.pepper ? { secret_value: this.owner.pepper } : null) as T | null;
    }
    if (this.sql.includes('INSERT INTO research_rate_limits')) {
      const [bucketKey, windowStart] = this.arguments;
      const key = `${String(bucketKey)}:${Number(windowStart)}`;
      const count = (this.owner.rateCounts.get(key) ?? 0) + 1;
      this.owner.rateCounts.set(key, count);
      return { request_count: count } as T;
    }
    if (this.sql.includes('FROM research_session_tokens WHERE token_hash')) {
      return (this.owner.tokens.get(String(this.arguments[0])) ?? null) as T | null;
    }
    return null;
  }
}

class SecurityDatabase {
  pepper: string | null = null;
  readonly rateCounts = new Map<string, number>();
  readonly tokens = new Map<string, ResearchSessionTokenRow>();

  async exec(): Promise<D1ExecResult> {
    return { count: 0, duration: 0 };
  }

  prepare(sql: string): D1PreparedStatement {
    return new SecurityStatement(this, sql) as unknown as D1PreparedStatement;
  }

  async batch<T = unknown>(statements: D1PreparedStatement[]): Promise<D1Result<T>[]> {
    const results: D1Result<T>[] = [];
    for (const statement of statements) {
      results.push((await statement.run<T>()) as D1Result<T>);
    }
    return results;
  }
}

describe('same-origin JSON request boundary', () => {
  it('accepts an application/json same-origin POST', () => {
    expect(validateResearchPostHeaders(researchRequest())).toEqual({ ok: true, value: undefined });
  });

  it('rejects missing or cross-origin Origin headers and non-JSON media', () => {
    const missing = new Request('https://study.example/api/research/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    expect(validateResearchPostHeaders(missing)).toMatchObject({ ok: false, status: 403 });

    const crossOrigin = researchRequest('https://attacker.example');
    expect(validateResearchPostHeaders(crossOrigin, 'https://study.example')).toMatchObject({
      ok: false,
      status: 403,
    });

    const text = new Request('https://study.example/api/research/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain', Origin: 'https://study.example' },
      body: '{}',
    });
    expect(validateResearchPostHeaders(text)).toMatchObject({ ok: false, status: 415 });
  });

  it('supports an explicit production SITE_ORIGIN without allowing a path', () => {
    const proxied = new Request('https://internal.invalid/api/research/batch', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Origin: 'https://study.example',
        'Sec-Fetch-Site': 'same-origin',
      },
      body: '{}',
    });
    expect(validateResearchPostHeaders(proxied, 'https://study.example')).toMatchObject({ ok: true });
    expect(validateResearchPostHeaders(proxied, 'https://study.example/path')).toMatchObject({
      ok: false,
      status: 503,
    });
  });

  it('stops reading a streaming body as soon as it exceeds the byte limit', async () => {
    const oversized = new Request('https://study.example/api/research/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Origin: 'https://study.example' },
      body: '{"payload":"too-large"}',
    });
    await expect(readBoundedJson(oversized, 8)).resolves.toMatchObject({
      ok: false,
      status: 413,
    });

    const valid = new Request('https://study.example/api/research/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Origin: 'https://study.example' },
      body: '{"ok":true}',
    });
    await expect(readBoundedJson(valid, 32)).resolves.toEqual({
      ok: true,
      value: { ok: true },
    });
  });
});

describe('current consent and batch identity validation', () => {
  it('accepts only a fresh grant for the current consent version', () => {
    const now = 1_800_000_000_000;
    expect(
      parseResearchSessionGrant(
        {
          sessionId: SESSION_ID,
          anonymousUserId: USER_ID,
          consentVersion: CONSENT_VERSION,
          consentedAt: now,
        },
        now,
      ).ok,
    ).toBe(true);
    expect(
      parseResearchSessionGrant(
        {
          sessionId: SESSION_ID,
          anonymousUserId: USER_ID,
          consentVersion: 'outdated-consent',
          consentedAt: now,
        },
        now,
      ).ok,
    ).toBe(false);
  });

  it('rejects duplicate event IDs inside one batch', () => {
    const now = Date.now();
    const event = sampleEvent(now);
    const parsed = parseResearchBatch({
      session: {
        sessionId: SESSION_ID,
        anonymousUserId: USER_ID,
        consentVersion: CONSENT_VERSION,
        consentedAt: now,
        startedAt: now,
        clientVersion: CLIENT_VERSION,
        schemaVersion: SCHEMA_VERSION,
      },
      events: [event, { ...event, sequenceNumber: 1 }],
    });
    expect(parsed).toMatchObject({ ok: false, error: '批次内 eventId 重复' });
  });
});

describe('opaque research session and durable rate limiting', () => {
  it('stores only a token hash and binds it to session + current consent', async () => {
    const now = Date.now();
    const database = new SecurityDatabase();
    const grant = {
      sessionId: SESSION_ID,
      anonymousUserId: USER_ID,
      consentVersion: CONSENT_VERSION,
      consentedAt: now,
    };
    const issued = await issueResearchSessionToken(database as unknown as D1Database, grant, now);
    const cookiePair = issued.cookie.split(';', 1)[0];
    const rawToken = cookiePair.slice(cookiePair.indexOf('=') + 1);
    expect(issued.cookie).toContain('HttpOnly');
    expect(issued.cookie).toContain('Secure');
    expect(issued.cookie).toContain('SameSite=Strict');
    expect(issued.cookie).toContain('Max-Age=7200');
    expect(database.tokens.has(rawToken)).toBe(false);
    expect(database.tokens.has(await sha256Hex(rawToken))).toBe(true);

    const request = new Request('https://study.example/api/research/batch', {
      headers: { Cookie: cookiePair },
    });
    await expect(
      verifyResearchSessionToken(database as unknown as D1Database, request, grant, now + 1),
    ).resolves.toMatchObject({ ok: true });
    await expect(
      verifyResearchSessionToken(
        database as unknown as D1Database,
        request,
        { ...grant, anonymousUserId: '44444444-4444-4444-8444-444444444444' },
        now + 1,
      ),
    ).resolves.toMatchObject({ ok: false });
  });

  it('rejects expired and outdated-consent token rows', () => {
    const now = Date.now();
    const row: ResearchSessionTokenRow = {
      token_hash: 'a'.repeat(64),
      session_id: SESSION_ID,
      anonymous_user_id: USER_ID,
      consent_version: CONSENT_VERSION,
      consented_at: now,
      issued_at: now,
      expires_at: now + RESEARCH_SESSION_TTL_MS,
      revoked_at: null,
    };
    const expected = {
      sessionId: SESSION_ID,
      anonymousUserId: USER_ID,
      consentVersion: CONSENT_VERSION,
      consentedAt: now,
    };
    expect(validateStoredResearchSession(row, expected, now + 1)).toMatchObject({ ok: true });
    expect(validateStoredResearchSession(row, expected, row.expires_at)).toMatchObject({ ok: false });
    expect(
      validateStoredResearchSession({ ...row, consent_version: 'old' }, expected, now + 1),
    ).toMatchObject({ ok: false });
  });

  it('uses a persistent server/IP-HMAC bucket without retaining the raw IP', async () => {
    const database = new SecurityDatabase();
    const request = new Request('https://study.example/api/research/batch', {
      headers: { 'CF-Connecting-IP': '203.0.113.9' },
    });
    const key = await researchRateLimitKey(
      database as unknown as D1Database,
      request,
      'batch',
    );
    expect(key).not.toContain('203.0.113.9');
    expect(key).toMatch(/^batch:[0-9a-f]{64}$/u);

    const first = await consumeResearchRateLimit(
      database as unknown as D1Database,
      key,
      2,
      60_000,
      1_800_000_000_000,
    );
    const second = await consumeResearchRateLimit(
      database as unknown as D1Database,
      key,
      2,
      60_000,
      1_800_000_000_001,
    );
    const third = await consumeResearchRateLimit(
      database as unknown as D1Database,
      key,
      2,
      60_000,
      1_800_000_000_002,
    );
    expect([first.allowed, second.allowed, third.allowed]).toEqual([true, true, false]);
  });

  it('builds the host-only cookie name explicitly', () => {
    expect(buildResearchSessionCookie('x'.repeat(43), Date.now() + 1_000)).toContain(
      `${RESEARCH_SESSION_COOKIE}=`,
    );
  });
});

describe('event idempotency is explicit', () => {
  it('distinguishes accepted candidates, exact duplicates, and payload conflicts', () => {
    const event = sampleEvent(Date.now());
    const row = storedEvent(event);
    expect(classifyExistingEvent([], [], event)).toBeNull();
    expect(classifyExistingEvent([row], [], event)).toEqual({
      eventId: EVENT_ID,
      status: 'duplicate',
    });
    expect(
      classifyExistingEvent([{ ...row, payload_json: '{"x":9,"y":2}' }], [], event),
    ).toMatchObject({ eventId: EVENT_ID, status: 'conflict', reason: 'event-payload-conflict' });
  });

  it('reports ID/sequence and feedback reuse as conflicts', () => {
    const event = sampleEvent(Date.now());
    const row = storedEvent(event);
    const otherRow = {
      ...row,
      event_id: '55555555-5555-4555-8555-555555555555',
    };
    expect(classifyExistingEvent([row, otherRow], [], event)).toMatchObject({
      status: 'conflict',
      reason: 'event-id-and-sequence-conflict',
    });
    const feedbackEvent: ResearchEvent = {
      ...event,
      eventType: 'feedback',
      feedback: {
        feedbackId: '66666666-6666-4666-8666-666666666666',
        utterance: 'Good route.',
        route: 'route1',
        topLabel: 'Evaluative',
        lowConfidence: false,
        probabilities: { Evaluative: 0.9, Imperative: 0.05, Descriptive: 0.05 },
        phrases: [
          {
            phrase: 'Good route.',
            label: 'Evaluative',
            confidence: 0.9,
            probabilities: { Evaluative: 0.9, Imperative: 0.05, Descriptive: 0.05 },
            abstained: false,
          },
        ],
        modelHash: 'model-v1',
        schemaVersion: SCHEMA_VERSION,
        routeTrace: 'route1',
      },
    };
    expect(
      classifyExistingEvent([], [{} as FeedbackRow], feedbackEvent),
    ).toMatchObject({ status: 'conflict', reason: 'feedback-id-conflict' });
  });
});

describe('browser queue session handshake and write statuses', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('obtains a server session before batching and surfaces permanent conflicts', async () => {
    vi.useFakeTimers();
    const calls: string[] = [];
    const statuses: Array<{ state: string; conflicts?: number }> = [];
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push(url);
      expect(init?.credentials).toBe('same-origin');
      if (url.endsWith('/session')) {
        return Response.json(
          { ok: true, consentVersion: CONSENT_VERSION, expiresAt: Date.now() + 60_000 },
          { status: 201 },
        );
      }
      const body = JSON.parse(String(init?.body)) as { events: Array<{ eventId: string }> };
      return Response.json({
        ok: true,
        accepted: 0,
        duplicate: 0,
        conflict: body.events.length,
        events: body.events.map((event) => ({ eventId: event.eventId, status: 'conflict' })),
      });
    });
    const queue = new ResearchEventQueue(
      USER_ID,
      (status) => statuses.push(status),
      fetcher,
    );
    await queue.flush();
    expect(calls).toEqual(['/api/research/session', '/api/research/batch']);
    expect(queue.pendingCount).toBe(0);
    expect(statuses.at(-1)).toMatchObject({ state: 'conflict', conflicts: 1 });
    expect(warning).toHaveBeenCalledOnce();
    vi.clearAllTimers();
  });
});
