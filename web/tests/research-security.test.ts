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
  type ResearchBatch,
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
    expect(parsed).toMatchObject({ ok: false, error: 'Duplicate eventId in batch' });
  });

  it('round-trips optional raw-score semantics on each feedback phrase', () => {
    const now = Date.now();
    const probabilities = { Evaluative: 0.1, Imperative: 0.8, Descriptive: 0.1 };
    const event: ResearchEvent = {
      ...sampleEvent(now),
      eventType: 'feedback',
      probabilities,
      feedback: {
        feedbackId: '66666666-6666-4666-8666-666666666666',
        utterance: 'Take it.',
        route: 'route1',
        topLabel: 'Imperative',
        lowConfidence: false,
        probabilities,
        phrases: [
          {
            phrase: 'Take it.',
            label: 'Imperative',
            confidence: 0.8,
            probabilities,
            abstained: false,
            scoreKind: 'raw_model_softmax_score',
            thresholdPolicy:
              'existing_web_preview_policy_not_validated_in_raw_score_space',
          },
        ],
        modelHash: 'model-v1',
        schemaVersion: SCHEMA_VERSION,
        routeTrace: 'route1',
      },
    };
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
      events: [event],
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) throw new Error(parsed.error);
    expect(parsed.value.events[0].feedback?.phrases[0]).toMatchObject({
      scoreKind: 'raw_model_softmax_score',
      thresholdPolicy: 'existing_web_preview_policy_not_validated_in_raw_score_space',
    });
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
  function syntheticQueueServer() {
    let online = true;
    const batches: Array<{ body: string; batch: ResearchBatch; keepalive?: boolean }> = [];
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (!online) return new Response(null, { status: 503 });
      if (String(input).endsWith('/session')) {
        return Response.json(
          { ok: true, consentVersion: CONSENT_VERSION, expiresAt: Date.now() + 60_000 },
          { status: 201 },
        );
      }
      const body = String(init?.body);
      const batch = JSON.parse(body) as ResearchBatch;
      batches.push({ body, batch, keepalive: init?.keepalive });
      return Response.json({
        ok: true,
        accepted: batch.events.length,
        duplicate: 0,
        conflict: 0,
        events: batch.events.map(({ eventId }) => ({ eventId, status: 'accepted' })),
      });
    });
    return { fetcher, batches, setOnline: (value: boolean) => { online = value; } };
  }

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('drains an offline multibyte backlog in byte-bounded batches without losing event order', async () => {
    vi.useFakeTimers();
    const server = syntheticQueueServer();
    const queue = new ResearchEventQueue(USER_ID, undefined, server.fetcher);
    await queue.flush();
    const expectedIds = [server.batches[0].batch.events[0].eventId];
    const syntheticNote = '厨房🍲'.repeat(1_800);

    server.setOnline(false);
    for (let index = 0; index < 45; index += 1) {
      const eventId = queue.enqueue('restart', { index, syntheticNote });
      expect(eventId).not.toBeNull();
      expectedIds.push(eventId!);
    }
    await vi.advanceTimersByTimeAsync(0);
    expect(queue.pendingCount).toBe(45);
    expect(server.batches).toHaveLength(1);

    server.setOnline(true);
    await queue.flush();
    await vi.advanceTimersByTimeAsync(0);
    expect(queue.pendingCount).toBe(0);
    const recovered = server.batches.slice(1);
    expect(recovered.length).toBeGreaterThan(3);
    const encoder = new TextEncoder();
    for (const { body, batch, keepalive } of server.batches) {
      const bytes = encoder.encode(body).byteLength;
      expect(bytes).toBeLessThanOrEqual(256 * 1024);
      expect(batch.events.length).toBeLessThanOrEqual(20);
      expect(parseResearchBatch(batch)).toMatchObject({ ok: true });
      expect(keepalive).toBe(bytes <= 60 * 1024);
    }
    expect(recovered.some(({ keepalive }) => keepalive === false)).toBe(true);

    const events = server.batches.flatMap(({ batch }) => batch.events);
    expect(events.map(({ eventId }) => eventId)).toEqual(expectedIds);
    expect(new Set(events.map(({ eventId }) => eventId)).size).toBe(46);
    expect(events.map(({ sequenceNumber }) => sequenceNumber)).toEqual(
      Array.from({ length: 46 }, (_, index) => index),
    );
    expect(events.slice(1).map(({ payload }) => payload)).toEqual(
      Array.from({ length: 45 }, (_, index) => ({ index, syntheticNote })),
    );
    // This fixture would pass a character-count check while exceeding the wire limit.
    const unsplitBody = JSON.stringify({
      session: recovered[0].batch.session,
      events: events.slice(1, 21),
    });
    expect(unsplitBody.length).toBeLessThan(256 * 1024);
    expect(encoder.encode(unsplitBody).byteLength).toBeGreaterThan(256 * 1024);
  });

  it('bounds multibyte beacons to 60 KiB and retains every pending event for acknowledged upload', async () => {
    vi.useFakeTimers();
    const server = syntheticQueueServer();
    const queue = new ResearchEventQueue(USER_ID, undefined, server.fetcher);
    await queue.flush();
    server.setOnline(false);
    const expectedIds = Array.from({ length: 8 }, (_, index) =>
      queue.enqueue('restart', { index, syntheticNote: '厨房🍲'.repeat(1_800) }),
    );
    const sendBeacon = vi.fn((_url: string, _body: Blob) => true);
    vi.stubGlobal('navigator', { sendBeacon });

    queue.flushWithBeacon();
    expect(sendBeacon).toHaveBeenCalledOnce();
    expect(queue.pendingCount).toBe(8);
    const [url, blob] = sendBeacon.mock.calls[0];
    expect(url).toBe('/api/research/batch');
    expect(blob.type).toBe('application/json');
    expect(blob.size).toBeLessThanOrEqual(60 * 1024);
    const body = await blob.text();
    expect(blob.size).toBeGreaterThan(body.length);
    const beaconBatch = JSON.parse(body) as ResearchBatch;
    expect(parseResearchBatch(beaconBatch)).toMatchObject({ ok: true });
    expect(beaconBatch.events.length).toBeGreaterThan(1);
    expect(beaconBatch.events.length).toBeLessThan(8);
    expect(beaconBatch.events.map(({ eventId }) => eventId)).toEqual(
      expectedIds.slice(0, beaconBatch.events.length),
    );
    expect(beaconBatch.events.map(({ sequenceNumber }) => sequenceNumber)).toEqual(
      Array.from({ length: beaconBatch.events.length }, (_, index) => index + 1),
    );

    sendBeacon.mockReturnValueOnce(false);
    queue.flushWithBeacon();
    expect(await sendBeacon.mock.calls[1][1].text()).toBe(body);
    expect(queue.pendingCount).toBe(8);

    server.setOnline(true);
    await queue.flush();
    await vi.advanceTimersByTimeAsync(0);
    expect(queue.pendingCount).toBe(0);
    const uploaded = server.batches.slice(1).flatMap(({ batch }) => batch.events);
    expect(uploaded.map(({ eventId }) => eventId)).toEqual(expectedIds);
    expect(uploaded.map(({ sequenceNumber }) => sequenceNumber)).toEqual(
      Array.from({ length: 8 }, (_, index) => index + 1),
    );
  });

  it('keeps keepalive enabled for small acknowledged batches', async () => {
    vi.useFakeTimers();
    const server = syntheticQueueServer();
    const queue = new ResearchEventQueue(USER_ID, undefined, server.fetcher);
    await queue.flush();
    queue.enqueue('restart', { syntheticNote: 'A small synthetic event.' });
    await queue.flush();

    expect(queue.pendingCount).toBe(0);
    expect(server.batches).toHaveLength(2);
    for (const { body, keepalive } of server.batches) {
      expect(new TextEncoder().encode(body).byteLength).toBeLessThan(60 * 1024);
      expect(keepalive).toBe(true);
    }
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

  it('records the consent click time and reuses one page session across a restart', async () => {
    vi.useFakeTimers();
    const consentedAt = Date.now() - 1_000;
    const sessionBodies: Array<{ consentedAt: number }> = [];
    const batchEventTypes: string[][] = [];
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = JSON.parse(String(init?.body)) as {
        consentedAt?: number;
        events?: Array<{ eventId: string; eventType: string }>;
      };
      if (url.endsWith('/session')) {
        sessionBodies.push({ consentedAt: Number(body.consentedAt) });
        return Response.json(
          { ok: true, consentVersion: CONSENT_VERSION, expiresAt: Date.now() + 60_000 },
          { status: 201 },
        );
      }
      const events = body.events ?? [];
      batchEventTypes.push(events.map((event) => event.eventType));
      return Response.json({
        ok: true,
        accepted: events.length,
        duplicate: 0,
        conflict: 0,
        events: events.map((event) => ({ eventId: event.eventId, status: 'accepted' })),
      });
    });
    const queue = new ResearchEventQueue(USER_ID, undefined, fetcher, consentedAt);

    await queue.flush();
    queue.enqueue('restart', { score: 0 });
    await queue.flush();

    expect(sessionBodies).toEqual([{ consentedAt }]);
    expect(batchEventTypes).toEqual([['session_start'], ['restart']]);
    expect(fetcher.mock.calls.filter(([url]) => String(url).endsWith('/session'))).toHaveLength(1);
    vi.clearAllTimers();
  });
});
