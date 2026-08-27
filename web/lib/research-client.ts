'use client';

import {
  CLIENT_VERSION,
  CONSENT_VERSION,
  SCHEMA_VERSION,
  type FeedbackProbabilities,
  type FeedbackRecord,
  type ResearchBatch,
  type ResearchBatchResponse,
  type ResearchEvent,
  type ResearchEventType,
  type ResearchSessionGrantRequest,
  type SessionRecord,
} from './research-types';

const FLUSH_INTERVAL_MS = 3_000;
const MAX_SEND_BATCH = 20;
const MOVE_SAMPLE_INTERVAL_MS = 250;

export interface QueueStatus {
  pending: number;
  state: 'idle' | 'syncing' | 'synced' | 'offline' | 'conflict';
  conflicts?: number;
}

interface EventMetadata {
  modelHash?: string;
  routeTrace?: string;
  probabilities?: FeedbackProbabilities;
  feedback?: FeedbackRecord;
}

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export class ResearchEventQueue {
  readonly sessionId: string;
  readonly anonymousUserId: string;
  private session: SessionRecord;
  private events: ResearchEvent[] = [];
  private sequence = 0;
  private lastMoveAt = 0;
  private flushing = false;
  private closed = false;
  private sessionEstablished = false;
  private sessionPromise: Promise<void> | null = null;
  private conflictCount = 0;
  private timer: ReturnType<typeof setInterval>;

  constructor(
    anonymousUserId: string,
    private readonly onStatus?: (status: QueueStatus) => void,
    private readonly fetcher: FetchLike = (input, init) => fetch(input, init),
  ) {
    const now = Date.now();
    this.sessionId = crypto.randomUUID();
    this.anonymousUserId = anonymousUserId;
    this.session = {
      sessionId: this.sessionId,
      anonymousUserId,
      consentVersion: CONSENT_VERSION,
      consentedAt: now,
      startedAt: now,
      clientVersion: CLIENT_VERSION,
      schemaVersion: SCHEMA_VERSION,
    };
    this.timer = setInterval(() => void this.flush(), FLUSH_INTERVAL_MS);
    this.enqueue('session_start', {
      consentVersion: CONSENT_VERSION,
      privacyMode: 'anonymous-no-raw-ip-no-email',
    });
    void this.ensureResearchSession().catch(() => this.report('offline'));
  }

  get pendingCount(): number {
    return this.events.length;
  }

  enqueue(
    eventType: ResearchEventType,
    payload: Record<string, unknown>,
    metadata: EventMetadata = {},
  ): string | null {
    if (this.closed && eventType !== 'session_end') return null;
    const now = Date.now();
    if (eventType === 'move' && now - this.lastMoveAt < MOVE_SAMPLE_INTERVAL_MS) return null;
    if (eventType === 'move') this.lastMoveAt = now;
    const eventId = crypto.randomUUID();
    this.events.push({
      eventId,
      sessionId: this.sessionId,
      sequenceNumber: this.sequence,
      eventType,
      occurredAt: now,
      payload,
      schemaVersion: SCHEMA_VERSION,
      ...metadata,
    });
    this.sequence += 1;
    this.report('idle');
    if (this.events.length >= MAX_SEND_BATCH) void this.flush();
    return eventId;
  }

  end(summary: Record<string, unknown>): void {
    if (this.closed) return;
    this.session = { ...this.session, endedAt: Date.now(), summary };
    this.enqueue('session_end', summary);
    this.closed = true;
    clearInterval(this.timer);
    void this.flush();
  }

  async flush(): Promise<void> {
    if (this.flushing || this.events.length === 0) return;
    this.flushing = true;
    this.report('syncing');
    const sending = this.events.slice(0, MAX_SEND_BATCH);
    const body: ResearchBatch = { session: this.session, events: sending };
    try {
      await this.ensureResearchSession();
      let response = await this.postBatch(body);
      if (response.status === 401) {
        await this.ensureResearchSession(true);
        response = await this.postBatch(body);
      }
      if (!response.ok) throw new Error(`research API returned ${response.status}`);
      const result = (await response.json()) as Partial<ResearchBatchResponse>;
      if (!Array.isArray(result.events) || result.events.length !== sending.length) {
        throw new Error('research API returned an incomplete event result');
      }
      const expectedIds = new Set(sending.map((event) => event.eventId));
      const returnedIds = new Set<string>();
      let conflicts = 0;
      for (const eventResult of result.events) {
        if (
          !eventResult ||
          typeof eventResult.eventId !== 'string' ||
          !expectedIds.has(eventResult.eventId) ||
          returnedIds.has(eventResult.eventId) ||
          !['accepted', 'duplicate', 'conflict'].includes(eventResult.status)
        ) {
          throw new Error('research API returned an invalid event result');
        }
        returnedIds.add(eventResult.eventId);
        if (eventResult.status === 'conflict') conflicts += 1;
      }
      this.events.splice(0, sending.length);
      if (conflicts > 0) {
        this.conflictCount += conflicts;
        console.warn(`research API rejected ${conflicts} conflicting event(s)`);
        this.report('conflict');
      } else {
        this.report(this.events.length === 0 ? 'synced' : 'idle');
      }
      if (this.events.length > 0) queueMicrotask(() => void this.flush());
    } catch {
      this.report('offline');
    } finally {
      this.flushing = false;
    }
  }

  flushWithBeacon(): void {
    if (
      !this.sessionEstablished ||
      this.events.length === 0 ||
      typeof navigator === 'undefined' ||
      !navigator.sendBeacon
    ) {
      return;
    }
    const body: ResearchBatch = {
      session: this.session,
      events: this.events.slice(0, MAX_SEND_BATCH),
    };
    navigator.sendBeacon(
      '/api/research/batch',
      new Blob([JSON.stringify(body)], { type: 'application/json' }),
    );
  }

  private report(state: QueueStatus['state']): void {
    this.onStatus?.({ pending: this.events.length, state, conflicts: this.conflictCount });
  }

  private async ensureResearchSession(force = false): Promise<void> {
    if (force) {
      this.sessionEstablished = false;
      this.sessionPromise = null;
    }
    if (this.sessionEstablished) return;
    if (!this.sessionPromise) {
      const grant: ResearchSessionGrantRequest = {
        sessionId: this.session.sessionId,
        anonymousUserId: this.session.anonymousUserId,
        consentVersion: this.session.consentVersion,
        consentedAt: this.session.consentedAt,
      };
      this.sessionPromise = this.fetcher('/api/research/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(grant),
      }).then(async (response) => {
        if (!response.ok) throw new Error(`research session API returned ${response.status}`);
        const result = (await response.json()) as {
          ok?: boolean;
          consentVersion?: string;
          expiresAt?: number;
        };
        if (
          result.ok !== true ||
          result.consentVersion !== CONSENT_VERSION ||
          !Number.isSafeInteger(result.expiresAt)
        ) {
          throw new Error('research session API returned an invalid grant');
        }
        this.sessionEstablished = true;
      });
    }
    try {
      await this.sessionPromise;
    } catch (error) {
      this.sessionPromise = null;
      throw error;
    }
  }

  private postBatch(body: ResearchBatch): Promise<Response> {
    return this.fetcher('/api/research/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body),
      keepalive: true,
    });
  }
}

export function getOrCreateAnonymousUserId(): string {
  const key = 'durf-anonymous-user-v1';
  const existing = localStorage.getItem(key);
  if (
    existing &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu.test(
      existing,
    )
  ) {
    return existing;
  }
  const created = crypto.randomUUID();
  localStorage.setItem(key, created);
  return created;
}
