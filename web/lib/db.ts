import type {
  FeedbackRecord,
  ResearchBatch,
  ResearchBatchResponse,
  ResearchEvent,
  ResearchEventWriteResult,
  SessionRecord,
} from './research-types';

const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  anonymous_user_id TEXT NOT NULL,
  consent_version TEXT NOT NULL,
  consented_at INTEGER NOT NULL,
  started_at INTEGER NOT NULL,
  ended_at INTEGER,
  client_version TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  summary_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  sequence_number INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  occurred_at INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  model_hash TEXT,
  route_trace TEXT,
  probabilities_json TEXT,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_session_sequence
  ON events(session_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at);
CREATE TABLE IF NOT EXISTS feedback (
  feedback_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL UNIQUE,
  session_id TEXT NOT NULL,
  utterance TEXT NOT NULL,
  route TEXT NOT NULL,
  top_label TEXT NOT NULL,
  low_confidence INTEGER NOT NULL,
  probabilities_json TEXT NOT NULL,
  phrases_json TEXT NOT NULL,
  model_hash TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  route_trace TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (event_id) REFERENCES events(event_id),
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_top_label ON feedback(top_label);
CREATE TABLE IF NOT EXISTS research_session_tokens (
  token_hash TEXT PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE,
  anonymous_user_id TEXT NOT NULL,
  consent_version TEXT NOT NULL,
  consented_at INTEGER NOT NULL,
  issued_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  revoked_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_research_session_tokens_expiry
  ON research_session_tokens(expires_at);
CREATE TABLE IF NOT EXISTS research_rate_limits (
  bucket_key TEXT NOT NULL,
  window_start INTEGER NOT NULL,
  request_count INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  PRIMARY KEY (bucket_key, window_start)
);
CREATE INDEX IF NOT EXISTS idx_research_rate_limits_expiry
  ON research_rate_limits(expires_at);
CREATE TABLE IF NOT EXISTS research_security_secrets (
  secret_name TEXT PRIMARY KEY,
  secret_value TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
`;

const ready = new WeakMap<object, Promise<void>>();

export function initializeDatabase(db: D1Database): Promise<void> {
  const cached = ready.get(db as object);
  if (cached) return cached;
  // D1's exec() treats newlines as statement separators, so multiline CREATE
  // statements fail with "incomplete input". Prepare complete semicolon-delimited
  // statements and execute them as one batch instead.
  const schemaStatements = SCHEMA_SQL.split(';')
    .map((statement) => statement.trim())
    .filter(Boolean)
    .map((statement) => db.prepare(statement));
  const promise = db.batch(schemaStatements).then(() => undefined);
  ready.set(db as object, promise);
  promise.catch(() => ready.delete(db as object));
  return promise;
}

export class ResearchSessionConflictError extends Error {
  constructor() {
    super('session_id already exists with different session metadata');
    this.name = 'ResearchSessionConflictError';
  }
}

interface SessionRow {
  session_id: string;
  anonymous_user_id: string;
  consent_version: string;
  consented_at: number;
  started_at: number;
  ended_at: number | null;
  client_version: string;
  schema_version: string;
  summary_json: string | null;
}

export interface EventRow {
  event_id: string;
  session_id: string;
  sequence_number: number;
  event_type: string;
  occurred_at: number;
  payload_json: string;
  schema_version: string;
  model_hash: string | null;
  route_trace: string | null;
  probabilities_json: string | null;
  created_at: number;
}

export interface FeedbackRow {
  feedback_id: string;
  event_id: string;
  session_id: string;
  utterance: string;
  route: string;
  top_label: string;
  low_confidence: number;
  probabilities_json: string;
  phrases_json: string;
  model_hash: string;
  schema_version: string;
  route_trace: string;
  created_at: number;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
      .map(([key, nested]) => `${JSON.stringify(key)}:${canonicalJson(nested)}`);
    return `{${entries.join(',')}}`;
  }
  return JSON.stringify(value);
}

function storedJsonMatches(stored: string | null, incoming: unknown | undefined): boolean {
  if (stored === null) return incoming === undefined;
  if (incoming === undefined) return false;
  try {
    return canonicalJson(JSON.parse(stored)) === canonicalJson(incoming);
  } catch {
    return false;
  }
}

function sessionIdentityMatches(row: SessionRow, session: SessionRecord): boolean {
  return (
    row.session_id === session.sessionId &&
    row.anonymous_user_id === session.anonymousUserId &&
    row.consent_version === session.consentVersion &&
    row.consented_at === session.consentedAt &&
    row.started_at === session.startedAt &&
    row.client_version === session.clientVersion &&
    row.schema_version === session.schemaVersion
  );
}

async function existingSession(db: D1Database, sessionId: string): Promise<SessionRow | null> {
  return db
    .prepare('SELECT * FROM sessions WHERE session_id = ?')
    .bind(sessionId)
    .first<SessionRow>();
}

async function persistSession(db: D1Database, session: SessionRecord): Promise<void> {
  let row = await existingSession(db, session.sessionId);
  if (!row) {
    try {
      await db
        .prepare(
          `INSERT INTO sessions (
             session_id, anonymous_user_id, consent_version, consented_at,
             started_at, ended_at, client_version, schema_version, summary_json
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .bind(
          session.sessionId,
          session.anonymousUserId,
          session.consentVersion,
          session.consentedAt,
          session.startedAt,
          session.endedAt ?? null,
          session.clientVersion,
          session.schemaVersion,
          session.summary ? canonicalJson(session.summary) : null,
        )
        .run();
      return;
    } catch (error) {
      row = await existingSession(db, session.sessionId);
      if (!row) throw error;
    }
  }
  if (!sessionIdentityMatches(row, session)) throw new ResearchSessionConflictError();
  await db
    .prepare(
      `UPDATE sessions SET
         ended_at = COALESCE(ended_at, ?),
         summary_json = COALESCE(summary_json, ?)
       WHERE session_id = ?`,
    )
    .bind(
      session.endedAt ?? null,
      session.summary ? canonicalJson(session.summary) : null,
      session.sessionId,
    )
    .run();
}

function storedEventMatches(row: EventRow, event: ResearchEvent): boolean {
  return (
    row.event_id === event.eventId &&
    row.session_id === event.sessionId &&
    row.sequence_number === event.sequenceNumber &&
    row.event_type === event.eventType &&
    row.occurred_at === event.occurredAt &&
    storedJsonMatches(row.payload_json, event.payload) &&
    row.schema_version === event.schemaVersion &&
    row.model_hash === (event.modelHash ?? null) &&
    row.route_trace === (event.routeTrace ?? null) &&
    storedJsonMatches(row.probabilities_json, event.probabilities)
  );
}

function storedFeedbackMatches(
  row: FeedbackRow,
  event: ResearchEvent,
  feedback: FeedbackRecord,
): boolean {
  return (
    row.feedback_id === feedback.feedbackId &&
    row.event_id === event.eventId &&
    row.session_id === event.sessionId &&
    row.utterance === feedback.utterance &&
    row.route === feedback.route &&
    row.top_label === feedback.topLabel &&
    row.low_confidence === (feedback.lowConfidence ? 1 : 0) &&
    storedJsonMatches(row.probabilities_json, feedback.probabilities) &&
    storedJsonMatches(row.phrases_json, feedback.phrases) &&
    row.model_hash === feedback.modelHash &&
    row.schema_version === feedback.schemaVersion &&
    row.route_trace === feedback.routeTrace
  );
}

async function inspectExistingEvent(
  db: D1Database,
  event: ResearchEvent,
): Promise<ResearchEventWriteResult | null> {
  const eventRows = await db
    .prepare(
      `SELECT * FROM events
       WHERE event_id = ? OR (session_id = ? AND sequence_number = ?)
       LIMIT 2`,
    )
    .bind(event.eventId, event.sessionId, event.sequenceNumber)
    .all<EventRow>();
  let feedbackRows: FeedbackRow[] = [];
  if (event.feedback) {
    const feedbackResult = await db
      .prepare('SELECT * FROM feedback WHERE event_id = ? OR feedback_id = ? LIMIT 2')
      .bind(event.eventId, event.feedback.feedbackId)
      .all<FeedbackRow>();
    feedbackRows = feedbackResult.results;
  }
  return classifyExistingEvent(eventRows.results, feedbackRows, event);
}

export function classifyExistingEvent(
  eventRows: EventRow[],
  feedbackRows: FeedbackRow[],
  event: ResearchEvent,
): ResearchEventWriteResult | null {
  if (eventRows.length > 1) {
    return { eventId: event.eventId, status: 'conflict', reason: 'event-id-and-sequence-conflict' };
  }
  const existingEvent = eventRows[0];
  if (existingEvent) {
    if (!storedEventMatches(existingEvent, event)) {
      return { eventId: event.eventId, status: 'conflict', reason: 'event-payload-conflict' };
    }
    if (!event.feedback) return { eventId: event.eventId, status: 'duplicate' };
    if (
      feedbackRows.length !== 1 ||
      !storedFeedbackMatches(feedbackRows[0], event, event.feedback)
    ) {
      return { eventId: event.eventId, status: 'conflict', reason: 'feedback-conflict' };
    }
    return { eventId: event.eventId, status: 'duplicate' };
  }

  if (event.feedback && feedbackRows.length > 0) {
    return { eventId: event.eventId, status: 'conflict', reason: 'feedback-id-conflict' };
  }
  return null;
}

async function insertEvent(
  db: D1Database,
  event: ResearchEvent,
  now: number,
): Promise<void> {
  const statements: D1PreparedStatement[] = [
    db
      .prepare(
        `INSERT INTO events (
           event_id, session_id, sequence_number, event_type, occurred_at,
           payload_json, schema_version, model_hash, route_trace,
           probabilities_json, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .bind(
        event.eventId,
        event.sessionId,
        event.sequenceNumber,
        event.eventType,
        event.occurredAt,
        canonicalJson(event.payload),
        event.schemaVersion,
        event.modelHash ?? null,
        event.routeTrace ?? null,
        event.probabilities ? canonicalJson(event.probabilities) : null,
        now,
      ),
  ];
  if (event.feedback) {
    const feedback = event.feedback;
    statements.push(
      db
        .prepare(
          `INSERT INTO feedback (
             feedback_id, event_id, session_id, utterance, route, top_label,
             low_confidence, probabilities_json, phrases_json, model_hash,
             schema_version, route_trace, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .bind(
          feedback.feedbackId,
          event.eventId,
          event.sessionId,
          feedback.utterance,
          feedback.route,
          feedback.topLabel,
          feedback.lowConfidence ? 1 : 0,
          canonicalJson(feedback.probabilities),
          canonicalJson(feedback.phrases),
          feedback.modelHash,
          feedback.schemaVersion,
          feedback.routeTrace,
          now,
        ),
    );
  }
  await db.batch(statements);
}

export async function persistResearchBatch(
  db: D1Database,
  batch: ResearchBatch,
): Promise<ResearchBatchResponse> {
  await initializeDatabase(db);
  await persistSession(db, batch.session);
  const now = Date.now();
  const events: ResearchEventWriteResult[] = [];

  for (const event of batch.events) {
    const existing = await inspectExistingEvent(db, event);
    if (existing) {
      events.push(existing);
      continue;
    }
    try {
      await insertEvent(db, event, now);
      events.push({ eventId: event.eventId, status: 'accepted' });
    } catch (error) {
      // A concurrent retry may have won after the first read. Re-read so that
      // an idempotent duplicate and a true payload collision are never conflated.
      const raced = await inspectExistingEvent(db, event);
      if (!raced) throw error;
      events.push(raced);
    }
  }

  return {
    ok: true,
    accepted: events.filter((event) => event.status === 'accepted').length,
    duplicate: events.filter((event) => event.status === 'duplicate').length,
    conflict: events.filter((event) => event.status === 'conflict').length,
    events,
  };
}

function parseJson(value: string | null): unknown {
  if (value === null) return null;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

export async function exportTrainingJsonl(db: D1Database): Promise<string> {
  await initializeDatabase(db);
  const [sessionResult, eventResult, feedbackResult] = await Promise.all([
    db.prepare('SELECT * FROM sessions ORDER BY started_at, session_id').all<SessionRow>(),
    db.prepare('SELECT * FROM events ORDER BY occurred_at, session_id, sequence_number').all<EventRow>(),
    db.prepare('SELECT * FROM feedback ORDER BY created_at, feedback_id').all<FeedbackRow>(),
  ]);

  const lines: string[] = [];
  for (const row of sessionResult.results) {
    lines.push(
      JSON.stringify({
        recordType: 'session',
        ...row,
        summary: parseJson(row.summary_json),
        summary_json: undefined,
      }),
    );
  }
  for (const row of eventResult.results) {
    lines.push(
      JSON.stringify({
        recordType: 'event',
        ...row,
        payload: parseJson(row.payload_json),
        probabilities: parseJson(row.probabilities_json),
        payload_json: undefined,
        probabilities_json: undefined,
      }),
    );
  }
  for (const row of feedbackResult.results) {
    lines.push(
      JSON.stringify({
        recordType: 'feedback',
        ...row,
        low_confidence: Boolean(row.low_confidence),
        probabilities: parseJson(row.probabilities_json),
        phrases: parseJson(row.phrases_json),
        probabilities_json: undefined,
        phrases_json: undefined,
      }),
    );
  }
  return lines.length === 0 ? '' : `${lines.join('\n')}\n`;
}
