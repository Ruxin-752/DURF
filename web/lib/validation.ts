import {
  CONSENT_VERSION,
  type FeedbackLabel,
  type FeedbackProbabilities,
  type FeedbackRecord,
  type FeedbackRoute,
  type PhraseResearchPrediction,
  type ResearchBatch,
  type ResearchEvent,
  type ResearchEventType,
  type ResearchSessionGrantRequest,
  type SessionRecord,
} from './research-types';

export const MAX_REQUEST_BYTES = 256 * 1024;
export const MAX_SESSION_REQUEST_BYTES = 8 * 1024;
export const MAX_BATCH_EVENTS = 100;
export const MAX_UTTERANCE_LENGTH = 500;
export const MAX_CONSENT_AGE_MS = 5 * 60 * 1000;

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SAFE_VERSION_PATTERN = /^[a-zA-Z0-9._:+-]{1,128}$/;
const LABELS = new Set<FeedbackLabel>(['Evaluative', 'Imperative', 'Descriptive']);
const ROUTES = new Set<FeedbackRoute>(['route1', 'route2']);
const EVENT_TYPES = new Set<ResearchEventType>([
  'session_start',
  'move',
  'interact',
  'pause',
  'resume',
  'feedback',
  'restart',
  'session_end',
  'tick_summary',
]);
const FORBIDDEN_KEYS = new Set(['email', 'e-mail', 'ip', 'ip_address', 'ipaddress']);

export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function text(value: unknown, name: string, max: number): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > max) {
    throw new Error(`${name} 必须是 1-${max} 字符的字符串`);
  }
  return value;
}

function uuid(value: unknown, name: string): string {
  const parsed = text(value, name, 64);
  if (!UUID_PATTERN.test(parsed)) throw new Error(`${name} 不是有效 UUID`);
  return parsed;
}

function integer(value: unknown, name: string, min = 0): number {
  if (!Number.isSafeInteger(value) || (value as number) < min) {
    throw new Error(`${name} 必须是大于等于 ${min} 的整数`);
  }
  return value as number;
}

function timestamp(value: unknown, name: string, now = Date.now()): number {
  const parsed = integer(value, name, 1_500_000_000_000);
  if (parsed > now + 86_400_000) throw new Error(`${name} 超出允许时间范围`);
  return parsed;
}

function version(value: unknown, name: string): string {
  const parsed = text(value, name, 128);
  if (!SAFE_VERSION_PATTERN.test(parsed)) throw new Error(`${name} 包含非法字符`);
  return parsed;
}

function assertNoSensitiveKeys(value: unknown, path = 'payload', depth = 0): void {
  if (depth > 8) throw new Error(`${path} 嵌套过深`);
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertNoSensitiveKeys(item, `${path}[${index}]`, depth + 1));
    return;
  }
  if (!isRecord(value)) return;
  for (const [key, nested] of Object.entries(value)) {
    if (FORBIDDEN_KEYS.has(key.toLowerCase())) {
      throw new Error(`${path} 不允许包含 ${key}`);
    }
    assertNoSensitiveKeys(nested, `${path}.${key}`, depth + 1);
  }
}

function probabilities(value: unknown): FeedbackProbabilities {
  if (!isRecord(value)) throw new Error('probabilities 必须是对象');
  const result = {
    Evaluative: Number(value.Evaluative),
    Imperative: Number(value.Imperative),
    Descriptive: Number(value.Descriptive),
  };
  for (const [label, score] of Object.entries(result)) {
    if (!Number.isFinite(score) || score < 0 || score > 1) {
      throw new Error(`${label} 概率必须在 0 到 1 之间`);
    }
  }
  const sum = result.Evaluative + result.Imperative + result.Descriptive;
  if (Math.abs(sum - 1) > 0.02) throw new Error('三类概率之和必须接近 1');
  return result;
}

function phrase(value: unknown): PhraseResearchPrediction {
  if (!isRecord(value)) throw new Error('phrase prediction 必须是对象');
  const label = value.label as FeedbackLabel;
  if (!LABELS.has(label)) throw new Error('phrase label 非法');
  const confidence = Number(value.confidence);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new Error('phrase confidence 必须在 0 到 1 之间');
  }
  return {
    phrase: text(value.phrase, 'phrase', 240),
    label,
    confidence,
    probabilities: probabilities(value.probabilities),
    abstained: Boolean(value.abstained),
  };
}

function feedback(value: unknown): FeedbackRecord {
  if (!isRecord(value)) throw new Error('feedback 必须是对象');
  const route = value.route as FeedbackRoute;
  const topLabel = value.topLabel as FeedbackLabel;
  if (!ROUTES.has(route)) throw new Error('feedback route 非法');
  if (!LABELS.has(topLabel)) throw new Error('feedback topLabel 非法');
  if (!Array.isArray(value.phrases) || value.phrases.length < 1 || value.phrases.length > 12) {
    throw new Error('feedback phrases 必须包含 1-12 个短语');
  }
  return {
    feedbackId: uuid(value.feedbackId, 'feedbackId'),
    utterance: text(value.utterance, 'utterance', MAX_UTTERANCE_LENGTH),
    route,
    topLabel,
    lowConfidence: Boolean(value.lowConfidence),
    probabilities: probabilities(value.probabilities),
    phrases: value.phrases.map(phrase),
    modelHash: version(value.modelHash, 'modelHash'),
    schemaVersion: version(value.schemaVersion, 'feedback schemaVersion'),
    routeTrace: text(value.routeTrace, 'routeTrace', 512),
  };
}

function session(value: unknown): SessionRecord {
  if (!isRecord(value)) throw new Error('session 必须是对象');
  const summary = value.summary;
  if (summary !== undefined && !isRecord(summary)) throw new Error('summary 必须是对象');
  if (summary) assertNoSensitiveKeys(summary, 'session.summary');
  return {
    sessionId: uuid(value.sessionId, 'sessionId'),
    anonymousUserId: uuid(value.anonymousUserId, 'anonymousUserId'),
    consentVersion: version(value.consentVersion, 'consentVersion'),
    consentedAt: timestamp(value.consentedAt, 'consentedAt'),
    startedAt: timestamp(value.startedAt, 'startedAt'),
    endedAt: value.endedAt === undefined ? undefined : timestamp(value.endedAt, 'endedAt'),
    clientVersion: version(value.clientVersion, 'clientVersion'),
    schemaVersion: version(value.schemaVersion, 'session schemaVersion'),
    summary,
  };
}

function event(value: unknown, sessionId: string): ResearchEvent {
  if (!isRecord(value)) throw new Error('event 必须是对象');
  const eventType = value.eventType as ResearchEventType;
  if (!EVENT_TYPES.has(eventType)) throw new Error('eventType 非法');
  if (!isRecord(value.payload)) throw new Error('event payload 必须是对象');
  assertNoSensitiveKeys(value.payload);
  const payloadJson = JSON.stringify(value.payload);
  if (payloadJson.length > 16_384) throw new Error('event payload 过大');
  const parsedSessionId = uuid(value.sessionId, 'event sessionId');
  if (parsedSessionId !== sessionId) throw new Error('event sessionId 与批次不一致');
  const modelHash =
    value.modelHash === undefined ? undefined : version(value.modelHash, 'event modelHash');
  const routeTrace =
    value.routeTrace === undefined ? undefined : text(value.routeTrace, 'event routeTrace', 512);
  const parsedFeedback = value.feedback === undefined ? undefined : feedback(value.feedback);
  if (eventType === 'feedback' && !parsedFeedback) throw new Error('feedback 事件缺少反馈记录');
  return {
    eventId: uuid(value.eventId, 'eventId'),
    sessionId: parsedSessionId,
    sequenceNumber: integer(value.sequenceNumber, 'sequenceNumber'),
    eventType,
    occurredAt: timestamp(value.occurredAt, 'occurredAt'),
    payload: value.payload,
    schemaVersion: version(value.schemaVersion, 'event schemaVersion'),
    modelHash,
    routeTrace,
    probabilities: value.probabilities === undefined ? undefined : probabilities(value.probabilities),
    feedback: parsedFeedback,
  };
}

export function parseResearchBatch(value: unknown): ValidationResult<ResearchBatch> {
  try {
    if (!isRecord(value)) throw new Error('请求体必须是对象');
    const parsedSession = session(value.session);
    if (!Array.isArray(value.events) || value.events.length < 1) {
      throw new Error('events 不能为空');
    }
    if (value.events.length > MAX_BATCH_EVENTS) {
      throw new Error(`每批最多 ${MAX_BATCH_EVENTS} 个事件`);
    }
    const events = value.events.map((item) => event(item, parsedSession.sessionId));
    const sequenceNumbers = new Set(events.map((item) => item.sequenceNumber));
    if (sequenceNumbers.size !== events.length) throw new Error('批次内 sequenceNumber 重复');
    const eventIds = new Set(events.map((item) => item.eventId));
    if (eventIds.size !== events.length) throw new Error('批次内 eventId 重复');
    return { ok: true, value: { session: parsedSession, events } };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : '请求无效' };
  }
}

export function parseResearchSessionGrant(
  value: unknown,
  now = Date.now(),
): ValidationResult<ResearchSessionGrantRequest> {
  try {
    if (!isRecord(value)) throw new Error('请求体必须是对象');
    const consentVersion = version(value.consentVersion, 'consentVersion');
    if (consentVersion !== CONSENT_VERSION) throw new Error('同意书版本已更新，请重新确认');
    const consentedAt = timestamp(value.consentedAt, 'consentedAt', now);
    if (consentedAt < now - MAX_CONSENT_AGE_MS || consentedAt > now + 60_000) {
      throw new Error('同意时间无效，请重新确认');
    }
    return {
      ok: true,
      value: {
        sessionId: uuid(value.sessionId, 'sessionId'),
        anonymousUserId: uuid(value.anonymousUserId, 'anonymousUserId'),
        consentVersion,
        consentedAt,
      },
    };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : '请求无效' };
  }
}
