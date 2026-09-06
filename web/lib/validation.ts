import {
  CONSENT_VERSION,
  TICK_SUMMARY_RECEIPT_SCHEMA_VERSION,
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
import { canonicalReceiptValue } from './policy-execution-receipt';
import { MAX_RESEARCH_REQUEST_BYTES, researchPayloadFits } from './research-limits';

export const MAX_REQUEST_BYTES = MAX_RESEARCH_REQUEST_BYTES;
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
const GAME_ACTIONS = new Set(['up', 'down', 'left', 'right', 'interact', 'stay']);
const AI_EXECUTION_OUTCOMES = new Set([
  'moved',
  'movement_blocked',
  'movement_collision',
  'interaction_succeeded',
  'interaction_failed',
  'stayed',
]);
const HELD_ITEMS = new Set(['tomato', 'onion', 'dish', 'soup']);
const POT_STAGES = new Set(['empty', 'filling', 'cooking', 'ready']);

export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function text(value: unknown, name: string, max: number): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > max) {
    throw new Error(`${name} must be a string with 1-${max} characters`);
  }
  return value;
}

function uuid(value: unknown, name: string): string {
  const parsed = text(value, name, 64);
  if (!UUID_PATTERN.test(parsed)) throw new Error(`${name} is not a valid UUID`);
  return parsed;
}

function integer(value: unknown, name: string, min = 0): number {
  if (!Number.isSafeInteger(value) || (value as number) < min) {
    throw new Error(`${name} must be an integer greater than or equal to ${min}`);
  }
  return value as number;
}

function timestamp(value: unknown, name: string, now = Date.now()): number {
  const parsed = integer(value, name, 1_500_000_000_000);
  if (parsed > now + 86_400_000) throw new Error(`${name} is outside the allowed time range`);
  return parsed;
}

function version(value: unknown, name: string): string {
  const parsed = text(value, name, 128);
  if (!SAFE_VERSION_PATTERN.test(parsed)) throw new Error(`${name} contains invalid characters`);
  return parsed;
}

function assertNoSensitiveKeys(value: unknown, path = 'payload', depth = 0): void {
  if (depth > 8) throw new Error(`${path} is nested too deeply`);
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertNoSensitiveKeys(item, `${path}[${index}]`, depth + 1));
    return;
  }
  if (!isRecord(value)) return;
  for (const [key, nested] of Object.entries(value)) {
    if (FORBIDDEN_KEYS.has(key.toLowerCase())) {
      throw new Error(`${path} must not contain ${key}`);
    }
    assertNoSensitiveKeys(nested, `${path}.${key}`, depth + 1);
  }
}

function booleanValue(value: unknown, name: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${name} must be a boolean`);
  return value;
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function nullableHeldItem(value: unknown, name: string): void {
  if (value !== null && (typeof value !== 'string' || !HELD_ITEMS.has(value))) {
    throw new Error(`${name} is invalid`);
  }
}

function pointOrNull(value: unknown, name: string): void {
  if (value === null) return;
  if (!isRecord(value)) throw new Error(`${name} must be a point or null`);
  integer(value.x, `${name}.x`);
  integer(value.y, `${name}.y`);
}

function soupOrNull(value: unknown, name: string): void {
  if (value === null) return;
  if (!isRecord(value)) throw new Error(`${name} must be soup contents or null`);
  integer(value.tomatoes, `${name}.tomatoes`);
  integer(value.onions, `${name}.onions`);
}

function aiObservableOutcome(value: unknown, name: string): void {
  if (!isRecord(value)) throw new Error(`${name} must be an object`);
  if (!isRecord(value.partner)) throw new Error(`${name}.partner must be an object`);
  integer(value.partner.x, `${name}.partner.x`);
  integer(value.partner.y, `${name}.partner.y`);
  version(value.partner.facing, `${name}.partner.facing`);
  nullableHeldItem(value.partner.held, `${name}.partner.held`);
  soupOrNull(value.partner.heldSoup, `${name}.partner.heldSoup`);

  if (!isRecord(value.task)) throw new Error(`${name}.task must be an object`);
  finiteNumber(value.task.score, `${name}.task.score`);
  integer(value.task.ordersCompleted, `${name}.task.ordersCompleted`);
  if (!isRecord(value.task.pot)) throw new Error(`${name}.task.pot must be an object`);
  if (typeof value.task.pot.stage !== 'string' || !POT_STAGES.has(value.task.pot.stage)) {
    throw new Error(`${name}.task.pot.stage is invalid`);
  }
  integer(value.task.pot.secondsRemaining, `${name}.task.pot.secondsRemaining`);
  integer(value.task.pot.tomatoes, `${name}.task.pot.tomatoes`);
  integer(value.task.pot.onions, `${name}.task.pot.onions`);
  if (!Array.isArray(value.task.counters)) {
    throw new Error(`${name}.task.counters must be an array`);
  }
  let previousCounterKey = '';
  value.task.counters.forEach((counter, index) => {
    const counterName = `${name}.task.counters[${index}]`;
    if (!isRecord(counter)) throw new Error(`${counterName} must be an object`);
    const key = text(counter.key, `${counterName}.key`, 32);
    if (index > 0 && key <= previousCounterKey) {
      throw new Error(`${name}.task.counters must have unique sorted keys`);
    }
    previousCounterKey = key;
    nullableHeldItem(counter.item, `${counterName}.item`);
    if (counter.item === null) throw new Error(`${counterName}.item must not be null`);
    soupOrNull(counter.soupContents, `${counterName}.soupContents`);
  });

  if (!Array.isArray(value.aiOrJointEvents)) {
    throw new Error(`${name}.aiOrJointEvents must be an array`);
  }
  value.aiOrJointEvents.forEach((item, index) => {
    const eventName = `${name}.aiOrJointEvents[${index}]`;
    if (!isRecord(item)) throw new Error(`${eventName} must be an object`);
    version(item.code, `${eventName}.code`);
    if (item.actor !== 'ai' && item.actor !== null) {
      throw new Error(`${eventName}.actor is invalid`);
    }
    nullableHeldItem(item.item, `${eventName}.item`);
    pointOrNull(item.position, `${eventName}.position`);
  });
}

function aiTickReceipt(value: unknown, name: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`${name} must be an object`);
  uuid(value.roundId, `${name}.roundId`);
  integer(value.policyRevision, `${name}.policyRevision`);
  const evaluatedAtTick = integer(value.evaluatedAtTick, `${name}.evaluatedAtTick`);
  const completedAtTick = integer(value.completedAtTick, `${name}.completedAtTick`);
  if (completedAtTick !== evaluatedAtTick + 1) {
    throw new Error(`${name} must cover exactly one game tick`);
  }
  version(value.proposedSubgoal, `${name}.proposedSubgoal`);
  if (typeof value.proposedAction !== 'string' || !GAME_ACTIONS.has(value.proposedAction)) {
    throw new Error(`${name}.proposedAction is invalid`);
  }
  if (typeof value.humanAction !== 'string' || !GAME_ACTIONS.has(value.humanAction)) {
    throw new Error(`${name}.humanAction is invalid`);
  }
  const actuallyExecuted = booleanValue(value.actuallyExecuted, `${name}.actuallyExecuted`);
  if (
    typeof value.executionOutcome !== 'string' ||
    !AI_EXECUTION_OUTCOMES.has(value.executionOutcome)
  ) {
    throw new Error(`${name}.executionOutcome is invalid`);
  }
  const outcomeMeansExecuted = ['moved', 'interaction_succeeded', 'stayed'].includes(
    value.executionOutcome,
  );
  if (actuallyExecuted !== outcomeMeansExecuted) {
    throw new Error(`${name}.actuallyExecuted contradicts its executionOutcome`);
  }
  return value;
}

function validateTickSummaryReceipts(payload: Record<string, unknown>): void {
  // Existing durf-web-event-v2 tick summaries predate this nested contract.
  // They remain valid when no receiptSchemaVersion is present.
  if (payload.receiptSchemaVersion === undefined) {
    if (
      payload.aiExecutionReceipt !== undefined ||
      payload.policyExecutionReceipt !== undefined
    ) {
      throw new Error('tick_summary receipts require receiptSchemaVersion');
    }
    return;
  }
  if (payload.receiptSchemaVersion !== TICK_SUMMARY_RECEIPT_SCHEMA_VERSION) {
    throw new Error('tick_summary receiptSchemaVersion is invalid');
  }
  const roundId = uuid(payload.roundId, 'tick_summary.roundId');
  const completedTick = integer(payload.tick, 'tick_summary.tick');
  if (!isRecord(payload.requestedJointActions)) {
    throw new Error('tick_summary.requestedJointActions must be an object');
  }
  if (
    typeof payload.requestedJointActions.ai !== 'string' ||
    !GAME_ACTIONS.has(payload.requestedJointActions.ai) ||
    typeof payload.requestedJointActions.human !== 'string' ||
    !GAME_ACTIONS.has(payload.requestedJointActions.human)
  ) {
    throw new Error('tick_summary.requestedJointActions is invalid');
  }
  const actual = aiTickReceipt(
    payload.aiExecutionReceipt,
    'tick_summary.aiExecutionReceipt',
  );
  if (
    actual.roundId !== roundId ||
    actual.completedAtTick !== completedTick ||
    actual.proposedAction !== payload.requestedJointActions.ai ||
    actual.humanAction !== payload.requestedJointActions.human
  ) {
    throw new Error('tick_summary ai receipt does not match its tick context');
  }
  if (payload.policyExecutionReceipt === undefined) return;
  const policy = aiTickReceipt(
    payload.policyExecutionReceipt,
    'tick_summary.policyExecutionReceipt',
  );
  uuid(policy.feedbackId, 'tick_summary.policyExecutionReceipt.feedbackId');
  version(
    policy.counterfactualSubgoal,
    'tick_summary.policyExecutionReceipt.counterfactualSubgoal',
  );
  if (
    typeof policy.counterfactualAction !== 'string' ||
    !GAME_ACTIONS.has(policy.counterfactualAction)
  ) {
    throw new Error('tick_summary.policyExecutionReceipt.counterfactualAction is invalid');
  }
  if (policy.counterfactualContract !== 'same_previous_state_same_human_action') {
    throw new Error('tick_summary.policyExecutionReceipt counterfactual contract is invalid');
  }
  aiObservableOutcome(
    policy.actualObservableOutcome,
    'tick_summary.policyExecutionReceipt.actualObservableOutcome',
  );
  aiObservableOutcome(
    policy.counterfactualObservableOutcome,
    'tick_summary.policyExecutionReceipt.counterfactualObservableOutcome',
  );
  const behaviorChangeConfirmed = booleanValue(
    policy.behaviorChangeConfirmed,
    'tick_summary.policyExecutionReceipt.behaviorChangeConfirmed',
  );
  const observedDifference =
    canonicalReceiptValue(policy.actualObservableOutcome) !==
    canonicalReceiptValue(policy.counterfactualObservableOutcome);
  if (behaviorChangeConfirmed !== observedDifference) {
    throw new Error(
      'tick_summary.policyExecutionReceipt behaviorChangeConfirmed contradicts its outcomes',
    );
  }
  for (const field of [
    'roundId',
    'policyRevision',
    'evaluatedAtTick',
    'completedAtTick',
    'proposedSubgoal',
    'proposedAction',
    'humanAction',
    'actuallyExecuted',
    'executionOutcome',
  ]) {
    if (policy[field] !== actual[field]) {
      throw new Error(`tick_summary policy receipt does not match ai receipt field ${field}`);
    }
  }
}

function probabilities(value: unknown): FeedbackProbabilities {
  if (!isRecord(value)) throw new Error('probabilities must be an object');
  const result = {
    Evaluative: Number(value.Evaluative),
    Imperative: Number(value.Imperative),
    Descriptive: Number(value.Descriptive),
  };
  for (const [label, score] of Object.entries(result)) {
    if (!Number.isFinite(score) || score < 0 || score > 1) {
      throw new Error(`${label} probability must be between 0 and 1`);
    }
  }
  const sum = result.Evaluative + result.Imperative + result.Descriptive;
  if (Math.abs(sum - 1) > 0.02) throw new Error('The three probabilities must sum to approximately 1');
  return result;
}

function phrase(value: unknown): PhraseResearchPrediction {
  if (!isRecord(value)) throw new Error('phrase prediction must be an object');
  const label = value.label as FeedbackLabel;
  if (!LABELS.has(label)) throw new Error('phrase label is invalid');
  const confidence = Number(value.confidence);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new Error('phrase confidence must be between 0 and 1');
  }
  const hasScoreSemantics =
    value.scoreKind !== undefined || value.thresholdPolicy !== undefined;
  if (
    hasScoreSemantics &&
    (value.scoreKind !== 'raw_model_softmax_score' ||
      (value.thresholdPolicy !== 'existing_web_preview_policy_not_validated_in_raw_score_space' &&
       value.thresholdPolicy !== 'synthetic_dev_threshold_v1'))
  ) {
    throw new Error('phrase score semantics are invalid');
  }
  return {
    phrase: text(value.phrase, 'phrase', 240),
    label,
    confidence,
    probabilities: probabilities(value.probabilities),
    abstained: Boolean(value.abstained),
    ...(hasScoreSemantics
      ? {
          scoreKind: 'raw_model_softmax_score' as const,
          thresholdPolicy:
            value.thresholdPolicy as 'existing_web_preview_policy_not_validated_in_raw_score_space' | 'synthetic_dev_threshold_v1',
        }
      : {}),
  };
}

function feedback(value: unknown): FeedbackRecord {
  if (!isRecord(value)) throw new Error('feedback must be an object');
  const route = value.route as FeedbackRoute;
  const topLabel = value.topLabel as FeedbackLabel;
  if (!ROUTES.has(route)) throw new Error('feedback route is invalid');
  if (!LABELS.has(topLabel)) throw new Error('feedback topLabel is invalid');
  if (!Array.isArray(value.phrases) || value.phrases.length < 1 || value.phrases.length > 12) {
    throw new Error('feedback phrases must contain 1-12 phrases');
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
  if (!isRecord(value)) throw new Error('session must be an object');
  const summary = value.summary;
  if (summary !== undefined && !isRecord(summary)) throw new Error('summary must be an object');
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
  if (!isRecord(value)) throw new Error('event must be an object');
  const eventType = value.eventType as ResearchEventType;
  if (!EVENT_TYPES.has(eventType)) throw new Error('eventType is invalid');
  if (!isRecord(value.payload)) throw new Error('event payload must be an object');
  assertNoSensitiveKeys(value.payload);
  if (eventType === 'tick_summary') validateTickSummaryReceipts(value.payload);
  if (!researchPayloadFits(value.payload)) throw new Error('event payload is too large');
  const parsedSessionId = uuid(value.sessionId, 'event sessionId');
  if (parsedSessionId !== sessionId) throw new Error('event sessionId does not match the batch');
  const modelHash =
    value.modelHash === undefined ? undefined : version(value.modelHash, 'event modelHash');
  const routeTrace =
    value.routeTrace === undefined ? undefined : text(value.routeTrace, 'event routeTrace', 512);
  const parsedFeedback = value.feedback === undefined ? undefined : feedback(value.feedback);
  if (eventType === 'feedback' && !parsedFeedback) throw new Error('feedback event is missing its feedback record');
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
    if (!isRecord(value)) throw new Error('Request body must be an object');
    const parsedSession = session(value.session);
    if (!Array.isArray(value.events) || value.events.length < 1) {
      throw new Error('events must not be empty');
    }
    if (value.events.length > MAX_BATCH_EVENTS) {
      throw new Error(`Each batch may contain at most ${MAX_BATCH_EVENTS} events`);
    }
    const events = value.events.map((item) => event(item, parsedSession.sessionId));
    const sequenceNumbers = new Set(events.map((item) => item.sequenceNumber));
    if (sequenceNumbers.size !== events.length) throw new Error('Duplicate sequenceNumber in batch');
    const eventIds = new Set(events.map((item) => item.eventId));
    if (eventIds.size !== events.length) throw new Error('Duplicate eventId in batch');
    return { ok: true, value: { session: parsedSession, events } };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Invalid request' };
  }
}

export function parseResearchSessionGrant(
  value: unknown,
  now = Date.now(),
): ValidationResult<ResearchSessionGrantRequest> {
  try {
    if (!isRecord(value)) throw new Error('Request body must be an object');
    const consentVersion = version(value.consentVersion, 'consentVersion');
    if (consentVersion !== CONSENT_VERSION) throw new Error('The consent form has changed. Please confirm it again.');
    const consentedAt = timestamp(value.consentedAt, 'consentedAt', now);
    if (consentedAt < now - MAX_CONSENT_AGE_MS || consentedAt > now + 60_000) {
      throw new Error('Consent time is invalid. Please confirm it again.');
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
    return { ok: false, error: error instanceof Error ? error.message : 'Invalid request' };
  }
}
