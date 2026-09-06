export const SCHEMA_VERSION = 'durf-web-event-v2';
export const CLIENT_VERSION = 'durf-kitchen-synthetic-0.3.0';
export const FEEDBACK_SEMANTICS_VERSION = 'speech-act-grounding-v1';
export const CONSENT_VERSION = 'durf-anonymous-research-en-2026-08-27-v3';
export const TICK_SUMMARY_RECEIPT_SCHEMA_VERSION =
  'durf-tick-summary-receipt-v1';

export type FeedbackLabel = 'Evaluative' | 'Imperative' | 'Descriptive';
export type FeedbackRoute = 'route1' | 'route2';
export type ResearchGameAction =
  | 'up'
  | 'down'
  | 'left'
  | 'right'
  | 'interact'
  | 'stay';
export type ResearchHeldItem = 'tomato' | 'onion' | 'dish' | 'soup' | null;
export type AiExecutionOutcome =
  | 'moved'
  | 'movement_blocked'
  | 'movement_collision'
  | 'interaction_succeeded'
  | 'interaction_failed'
  | 'stayed';

export interface AiRelevantObservableOutcome {
  partner: {
    x: number;
    y: number;
    facing: string;
    held: ResearchHeldItem;
    heldSoup: { tomatoes: number; onions: number } | null;
  };
  task: {
    score: number;
    ordersCompleted: number;
    pot: {
      stage: string;
      secondsRemaining: number;
      tomatoes: number;
      onions: number;
    };
    counters: Array<{
      key: string;
      item: Exclude<ResearchHeldItem, null>;
      soupContents: { tomatoes: number; onions: number } | null;
    }>;
  };
  aiOrJointEvents: Array<{
    code: string;
    actor: 'ai' | null;
    item: Exclude<ResearchHeldItem, null> | null;
    position: { x: number; y: number } | null;
  }>;
}

export interface AiTickReceipt {
  roundId: string;
  policyRevision: number;
  evaluatedAtTick: number;
  completedAtTick: number;
  proposedSubgoal: string;
  proposedAction: ResearchGameAction;
  humanAction: ResearchGameAction;
  actuallyExecuted: boolean;
  executionOutcome: AiExecutionOutcome;
}

export interface PolicyExecutionReceipt extends AiTickReceipt {
  feedbackId: string;
  counterfactualSubgoal: string;
  counterfactualAction: ResearchGameAction;
  counterfactualContract: 'same_previous_state_same_human_action';
  actualObservableOutcome: AiRelevantObservableOutcome;
  counterfactualObservableOutcome: AiRelevantObservableOutcome;
  behaviorChangeConfirmed: boolean;
}

export interface TickSummaryReceiptPayload {
  receiptSchemaVersion: typeof TICK_SUMMARY_RECEIPT_SCHEMA_VERSION;
  aiExecutionReceipt: AiTickReceipt;
  policyExecutionReceipt?: PolicyExecutionReceipt;
}

export interface FeedbackProbabilities {
  Evaluative: number;
  Imperative: number;
  Descriptive: number;
}

export interface PhraseResearchPrediction {
  phrase: string;
  label: FeedbackLabel;
  confidence: number;
  probabilities: FeedbackProbabilities;
  abstained: boolean;
  scoreKind?: 'raw_model_softmax_score';
  thresholdPolicy?: 'existing_web_preview_policy_not_validated_in_raw_score_space' | 'synthetic_dev_threshold_v1';
}

export interface SessionRecord {
  sessionId: string;
  anonymousUserId: string;
  consentVersion: string;
  consentedAt: number;
  startedAt: number;
  endedAt?: number;
  clientVersion: string;
  schemaVersion: string;
  summary?: Record<string, unknown>;
}

export interface FeedbackRecord {
  feedbackId: string;
  utterance: string;
  route: FeedbackRoute;
  topLabel: FeedbackLabel;
  lowConfidence: boolean;
  probabilities: FeedbackProbabilities;
  phrases: PhraseResearchPrediction[];
  modelHash: string;
  schemaVersion: string;
  routeTrace: string;
}

export type ResearchEventType =
  | 'session_start'
  | 'move'
  | 'interact'
  | 'pause'
  | 'resume'
  | 'feedback'
  | 'restart'
  | 'session_end'
  | 'tick_summary';

export interface ResearchEvent {
  eventId: string;
  sessionId: string;
  sequenceNumber: number;
  eventType: ResearchEventType;
  occurredAt: number;
  payload: Record<string, unknown>;
  schemaVersion: string;
  modelHash?: string;
  routeTrace?: string;
  probabilities?: FeedbackProbabilities;
  feedback?: FeedbackRecord;
}

export interface ResearchBatch {
  session: SessionRecord;
  events: ResearchEvent[];
}

export interface ResearchSessionGrantRequest {
  sessionId: string;
  anonymousUserId: string;
  consentVersion: string;
  consentedAt: number;
}

export type ResearchEventWriteStatus = 'accepted' | 'duplicate' | 'conflict';

export interface ResearchEventWriteResult {
  eventId: string;
  status: ResearchEventWriteStatus;
  reason?: string;
}

export interface ResearchBatchResponse {
  ok: true;
  accepted: number;
  duplicate: number;
  conflict: number;
  events: ResearchEventWriteResult[];
}
