export const SCHEMA_VERSION = 'durf-web-event-v2';
export const CLIENT_VERSION = 'durf-kitchen-beta-0.2.1';
export const CONSENT_VERSION = 'durf-anonymous-research-en-2026-08-27-v3';

export type FeedbackLabel = 'Evaluative' | 'Imperative' | 'Descriptive';
export type FeedbackRoute = 'route1' | 'route2';

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
