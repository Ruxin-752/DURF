import { describe, expect, it } from 'vitest';

import {
  CLIENT_VERSION,
  CONSENT_VERSION,
  SCHEMA_VERSION,
  TICK_SUMMARY_RECEIPT_SCHEMA_VERSION,
  type ResearchEvent,
} from '../lib/research-types';
import { parseResearchBatch } from '../lib/validation';

const SESSION_ID = '11111111-1111-4111-8111-111111111111';
const USER_ID = '22222222-2222-4222-8222-222222222222';
const ROUND_ID = '33333333-3333-4333-8333-333333333333';
const FEEDBACK_ID = '44444444-4444-4444-8444-444444444444';

function observableOutcome(x: number) {
  return {
    partner: { x, y: 1, facing: 'right', held: null, heldSoup: null },
    task: {
      score: 0,
      ordersCompleted: 0,
      pot: {
        stage: 'empty',
        secondsRemaining: 0,
        tomatoes: 0,
        onions: 0,
      },
      counters: [],
    },
    aiOrJointEvents: [],
  };
}

function event(now: number, payload: Record<string, unknown>): ResearchEvent {
  return {
    eventId: '55555555-5555-4555-8555-555555555555',
    sessionId: SESSION_ID,
    sequenceNumber: 0,
    eventType: 'tick_summary',
    occurredAt: now,
    payload,
    schemaVersion: SCHEMA_VERSION,
  };
}

function parse(payload: Record<string, unknown>) {
  const now = Date.now();
  return parseResearchBatch({
    session: {
      sessionId: SESSION_ID,
      anonymousUserId: USER_ID,
      consentVersion: CONSENT_VERSION,
      consentedAt: now,
      startedAt: now,
      clientVersion: CLIENT_VERSION,
      schemaVersion: SCHEMA_VERSION,
    },
    events: [event(now, payload)],
  });
}

function receiptPayload() {
  const baseReceipt = {
    roundId: ROUND_ID,
    policyRevision: 1,
    evaluatedAtTick: 3,
    completedAtTick: 4,
    proposedSubgoal: 'GET_ONION',
    proposedAction: 'right',
    humanAction: 'stay',
    actuallyExecuted: true,
    executionOutcome: 'moved',
  };
  return {
    receiptSchemaVersion: TICK_SUMMARY_RECEIPT_SCHEMA_VERSION,
    roundId: ROUND_ID,
    tick: 4,
    requestedJointActions: { ai: 'right', human: 'stay' },
    aiExecutionReceipt: baseReceipt,
    policyExecutionReceipt: {
      ...baseReceipt,
      feedbackId: FEEDBACK_ID,
      counterfactualSubgoal: 'GET_TOMATO',
      counterfactualAction: 'stay',
      counterfactualContract: 'same_previous_state_same_human_action',
      actualObservableOutcome: observableOutcome(4),
      counterfactualObservableOutcome: observableOutcome(3),
      behaviorChangeConfirmed: true,
    },
  };
}

describe('versioned tick receipt validation', () => {
  it('accepts a consistent versioned execution receipt', () => {
    expect(parse(receiptPayload())).toMatchObject({ ok: true });
  });

  it('rejects a behavior claim that contradicts the recorded counterfactual outcomes', () => {
    const payload = receiptPayload();
    payload.policyExecutionReceipt.behaviorChangeConfirmed = false;

    expect(parse(payload)).toMatchObject({
      ok: false,
      error:
        'tick_summary.policyExecutionReceipt behaviorChangeConfirmed contradicts its outcomes',
    });
  });

  it('keeps legacy v2 tick payloads readable when they have no receipt version', () => {
    expect(
      parse({
        tick: 4,
        requestedJointActions: { ai: 'right', human: 'stay' },
        executedJointActions: { ai: 'right', human: 'stay' },
      }),
    ).toMatchObject({ ok: true });
  });

  it('rejects unversioned payloads that try to use the new receipt fields', () => {
    expect(
      parse({
        tick: 4,
        aiExecutionReceipt: { malformedReceipt: true },
      }),
    ).toMatchObject({
      ok: false,
      error: 'tick_summary receipts require receiptSchemaVersion',
    });
  });
});
