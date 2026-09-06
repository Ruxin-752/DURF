import type {
  GameAction,
  GameState,
} from './game';
import {
  TICK_SUMMARY_RECEIPT_SCHEMA_VERSION,
  type AiRelevantObservableOutcome,
  type AiTickReceipt,
  type PolicyExecutionReceipt,
} from './research-types';

export {
  TICK_SUMMARY_RECEIPT_SCHEMA_VERSION,
  type AiRelevantObservableOutcome,
  type AiTickReceipt,
  type PolicyExecutionReceipt,
};

export function aiExecutionResult(
  before: GameState,
  after: GameState,
  proposedAction: GameAction,
): Pick<AiTickReceipt, 'actuallyExecuted' | 'executionOutcome'> {
  if (proposedAction === 'stay') {
    return { actuallyExecuted: true, executionOutcome: 'stayed' };
  }
  if (proposedAction === 'interact') {
    const interaction = after.lastStepEvents.find(
      (event) => event.actor === 'ai' && event.code !== 'waiting',
    );
    const succeeded = Boolean(interaction && interaction.code !== 'interaction_failed');
    return {
      actuallyExecuted: succeeded,
      executionOutcome: succeeded ? 'interaction_succeeded' : 'interaction_failed',
    };
  }
  const moved = before.partner.x !== after.partner.x || before.partner.y !== after.partner.y;
  if (moved) return { actuallyExecuted: true, executionOutcome: 'moved' };
  if (after.lastStepEvents.some((event) => event.code === 'movement_collision')) {
    return { actuallyExecuted: false, executionOutcome: 'movement_collision' };
  }
  return { actuallyExecuted: false, executionOutcome: 'movement_blocked' };
}

export function observeAiRelevantOutcome(
  state: GameState,
): AiRelevantObservableOutcome {
  return {
    partner: {
      x: state.partner.x,
      y: state.partner.y,
      facing: state.partner.facing,
      held: state.partner.held,
      heldSoup: state.partner.heldSoup ? { ...state.partner.heldSoup } : null,
    },
    task: {
      score: state.score,
      ordersCompleted: state.ordersCompleted,
      pot: { ...state.pot },
      counters: Object.entries(state.counterObjects)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, object]) => ({
          key,
          item: object.item,
          soupContents: object.soupContents ? { ...object.soupContents } : null,
        })),
    },
    aiOrJointEvents: state.lastStepEvents
      .filter(
        (event) =>
          event.actor === 'ai' ||
          (event.actor === null && event.code === 'movement_collision'),
      )
      .map((event) => ({
        code: event.code,
        actor: event.actor as 'ai' | null,
        item: event.item ?? null,
        position: event.position ? { ...event.position } : null,
      })),
  };
}

export function observableAiOutcomesDiffer(
  actual: AiRelevantObservableOutcome,
  counterfactual: AiRelevantObservableOutcome,
): boolean {
  return canonicalReceiptValue(actual) !== canonicalReceiptValue(counterfactual);
}

export function canonicalReceiptValue(value: unknown): string {
  const normalize = (candidate: unknown): unknown => {
    if (Array.isArray(candidate)) return candidate.map(normalize);
    if (candidate && typeof candidate === 'object') {
      return Object.fromEntries(
        Object.entries(candidate as Record<string, unknown>)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, nested]) => [key, normalize(nested)]),
      );
    }
    return candidate;
  };
  return JSON.stringify(normalize(value));
}

export function counterfactualBehaviorChanged(
  actualAfter: GameState,
  counterfactualAfter: GameState,
): {
  actualObservableOutcome: AiRelevantObservableOutcome;
  counterfactualObservableOutcome: AiRelevantObservableOutcome;
  behaviorChangeConfirmed: boolean;
} {
  const actualObservableOutcome = observeAiRelevantOutcome(actualAfter);
  const counterfactualObservableOutcome = observeAiRelevantOutcome(counterfactualAfter);
  return {
    actualObservableOutcome,
    counterfactualObservableOutcome,
    behaviorChangeConfirmed: observableAiOutcomesDiffer(
      actualObservableOutcome,
      counterfactualObservableOutcome,
    ),
  };
}
