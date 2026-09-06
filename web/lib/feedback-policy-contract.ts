import type { BrowserModels } from './browser-models';
import type { GameState } from './game';

export const FEEDBACK_ROUTE_TRACE_MAX_LENGTH = 512;

interface FeedbackRouteTraceInput {
  route: 'route1' | 'route2';
  result: 'updated' | 'rejected';
  reason?: string;
  classifierVariant: BrowserModels['classifierVariant'] | 'synthetic-v1';
  policyRevision: number;
  snapshotStatus: GameState['status'];
}

function routeTraceToken(value: string, maxLength = 64): string {
  return value.replace(/[^a-zA-Z0-9_.:-]/g, '_').slice(0, maxLength);
}

/** Compact, bounded identifiers; the full audit detail remains in the event payload. */
export function buildFeedbackRouteTrace(input: FeedbackRouteTraceInput): string {
  const pipeline = input.route === 'route1'
    ? input.classifierVariant === 'synthetic-v1'
      ? 'grounding3>features53>pragmatic_bayes'
      : 'fg3>features53>bayes_full'
    : 'ensemble10>reward53>gaussian_independent';
  const trace = [
    'trace=feedback-route-v2',
    `route=${input.route}`,
    `pipeline=${pipeline}`,
    `result=${input.result}`,
    `reason=${routeTraceToken(input.reason ?? 'none')}`,
    `classifier=${routeTraceToken(input.classifierVariant)}`,
    `policy_rev=${Math.max(0, Math.trunc(input.policyRevision))}`,
    `snapshot_status=${input.snapshotStatus}`,
    'proposal_eval_status=running',
    `execution=${input.result === 'updated' ? 'next_tick_receipt' : 'not_applicable'}`,
  ].join(';');
  return trace.slice(0, FEEDBACK_ROUTE_TRACE_MAX_LENGTH);
}

/** Policy proposals ignore only the pause flag; the frozen task state and tick stay intact. */
export function gameForPolicyProposal(game: GameState): GameState {
  return game.status === 'paused' ? { ...game, status: 'running' } : game;
}
