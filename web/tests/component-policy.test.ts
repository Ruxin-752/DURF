import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';
import {
  FEEDBACK_ROUTE_TRACE_MAX_LENGTH,
  buildFeedbackRouteTrace,
  gameForPolicyProposal,
} from '../lib/feedback-policy-contract';
import { configuredFeedbackRoute } from '../lib/feedback-route-config';
import {
  chooseAiDecision,
  createGameState,
  startGame,
  stepGame,
  togglePause,
} from '../lib/game';

const component = readFileSync(
  join(import.meta.dirname, '..', 'components', 'kitchen-game-app.tsx'),
  'utf8',
);
const environmentExample = readFileSync(join(import.meta.dirname, '..', '.env.example'), 'utf8');

describe('hidden feedback-route policy', () => {
  it('uses deployment configuration without exposing a route selector', () => {
    expect(configuredFeedbackRoute('')).toBe('route1');
    expect(configuredFeedbackRoute('route1')).toBe('route1');
    expect(configuredFeedbackRoute('route2')).toBe('route2');
    expect(() => configuredFeedbackRoute('invalid')).toThrow(
      'NEXT_PUBLIC_FEEDBACK_ROUTE must be route1 or route2',
    );
    expect(component).toContain('const DEPLOYMENT_FEEDBACK_ROUTE = configuredFeedbackRoute()');
    expect(component).not.toContain('setRoute');
    expect(component).not.toContain('route-switch');
    expect(component).not.toContain('Feedback update route');
    expect(environmentExample).toContain('NEXT_PUBLIC_FEEDBACK_ROUTE=route1');
  });

  it('retains an executable Route2 update path and its research trace', () => {
    expect(component).toContain("if (DEPLOYMENT_FEEDBACK_ROUTE === 'route1')");
    expect(component).toContain('models.route2(utterance');
    expect(component).toContain('applyRoute2Gaussian(prior, prediction.weights, 2)');
    expect(component).toContain('buildFeedbackRouteTrace({');
  });

  it('publishes each accepted posterior as a proposal for the next live AI decision', () => {
    expect(component).toContain('chooseAiDecision(previous, activePolicy.weights)');
    expect(component).toContain('weights: posteriorWeights(models.features, nextPosteriorMean)');
    expect(component).toContain('revision: policyBefore.revision + 1');
    expect(component).toContain('pendingPolicyReceiptRef.current = nextPendingReceipt');
    expect(component).not.toContain('chooseAiAction(previous)');
  });

  it('keeps research internals out of the player-facing UI', () => {
    expect(component).not.toContain('Model and paper status');
    expect(component).not.toContain('model SHA');
    expect(component).not.toContain('Paper migration');
    expect(component).not.toContain('AI policy revision');
    expect(component).not.toContain('Previous-policy counterfactual');
    expect(component).not.toContain('Last AI execution receipt');
    expect(component).not.toContain('Awaiting AI receipt');
    expect(component).toContain("AI partner's next move");
    expect(component).toContain('Feedback accepted');
    expect(component).toContain('Feedback not applied');
    expect(component).toContain('Your feedback will appear here');
    expect(component).toContain('Not sure — try a clearer sentence.');
    expect(component).not.toContain('Scores compare the three categories');
    expect(component).not.toContain('threshold.');
    expect(component).not.toContain('Retry model loading');
    expect(component).toContain('Reload and try again');
  });
});

describe('bounded feedback route trace contract', () => {
  it('constructs concise, versioned identifiers for both feedback routes', () => {
    const route1 = buildFeedbackRouteTrace({
      route: 'route1',
      result: 'rejected',
      reason: 'feedback_form_low_confidence',
      classifierVariant: 'production',
      policyRevision: 7,
      snapshotStatus: 'paused',
    });
    const route2 = buildFeedbackRouteTrace({
      route: 'route2',
      result: 'updated',
      classifierVariant: 'boundary-shadow-raw-v2',
      policyRevision: 18,
      snapshotStatus: 'running',
    });

    expect(route1).toBe(
      'trace=feedback-route-v2;route=route1;pipeline=fg3>features53>bayes_full;result=rejected;reason=feedback_form_low_confidence;classifier=production;policy_rev=7;snapshot_status=paused;proposal_eval_status=running;execution=not_applicable',
    );
    expect(route2).toBe(
      'trace=feedback-route-v2;route=route2;pipeline=ensemble10>reward53>gaussian_independent;result=updated;reason=none;classifier=boundary-shadow-raw-v2;policy_rev=18;snapshot_status=running;proposal_eval_status=running;execution=next_tick_receipt',
    );
    expect(route1.length).toBeLessThanOrEqual(FEEDBACK_ROUTE_TRACE_MAX_LENGTH);
    expect(route2.length).toBeLessThanOrEqual(FEEDBACK_ROUTE_TRACE_MAX_LENGTH);
  });

  it('bounds and escapes a defensive worst-case reason token', () => {
    const trace = buildFeedbackRouteTrace({
      route: 'route1',
      result: 'rejected',
      reason: 'unexpected;detail='.repeat(100),
      classifierVariant: 'production',
      policyRevision: Number.MAX_SAFE_INTEGER,
      snapshotStatus: 'paused',
    });

    expect(trace.length).toBeLessThanOrEqual(FEEDBACK_ROUTE_TRACE_MAX_LENGTH);
    expect(trace).not.toContain(';detail=');
  });
});

describe('feedback snapshot and execution receipt', () => {
  it('freezes the round, tick, game, trajectory, and policy before the first await', () => {
    const start = component.indexOf('const submitFeedback = async');
    const end = component.indexOf('const statusCopy', start);
    const flow = component.slice(start, end);
    const snapshot = flow.indexOf('const submissionSnapshot = {');
    const firstAwait = flow.indexOf('await new Promise<void>');

    expect(snapshot).toBeGreaterThan(-1);
    expect(firstAwait).toBeGreaterThan(snapshot);
    expect(flow.slice(snapshot, firstAwait)).toContain('roundId,');
    expect(flow.slice(snapshot, firstAwait)).toContain('tick: stateAtSubmission.tick');
    expect(flow.slice(snapshot, firstAwait)).toContain('game: stateAtSubmission');
    expect(flow.slice(snapshot, firstAwait)).toContain('trajectory: [...trajectoryRef.current]');
    expect(flow.slice(snapshot, firstAwait)).toContain('weights: { ...policyRef.current.weights }');
    expect(flow).toContain('submissionSnapshot.roundEpoch !== roundEpochRef.current');
    expect(flow).toContain('submissionSnapshot.roundId !== roundIdRef.current');
    expect(flow).toContain('proposalGame = gameForPolicyProposal(submissionSnapshot.game)');
    expect(flow).toContain('chooseAiDecision(\n        proposalGame');
    expect(flow).toContain('(step) => step.stateAfter.tick > submissionSnapshot.tick');
  });

  it('blocks restart only during computation and cancels a pending receipt on restart', () => {
    const start = component.indexOf('const startRound = useCallback');
    const end = component.indexOf('const beginRound', start);
    const restart = component.slice(start, end);
    const submitStart = component.indexOf('const submitFeedback = async');
    const submitEnd = component.indexOf('const statusCopy', submitStart);
    const submit = component.slice(submitStart, submitEnd);

    expect(restart).toContain('if (processingFeedbackRef.current) return;');
    expect(restart).toContain('const cancelledReceipt = pendingPolicyReceiptRef.current;');
    expect(restart).toContain("reason: 'round_restarted_before_next_tick'");
    expect(restart).not.toContain('pendingPolicyReceiptRef.current) return');
    expect(submit).toContain("!['running', 'paused'].includes(stateAtSubmission.status)");
    expect(submit.indexOf("if (gameRef.current.status === 'finished')")).toBeLessThan(
      submit.indexOf('policyRef.current = nextPolicy'),
    );
    expect(submit).toContain('processingFeedbackRef.current = false;');
  });

  it('evaluates paused feedback against the same frozen state in running mode', () => {
    const running = startGame(createGameState());
    const paused = togglePause(running);
    const proposalGame = gameForPolicyProposal(paused);
    const proposal = chooseAiDecision(proposalGame, {});

    expect(paused.status).toBe('paused');
    expect(proposalGame).toEqual({ ...paused, status: 'running' });
    expect(proposalGame.tick).toBe(paused.tick);
    expect(chooseAiDecision(paused, {}).action).toBe('stay');
    expect(stepGame(paused, proposal.action, 'stay')).toBe(paused);

    const resumed = togglePause(paused);
    expect(stepGame(resumed, proposal.action, 'stay').tick).toBe(paused.tick + 1);
  });

  it('keeps pause/resume locked only while feedback computation is in flight', () => {
    const start = component.indexOf('const handlePause = useCallback');
    const end = component.indexOf('useEffect(() => {', start);
    const pauseHandler = component.slice(start, end);

    expect(pauseHandler).toContain('if (processingFeedbackRef.current) return;');
    expect(component).toContain("proposalGame = gameForPolicyProposal(submissionSnapshot.game)");
    expect(component).toContain('proposalEvaluationStatus: proposalGame.status');
    expect(component).toContain('starts after resume');
  });

  it('records proposed and actually executed actions without claiming behavior early', () => {
    expect(component).toContain('const execution = aiExecutionResult(previous, next, aiAction)');
    expect(component).toContain('const counterfactualNext = stepGame(');
    expect(component).toContain('counterfactual.action,\n          humanAction,');
    expect(component).toContain('counterfactualBehaviorChanged(');
    expect(component).toContain('aiExecutionReceipt: aiReceipt');
    expect(component).toContain('receiptSchemaVersion: TICK_SUMMARY_RECEIPT_SCHEMA_VERSION');
    expect(component).toContain('policyExecutionReceipt');
    expect(component).not.toContain('executedJointActions');
  });

  it('publishes the feedback event before committing posterior and receipt refs', () => {
    const start = component.indexOf('const submitFeedback = async');
    const end = component.indexOf('const statusCopy', start);
    const flow = component.slice(start, end);
    const enqueue = flow.indexOf("queue.enqueue('feedback'");
    const routeCommit = flow.indexOf('route1Ref.current = nextRoute1State');
    const policyCommit = flow.indexOf('policyRef.current = nextPolicy');
    const pendingCommit = flow.indexOf('pendingPolicyReceiptRef.current = nextPendingReceipt');

    expect(enqueue).toBeGreaterThan(-1);
    expect(routeCommit).toBeGreaterThan(enqueue);
    expect(policyCommit).toBeGreaterThan(enqueue);
    expect(pendingCommit).toBeGreaterThan(enqueue);
    expect(flow.slice(0, enqueue)).not.toContain('route1Ref.current = candidate');
    expect(flow.slice(0, enqueue)).not.toContain('route2Ref.current = result.state');
  });
});

describe('pagehide queue lifecycle', () => {
  it('does not close the queue when the page enters BFCache', () => {
    const start = component.indexOf('const handlePageHide');
    const end = component.indexOf("window.addEventListener('pagehide'", start);
    const handler = component.slice(start, end);

    expect(handler).toContain('(event: PageTransitionEvent)');
    expect(handler).toContain('if (!event.persisted) queue.end(gameSummary(gameRef.current));');
    expect(handler).toContain('queue.flushWithBeacon();');
  });
});

describe('player-facing classifier display', () => {
  it('keeps the three score display while hiding model explanations', () => {
    expect(component).not.toContain('Experimental language mode');
    expect(component).not.toContain('accuracy estimates');
    expect(component).not.toContain('drives the displayed result and Route1 update');
    expect(component).not.toContain('trial routing policy threshold');
    expect(component).toContain('aria-label={`${LABEL_COPY[label].name} score');
    expect(component).toContain("scorePolicyVersion: 'synthetic-raw-softmax-v1'");
    expect(component).toContain("thresholdPolicy: 'synthetic_dev_threshold_v1'");
    expect(component).toContain("{copy.name}{uncertain ? ' · uncertain' : ''}");
  });
});
