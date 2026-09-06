import { describe, expect, it } from 'vitest';
import {
  AI_LIVENESS_CONTRACT_VERSION,
  MAX_CONSECUTIVE_WAIT_STEPS,
  actionForAiSubgoal,
  chooseAiDecision,
  computeLivePolicyGeometry,
  createGameState,
  stepGame,
} from '../lib/game';
import {
  POLICY_ACTIVE_FEATURES,
  POLICY_CONTEXT_ONLY_FEATURES,
  POLICY_FEATURE_CONTRACT,
  POLICY_FEATURE_CONTRACT_VERSION,
  POLICY_UNSUPPORTED_FEATURES,
  REWARD_FEATURES,
  SUBGOALS,
  buildSubgoalContext,
  chooseRewardSubgoal,
  enumerateFeasibleSubgoals,
} from '../lib/subgoal-policy';

describe('reward-ranked subgoal policy', () => {
  it('locks the Python 10-subgoal vocabulary and 53-feature schema', () => {
    expect(SUBGOALS).toEqual([
      'GET_TOMATO',
      'PUT_TOMATO_IN_POT',
      'GET_ONION',
      'PUT_ONION_IN_POT',
      'GET_DISH',
      'PICKUP_SOUP',
      'SERVE_SOUP',
      'STASH_HELD_OBJECT',
      'YIELD_PATH',
      'WAIT',
    ]);
    expect(REWARD_FEATURES).toHaveLength(53);
    expect(new Set(REWARD_FEATURES)).toHaveLength(53);
    expect(POLICY_FEATURE_CONTRACT).toMatchObject({
      schemaVersion: POLICY_FEATURE_CONTRACT_VERSION,
      featureCount: 53,
    });
    expect(POLICY_ACTIVE_FEATURES).toHaveLength(33);
    expect(POLICY_CONTEXT_ONLY_FEATURES).toHaveLength(6);
    expect(POLICY_UNSUPPORTED_FEATURES).toHaveLength(14);
    const partition = [
      ...POLICY_ACTIVE_FEATURES,
      ...POLICY_CONTEXT_ONLY_FEATURES,
      ...POLICY_UNSUPPORTED_FEATURES,
    ];
    expect(new Set(partition)).toHaveLength(53);
    expect([...partition].sort()).toEqual([...REWARD_FEATURES].sort());
  });

  it('audits decision-null weights without letting them affect or claim a decision', () => {
    const decision = chooseRewardSubgoal(createGameState('running'), {
      pot_empty: 100,
      collision_risk: 100,
    });

    expect(decision.policyFeatureContractVersion).toBe(
      POLICY_FEATURE_CONTRACT_VERSION,
    );
    expect(decision.activeWeightFeatures).toEqual([]);
    expect(decision.ignoredDecisionNullWeightFeatures).toEqual([
      'pot_empty',
      'collision_risk',
    ]);
    expect(decision.decisionSource).toBe('decision_null_weights_h0_fallback');
    expect(decision.usedLearnedWeights).toBe(false);
    expect(decision.ranking.every(({ score }) => score === 0)).toBe(true);
    expect(decision.topContributions).toEqual([]);
  });

  it('uses the H0 tomato-first priority when zero weights tie', () => {
    const decision = chooseRewardSubgoal(createGameState('running'), {});

    expect(decision.feasibleSubgoals).toEqual([
      'GET_TOMATO',
      'GET_ONION',
      'WAIT',
    ]);
    expect(decision.chosenSubgoal).toBe('GET_TOMATO');
    expect(decision.h0Fallback).toBe('GET_TOMATO');
    expect(decision.decisionSource).toBe('h0_tie_fallback');
    expect(decision.usedLearnedWeights).toBe(false);
    expect(decision.rewardMargin).toBe(0);
    expect(decision.ranking.every(({ score }) => score === 0)).toBe(true);
  });

  it('flips the selected subgoal when tomato and onion weights are reversed', () => {
    const state = createGameState('running');
    const tomato = chooseRewardSubgoal(state, {
      pick_tomato: 4,
      pick_onion: 1,
    });
    const onion = chooseRewardSubgoal(state, {
      pick_tomato: 1,
      pick_onion: 4,
    });

    expect(tomato.chosenSubgoal).toBe('GET_TOMATO');
    expect(onion.chosenSubgoal).toBe('GET_ONION');
    expect(tomato.usedLearnedWeights).toBe(true);
    expect(onion.usedLearnedWeights).toBe(true);
    expect(onion.score).toBe(4);
    expect(onion.topContributions[0]).toEqual({
      feature: 'pick_onion',
      value: 1,
      weight: 4,
      contribution: 4,
    });
  });

  it('uses collaboration features in the same dot-product score', () => {
    const state = createGameState('running');
    const decision = chooseRewardSubgoal(
      state,
      {
        blocks_human_path: -3,
        clears_human_path: 3,
      },
      {
        contextOverrides: {
          humanIntent: 'serve the soup',
          candidatePathEffects: {
            GET_TOMATO: 'blocks',
            GET_ONION: 'clears',
          },
        },
      },
    );

    expect(decision.chosenSubgoal).toBe('GET_ONION');
    expect(decision.topContributions).toContainEqual({
      feature: 'clears_human_path',
      value: 1,
      weight: 3,
      contribution: 3,
    });
  });

  it('never lets a high-scoring but infeasible subgoal enter the ranking', () => {
    const decision = chooseRewardSubgoal(createGameState('running'), {
      serve_ready_soup: 1_000,
      completes_recipe: 1_000,
    });

    expect(decision.feasibleSubgoals).not.toContain('SERVE_SOUP');
    expect(decision.ranking.map(({ subgoal }) => subgoal)).not.toContain(
      'SERVE_SOUP',
    );
    expect(decision.chosenSubgoal).toBe('GET_TOMATO');
  });

  it('treats a requested live candidate set as a subset, never an injection', () => {
    const decision = chooseRewardSubgoal(
      createGameState('running'),
      { serve_ready_soup: 1_000 },
      { feasibleSubgoals: ['SERVE_SOUP'] },
    );

    expect(decision.feasibleSubgoals).toEqual(['WAIT']);
    expect(decision.chosenSubgoal).toBe('WAIT');
  });

  it('falls back safely to H0 when any supplied weight is non-finite', () => {
    const decision = chooseRewardSubgoal(createGameState('running'), {
      pick_onion: Number.POSITIVE_INFINITY,
      pick_tomato: 1,
    });

    expect(decision.chosenSubgoal).toBe('GET_TOMATO');
    expect(decision.decisionSource).toBe('invalid_weights_h0_fallback');
    expect(decision.invalidWeightFeatures).toEqual(['pick_onion']);
    expect(decision.usedLearnedWeights).toBe(false);
    expect(decision.ranking.every(({ score }) => score === 0)).toBe(true);
  });

  it('uses H0 to resolve an exact learned-score tie', () => {
    const decision = chooseRewardSubgoal(createGameState('running'), {
      pick_tomato: 5,
      pick_onion: 5,
    });

    expect(decision.chosenSubgoal).toBe('GET_TOMATO');
    expect(decision.decisionSource).toBe('h0_tie_fallback');
    expect(decision.rewardMargin).toBe(0);
    expect(decision.usedLearnedWeights).toBe(false);
  });

  it('adds YIELD_PATH only when live geometry provides a legal clearing move', () => {
    const state = createGameState('running');
    const ordinary = enumerateFeasibleSubgoals(buildSubgoalContext(state));
    const clearing = enumerateFeasibleSubgoals(
      buildSubgoalContext(state, {
        candidatePathEffects: { YIELD_PATH: 'clears' },
      }),
    );

    expect(ordinary).not.toContain('YIELD_PATH');
    expect(clearing.at(-2)).toBe('YIELD_PATH');
    expect(clearing.at(-1)).toBe('WAIT');
  });

  it('does not label moving out of a faced cell as shortest-path clearing', () => {
    const initial = createGameState('running');
    const blocked = {
      ...initial,
      partner: { ...initial.partner, x: 5, y: 1 },
      player: {
        ...initial.player,
        x: 6,
        y: 1,
        facing: 'left' as const,
        held: null,
      },
    };
    const geometry = computeLivePolicyGeometry(blocked);

    expect(geometry.yieldAction).toBeNull();
    expect(geometry.candidatePathEffects.YIELD_PATH).not.toBe('clears');
    expect(actionForAiSubgoal(blocked, 'YIELD_PATH')).toBe('stay');

    const decision = chooseAiDecision(blocked, { clears_human_shortest_path: 10 });
    expect(decision.feasibleSubgoals).not.toContain('YIELD_PATH');
    expect(decision.ranking.map(({ subgoal }) => subgoal)).not.toContain('YIELD_PATH');
  });

  it('executes YIELD_PATH when the move really shortens the human route', () => {
    const initial = createGameState('running');
    const blockingServingAccess = {
      ...initial,
      partner: { ...initial.partner, x: 8, y: 3 },
      player: {
        ...initial.player,
        x: 8,
        y: 4,
        facing: 'up' as const,
        held: 'soup' as const,
      },
    };
    const geometry = computeLivePolicyGeometry(blockingServingAccess);

    expect(geometry.yieldAction).toBe('up');
    expect(geometry.candidatePathEffects.YIELD_PATH).toBe('clears');
    expect(actionForAiSubgoal(blockingServingAccess, 'YIELD_PATH')).toBe('up');

    const decision = chooseAiDecision(blockingServingAccess, {
      clears_human_shortest_path: 10,
      avoids_duplicate_human_task: -1,
    });
    expect(decision.chosenSubgoal).toBe('YIELD_PATH');
    expect(decision.action).toBe('up');
    const advanced = stepGame(blockingServingAccess, decision.action, 'stay');
    expect(advanced.partner).toMatchObject({ x: 8, y: 2 });
  });

  it('infers an exact empty-hand pickup target and makes stealing actionable', () => {
    const initial = createGameState('running');
    const reachingForOnion = {
      ...initial,
      partner: { ...initial.partner, x: 3, y: 1 },
      player: {
        ...initial.player,
        x: 1,
        y: 2,
        facing: 'right' as const,
        held: null,
      },
      counterObjects: { '2,2': { item: 'onion' as const } },
    };
    const geometry = computeLivePolicyGeometry(reachingForOnion);

    expect(geometry.humanIntent).toBe('onion');
    expect(geometry.candidateTargetOverlapsHuman.GET_ONION).toBe(true);

    const decision = chooseAiDecision(reachingForOnion, {
      steals_human_target: -10,
    });
    const onion = decision.ranking.find(
      ({ subgoal }) => subgoal === 'GET_ONION',
    );
    expect(onion?.features.steals_human_target).toBe(1);
    expect(onion?.score).toBe(-10);
    expect(decision.activeWeightFeatures).toContain('steals_human_target');
    expect(decision.ignoredDecisionNullWeightFeatures).not.toContain(
      'steals_human_target',
    );
    expect(decision.chosenSubgoal).not.toBe('GET_ONION');

    const tomatoOverlap = computeLivePolicyGeometry({
      ...initial,
      pot: { ...initial.pot, stage: 'filling' as const, tomatoes: 1 },
      partner: { ...initial.partner, held: 'tomato' as const },
      player: { ...initial.player, held: 'tomato' as const },
    });
    expect(
      tomatoOverlap.candidateTargetOverlapsHuman.PUT_TOMATO_IN_POT,
    ).toBe(true);
  });

  it('removes WAIT after the Python-parity stall threshold when work is executable', () => {
    let stalled = createGameState('running');
    for (let step = 0; step < MAX_CONSECUTIVE_WAIT_STEPS; step += 1) {
      stalled = stepGame(stalled, 'stay', 'stay');
    }
    expect(stalled.aiPolicyLiveness).toMatchObject({
      contractVersion: AI_LIVENESS_CONTRACT_VERSION,
      consecutiveWaitSteps: MAX_CONSECUTIVE_WAIT_STEPS,
    });

    const decision = chooseAiDecision(stalled, { time_cost: 100 });
    expect(decision.waitGuardApplied).toBe(true);
    expect(decision.livenessReason).toBe('stalled_wait_infeasible');
    expect(decision.livenessRemovedSubgoals).toContain('WAIT');
    expect(decision.feasibleSubgoals).not.toContain('WAIT');
    expect(decision.chosenSubgoal).toBe('GET_TOMATO');
    expect(decision.action).not.toBe('stay');
  });

  it('still advances normal production under a reward that strongly prefers WAIT', () => {
    let state = createGameState('running');
    for (let step = 0; step < 120 && state.pot.tomatoes === 0; step += 1) {
      const decision = chooseAiDecision(state, { time_cost: 100 });
      state = stepGame(state, decision.action, 'stay');
    }

    expect(state.pot.tomatoes).toBeGreaterThan(0);
  });

  it('keeps WAIT valid while a complete soup is passively cooking', () => {
    const initial = createGameState('running');
    const cooking = {
      ...initial,
      pot: {
        stage: 'cooking' as const,
        secondsRemaining: 10,
        tomatoes: 2,
        onions: 1,
      },
      aiPolicyLiveness: {
        ...initial.aiPolicyLiveness,
        consecutiveWaitSteps: MAX_CONSECUTIVE_WAIT_STEPS,
      },
    };
    const decision = chooseAiDecision(cooking, { time_cost: 100 });

    expect(decision.waitGuardApplied).toBe(false);
    expect(decision.livenessReason).toBe('passive_cooking_wait_exempt');
    expect(decision.feasibleSubgoals).toContain('WAIT');
    expect(decision.chosenSubgoal).toBe('WAIT');
    expect(decision.action).toBe('stay');
  });

  it('breaks a detected GET-to-STASH cycle by forcing the safe pot action', () => {
    const initial = createGameState('running');
    const cycling = {
      ...initial,
      pot: { ...initial.pot, stage: 'filling' as const, tomatoes: 1 },
      partner: { ...initial.partner, held: 'tomato' as const },
      player: { ...initial.player, held: 'tomato' as const },
      aiPolicyLiveness: {
        ...initial.aiPolicyLiveness,
        recentObjectTransfer: { kind: 'stash' as const, item: 'tomato' as const },
        cycleBlockedStashItem: 'tomato' as const,
      },
    };
    const decision = chooseAiDecision(cycling, {
      avoids_duplicate_human_task: 20,
      duplicate_human_task: -20,
    });

    expect(decision.getStashCycleGuardApplied).toBe(true);
    expect(decision.livenessReason).toBe('get_stash_cycle_infeasible');
    expect(decision.livenessRemovedSubgoals).toEqual(
      expect.arrayContaining(['STASH_HELD_OBJECT', 'WAIT']),
    );
    expect(decision.feasibleSubgoals).toEqual(['PUT_TOMATO_IN_POT']);
    expect(decision.chosenSubgoal).toBe('PUT_TOMATO_IN_POT');
    expect(decision.action).not.toBe('stay');
  });

  it('records a physical GET-to-STASH transfer cycle in GameState', () => {
    const initial = createGameState('running');
    const carrying = {
      ...initial,
      partner: {
        ...initial.partner,
        x: 3,
        y: 1,
        facing: 'down' as const,
        held: 'tomato' as const,
      },
      aiPolicyLiveness: {
        ...initial.aiPolicyLiveness,
        recentObjectTransfer: { kind: 'get' as const, item: 'tomato' as const },
      },
    };
    const stashed = stepGame(carrying, 'interact', 'stay');

    expect(stashed.aiPolicyLiveness.recentObjectTransfer).toEqual({
      kind: 'stash',
      item: 'tomato',
    });
    expect(stashed.aiPolicyLiveness.cycleBlockedStashItem).toBe('tomato');
  });
});
