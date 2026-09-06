import type { GameAction, GameState, Point } from './game';

export const FEEDBACK_TRAJECTORY_WINDOW_STEPS = 40;

export interface TrajectoryStep {
  stateBefore: GameState;
  stateAfter: GameState;
  aiAction: GameAction;
  humanAction: GameAction;
  reward: number;
}

const MOVE_DELTAS: Partial<Record<GameAction, Point>> = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
};

function samePoint(left: Point, right: Point): boolean {
  return left.x === right.x && left.y === right.y;
}

/** Build the same transition record consumed by the Python trajectory bridge. */
export function recordTrajectoryStep(
  stateBefore: GameState,
  stateAfter: GameState,
  aiAction: GameAction,
  humanAction: GameAction,
  reward = stateAfter.score - stateBefore.score,
): TrajectoryStep {
  return { stateBefore, stateAfter, aiAction, humanAction, reward };
}

/**
 * Browser port of src/trajectory_featurizer.py::featurize_trajectory_steps.
 * Counts are intentionally left raw here; paper observations and Route 2
 * normalize the frozen window at their input boundary.
 */
export function featurizeTrajectorySteps(
  steps: readonly TrajectoryStep[],
): Record<string, number> {
  const counts: Record<string, number> = {};
  const indicators = new Set<string>();
  const add = (feature: string, value = 1) => {
    counts[feature] = (counts[feature] ?? 0) + value;
  };
  const indicate = (feature: string) => {
    if (indicators.has(feature)) return;
    indicators.add(feature);
    add(feature);
  };

  for (const step of steps) {
    if (step.aiAction === 'stay') add('time_cost');

    const heldBefore = step.stateBefore.partner.held;
    const heldAfter = step.stateAfter.partner.held;
    if (heldAfter !== heldBefore) {
      if (heldAfter === 'tomato') {
        add('ingredient_tomato');
        add('pick_tomato');
      } else if (heldAfter === 'onion') {
        add('ingredient_onion');
        add('pick_onion');
      } else if (heldAfter === 'dish') {
        add('pick_dish');
      } else if (heldAfter === 'soup') {
        add('pick_ready_soup');
      }

      if (step.aiAction === 'interact' && heldAfter === null) {
        if (heldBefore === 'tomato') add('adds_needed_tomato');
        else if (heldBefore === 'onion') add('adds_needed_onion');
        else if (heldBefore === 'soup' && step.reward > 0) add('serve_ready_soup');
      }
    }

    if (step.stateAfter.pot.stage === 'empty') indicate('pot_empty');
    if (step.stateAfter.pot.stage === 'cooking') indicate('pot_cooking');
    if (step.stateAfter.pot.stage === 'ready') {
      indicate('soup_ready');
      if (heldAfter !== 'dish') indicate('dish_needed_for_ready_soup');
    }

    const humanDelta = MOVE_DELTAS[step.humanAction];
    if (humanDelta) {
      const humanTarget = {
        x: step.stateBefore.player.x + humanDelta.x,
        y: step.stateBefore.player.y + humanDelta.y,
      };
      if (
        samePoint(step.stateAfter.player, step.stateBefore.player) &&
        samePoint(step.stateBefore.partner, humanTarget)
      ) {
        add('blocks_human_path');
        add('human_wait_cost');
      }
      if (
        samePoint(step.stateAfter.partner, humanTarget) &&
        !samePoint(step.stateBefore.partner, humanTarget)
      ) {
        add('cuts_in_front_of_human');
        add('collision_risk');
      }
    }
  }

  return counts;
}

export function normalizeTrajectoryFeatures(
  counts: Record<string, number>,
): Record<string, number> {
  const total = Object.values(counts).reduce(
    (sum, value) => sum + Math.abs(Number(value)),
    0,
  );
  if (total === 0) return {};
  return Object.fromEntries(
    Object.entries(counts).map(([feature, value]) => [feature, value / total]),
  );
}

export function appendTrajectoryStep(
  steps: readonly TrajectoryStep[],
  next: TrajectoryStep,
  limit = FEEDBACK_TRAJECTORY_WINDOW_STEPS,
): TrajectoryStep[] {
  if (!Number.isInteger(limit) || limit <= 0) {
    throw new Error('trajectory window limit must be a positive integer');
  }
  return [...steps, next].slice(-limit);
}
