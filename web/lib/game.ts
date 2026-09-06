import {
  SUBGOALS,
  buildSubgoalContext,
  chooseRewardSubgoal,
  enumerateFeasibleSubgoals,
  passiveCookingWaitIsValid,
  type PathEffect,
  type RewardSubgoalDecision,
  type RewardWeights,
  type Subgoal,
  type SubgoalContext,
} from './subgoal-policy';

export const SOURCE_LAYOUT_NAME = 'ring_tomato_onion_10x6_h0_full_task';

// Exact grid characters from src/overcooked_ai_py/data/layouts/
// ring_tomato_onion_10x6_h0_full_task.layout. Digits encode start positions.
export const SOURCE_LAYOUT_ROWS = [
  'XXXXPXXXXX',
  'D  1  2  X',
  'X XXXXXX X',
  'X XXXXXX S',
  'T        X',
  'XXXXOXXXXX',
] as const;

// The Overcooked parser turns the two start digits into walkable floor.
export const TERRAIN_ROWS = SOURCE_LAYOUT_ROWS.map((row) =>
  row.replace(/[12]/gu, ' '),
) as readonly string[];

export const BOARD_WIDTH = 10;
export const BOARD_HEIGHT = 6;
export const ROUND_STEPS = 800;
export const STEPS_PER_SECOND = 2;
export const GAME_STEP_INTERVAL_MS = 1_000 / STEPS_PER_SECOND;
export const ROUND_SECONDS = ROUND_STEPS / STEPS_PER_SECOND;
export const COOK_TIME_STEPS = 20;
export const CORRECT_SOUP_REWARD = 20;
export const AI_LIVENESS_CONTRACT_VERSION = 'durf-web-ai-liveness-v1';
export const MAX_CONSECUTIVE_WAIT_STEPS = 3;

export type GameStatus = 'waiting' | 'running' | 'paused' | 'finished';
export type Ingredient = 'tomato' | 'onion';
export type HeldItem = Ingredient | 'dish' | 'soup' | null;
export type Direction = 'up' | 'down' | 'left' | 'right';
export type GameAction = Direction | 'interact' | 'stay';
export type GameActor = 'ai' | 'human';
export type StationKind = Ingredient | 'dish' | 'pot' | 'serve';

export type GameEventCode =
  | 'waiting'
  | 'game_started'
  | 'game_paused'
  | 'game_resumed'
  | 'round_finished'
  | 'wait'
  | 'move'
  | 'movement_blocked'
  | 'movement_collision'
  | 'interaction_failed'
  | 'pick_tomato'
  | 'pick_onion'
  | 'pick_dish'
  | 'pick_soup'
  | 'drop_counter'
  | 'pickup_counter'
  | 'pot_tomato'
  | 'pot_onion'
  | 'serve_correct_soup'
  | 'serve_wrong_soup'
  | 'cooking_started'
  | 'soup_ready';

export interface Point {
  x: number;
  y: number;
}

// Pygame passes (AI action, human action) to the environment. Layout digit 1 is
// therefore the AI and digit 2 is the human player.
export const PLAYER_STARTS = {
  partner: { x: 3, y: 1 },
  player: { x: 6, y: 1 },
} as const;

export const STATIONS: ReadonlyArray<{
  kind: StationKind;
  terrain: 'T' | 'O' | 'D' | 'P' | 'S';
  label: string;
  shortLabel: string;
  position: Point;
}> = [
  { kind: 'pot', terrain: 'P', label: 'Soup pot', shortLabel: 'Pot', position: { x: 4, y: 0 } },
  { kind: 'dish', terrain: 'D', label: 'Dish dispenser', shortLabel: 'Dish', position: { x: 0, y: 1 } },
  { kind: 'serve', terrain: 'S', label: 'Serving window', shortLabel: 'Serve', position: { x: 9, y: 3 } },
  { kind: 'tomato', terrain: 'T', label: 'Tomato dispenser', shortLabel: 'Tomato', position: { x: 0, y: 4 } },
  { kind: 'onion', terrain: 'O', label: 'Onion dispenser', shortLabel: 'Onion', position: { x: 4, y: 5 } },
];

export interface SoupContents {
  tomatoes: number;
  onions: number;
}

export interface Chef extends Point {
  held: HeldItem;
  facing: Direction;
  heldSoup?: SoupContents;
}

export interface CounterObject {
  item: Exclude<HeldItem, null>;
  soupContents?: SoupContents;
}

export type CounterObjects = Record<string, CounterObject>;

export interface PotState {
  stage: 'empty' | 'filling' | 'cooking' | 'ready';
  // Kept for UI compatibility. The value is measured in environment steps,
  // matching the Python Overcooked MDP rather than wall-clock seconds.
  secondsRemaining: number;
  tomatoes: number;
  onions: number;
}

type AiObjectTransferKind = 'get' | 'stash';

export interface AiObjectTransfer {
  kind: AiObjectTransferKind;
  item: Exclude<HeldItem, null>;
}

export interface AiPolicyLiveness {
  contractVersion: typeof AI_LIVENESS_CONTRACT_VERSION;
  consecutiveWaitSteps: number;
  recentObjectTransfer: AiObjectTransfer | null;
  cycleBlockedStashItem: Exclude<HeldItem, null> | null;
}

export interface KitchenOrder {
  id: string;
  recipe: 'two-tomato-one-onion-soup';
  reward: number;
}

export interface GameEvent {
  code: GameEventCode;
  actor: GameActor | null;
  item?: Exclude<HeldItem, null>;
  position?: Point;
  message: string;
}

export interface GameState {
  status: GameStatus;
  score: number;
  secondsLeft: number;
  ordersCompleted: number;
  tick: number;
  player: Chef;
  partner: Chef;
  pot: PotState;
  counterObjects: CounterObjects;
  order: KitchenOrder;
  lastAction: string;
  lastEventCode: GameEventCode;
  lastStepEvents: GameEvent[];
  aiPolicyLiveness: AiPolicyLiveness;
}

const DIRECTION_DELTAS: Record<Direction, Point> = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
};

const DIRECTIONS: ReadonlyArray<[Direction, Point]> = [
  ['up', DIRECTION_DELTAS.up],
  ['left', DIRECTION_DELTAS.left],
  ['right', DIRECTION_DELTAS.right],
  ['down', DIRECTION_DELTAS.down],
];

function newOrder(index: number): KitchenOrder {
  return {
    id: `two-tomato-one-onion-soup-${index}`,
    recipe: 'two-tomato-one-onion-soup',
    reward: CORRECT_SOUP_REWARD,
  };
}

function emptyPot(): PotState {
  return { stage: 'empty', secondsRemaining: 0, tomatoes: 0, onions: 0 };
}

function event(
  code: GameEventCode,
  message: string,
  actor: GameActor | null = null,
  extra: Pick<GameEvent, 'item' | 'position'> = {},
): GameEvent {
  return { code, actor, message, ...extra };
}

function withEvents(state: GameState, events: GameEvent[]): GameState {
  const effectiveEvents = events.length > 0
    ? events
    : [event('wait', 'Both chefs waited.')];
  return {
    ...state,
    lastAction: effectiveEvents.map((item) => item.message).join(' '),
    lastEventCode: effectiveEvents[effectiveEvents.length - 1].code,
    lastStepEvents: effectiveEvents,
  };
}

export function createGameState(status: GameStatus = 'waiting'): GameState {
  const initialEvent = event('waiting', 'Waiting for research consent.');
  return {
    status,
    score: 0,
    secondsLeft: ROUND_SECONDS,
    ordersCompleted: 0,
    tick: 0,
    player: { ...PLAYER_STARTS.player, held: null, facing: 'up' },
    partner: { ...PLAYER_STARTS.partner, held: null, facing: 'up' },
    pot: emptyPot(),
    counterObjects: {},
    order: newOrder(1),
    lastAction: initialEvent.message,
    lastEventCode: initialEvent.code,
    lastStepEvents: [initialEvent],
    aiPolicyLiveness: {
      contractVersion: AI_LIVENESS_CONTRACT_VERSION,
      consecutiveWaitSteps: 0,
      recentObjectTransfer: null,
      cycleBlockedStashItem: null,
    },
  };
}

export function startGame(state: GameState): GameState {
  if (state.status !== 'waiting') return state;
  return withEvents(
    { ...state, status: 'running' },
    [event('game_started', 'The kitchen is open.')],
  );
}

export function togglePause(state: GameState): GameState {
  if (state.status === 'running') {
    return withEvents(
      { ...state, status: 'paused' },
      [event('game_paused', 'Game paused.')],
    );
  }
  if (state.status === 'paused') {
    return withEvents(
      { ...state, status: 'running' },
      [event('game_resumed', 'Game resumed.')],
    );
  }
  return state;
}

export function isWalkable(point: Point): boolean {
  return (
    point.x >= 0 &&
    point.x < BOARD_WIDTH &&
    point.y >= 0 &&
    point.y < BOARD_HEIGHT &&
    TERRAIN_ROWS[point.y][point.x] === ' '
  );
}

export function isCounter(point: Point): boolean {
  return (
    point.x >= 0 &&
    point.x < BOARD_WIDTH &&
    point.y >= 0 &&
    point.y < BOARD_HEIGHT &&
    TERRAIN_ROWS[point.y][point.x] === 'X'
  );
}

export function counterKey(point: Point): string {
  return `${point.x},${point.y}`;
}

function isSamePoint(left: Point, right: Point): boolean {
  return left.x === right.x && left.y === right.y;
}

function pointAhead(chef: Chef): Point {
  const delta = DIRECTION_DELTAS[chef.facing];
  return { x: chef.x + delta.x, y: chef.y + delta.y };
}

function stationAhead(chef: Chef): StationKind | null {
  const ahead = pointAhead(chef);
  return STATIONS.find((station) => isSamePoint(ahead, station.position))?.kind ?? null;
}

function actorName(actor: GameActor): string {
  return actor === 'ai' ? 'AI' : 'Human';
}

function cloneSoupContents(contents: SoupContents | undefined): SoupContents | undefined {
  return contents ? { ...contents } : undefined;
}

interface InteractionResult {
  chef: Chef;
  pot: PotState;
  counterObjects: CounterObjects;
  reward: number;
  completedOrder: boolean;
  event: GameEvent;
}

function failedInteraction(
  chef: Chef,
  pot: PotState,
  counterObjects: CounterObjects,
  actor: GameActor,
): InteractionResult {
  return {
    chef,
    pot,
    counterObjects,
    reward: 0,
    completedOrder: false,
    event: event('interaction_failed', `${actorName(actor)} cannot interact here.`, actor),
  };
}

function interactChef(
  chef: Chef,
  pot: PotState,
  counterObjects: CounterObjects,
  actor: GameActor,
): InteractionResult {
  const ahead = pointAhead(chef);
  const name = actorName(actor);

  if (isCounter(ahead)) {
    const key = counterKey(ahead);
    const counterObject = counterObjects[key];
    if (chef.held !== null && counterObject === undefined) {
      const nextCounters = { ...counterObjects };
      nextCounters[key] = {
        item: chef.held,
        soupContents: chef.held === 'soup' ? cloneSoupContents(chef.heldSoup) : undefined,
      };
      const droppedItem = chef.held;
      return {
        chef: { ...chef, held: null, heldSoup: undefined },
        pot,
        counterObjects: nextCounters,
        reward: 0,
        completedOrder: false,
        event: event(
          'drop_counter',
          `${name} placed ${itemLabel(droppedItem)} on a counter.`,
          actor,
          { item: droppedItem, position: ahead },
        ),
      };
    }
    if (chef.held === null && counterObject !== undefined) {
      const nextCounters = { ...counterObjects };
      delete nextCounters[key];
      return {
        chef: {
          ...chef,
          held: counterObject.item,
          heldSoup: counterObject.item === 'soup'
            ? cloneSoupContents(counterObject.soupContents)
            : undefined,
        },
        pot,
        counterObjects: nextCounters,
        reward: 0,
        completedOrder: false,
        event: event(
          'pickup_counter',
          `${name} picked up ${itemLabel(counterObject.item)} from a counter.`,
          actor,
          { item: counterObject.item, position: ahead },
        ),
      };
    }
    return failedInteraction(chef, pot, counterObjects, actor);
  }

  const station = stationAhead(chef);
  if (station === 'tomato' && chef.held === null) {
    return {
      chef: { ...chef, held: 'tomato', heldSoup: undefined },
      pot,
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event('pick_tomato', `${name} picked up a tomato.`, actor, { item: 'tomato' }),
    };
  }
  if (station === 'onion' && chef.held === null) {
    return {
      chef: { ...chef, held: 'onion', heldSoup: undefined },
      pot,
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event('pick_onion', `${name} picked up an onion.`, actor, { item: 'onion' }),
    };
  }
  if (station === 'dish' && chef.held === null) {
    return {
      chef: { ...chef, held: 'dish', heldSoup: undefined },
      pot,
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event('pick_dish', `${name} picked up a dish.`, actor, { item: 'dish' }),
    };
  }

  const ingredientCount = pot.tomatoes + pot.onions;
  if (
    station === 'pot' &&
    (chef.held === 'tomato' || chef.held === 'onion') &&
    ingredientCount < 3 &&
    (pot.stage === 'empty' || pot.stage === 'filling')
  ) {
    const ingredient = chef.held;
    const nextPot = {
      ...pot,
      stage: 'filling' as const,
      tomatoes: pot.tomatoes + Number(ingredient === 'tomato'),
      onions: pot.onions + Number(ingredient === 'onion'),
    };
    return {
      chef: { ...chef, held: null, heldSoup: undefined },
      pot: nextPot,
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event(
        ingredient === 'tomato' ? 'pot_tomato' : 'pot_onion',
        `${name} added ${itemLabel(ingredient)} to the pot.`,
        actor,
        { item: ingredient },
      ),
    };
  }
  if (station === 'pot' && chef.held === 'dish' && pot.stage === 'ready') {
    return {
      chef: {
        ...chef,
        held: 'soup',
        heldSoup: { tomatoes: pot.tomatoes, onions: pot.onions },
      },
      pot: emptyPot(),
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event('pick_soup', `${name} plated the cooked soup.`, actor, { item: 'soup' }),
    };
  }
  if (station === 'serve' && chef.held === 'soup') {
    const contents = chef.heldSoup ?? { tomatoes: 2, onions: 1 };
    const correct = contents.tomatoes === 2 && contents.onions === 1;
    return {
      chef: { ...chef, held: null, heldSoup: undefined },
      pot,
      counterObjects,
      reward: correct ? CORRECT_SOUP_REWARD : 0,
      completedOrder: correct,
      event: event(
        correct ? 'serve_correct_soup' : 'serve_wrong_soup',
        correct
          ? `${name} delivered the correct tomato-onion soup.`
          : `${name} delivered a soup that does not match the order.`,
        actor,
        { item: 'soup' },
      ),
    };
  }

  return failedInteraction(chef, pot, counterObjects, actor);
}

function applyActorInteraction(
  state: GameState,
  chefKey: 'player' | 'partner',
  actor: GameActor,
): { state: GameState; event: GameEvent } {
  const result = interactChef(
    state[chefKey],
    state.pot,
    state.counterObjects,
    actor,
  );
  const completed = state.ordersCompleted + Number(result.completedOrder);
  return {
    state: {
      ...state,
      [chefKey]: result.chef,
      pot: result.pot,
      counterObjects: result.counterObjects,
      score: state.score + result.reward,
      ordersCompleted: completed,
      order: result.completedOrder ? newOrder(completed + 1) : state.order,
    },
    event: result.event,
  };
}

function intendedChef(chef: Chef, action: GameAction): Chef {
  if (!(action in DIRECTION_DELTAS)) return chef;
  const direction = action as Direction;
  const delta = DIRECTION_DELTAS[direction];
  const target = { x: chef.x + delta.x, y: chef.y + delta.y };
  return {
    ...chef,
    ...(isWalkable(target) ? target : { x: chef.x, y: chef.y }),
    facing: direction,
  };
}

function resolveMovements(
  state: GameState,
  aiAction: GameAction,
  humanAction: GameAction,
): { state: GameState; events: GameEvent[] } {
  const oldAi = state.partner;
  const oldHuman = state.player;
  const intendedAi = intendedChef(oldAi, aiAction);
  const intendedHuman = intendedChef(oldHuman, humanAction);
  const collision =
    isSamePoint(intendedAi, intendedHuman) ||
    (isSamePoint(intendedAi, oldHuman) && isSamePoint(intendedHuman, oldAi));
  const nextAi = collision
    ? { ...oldAi, facing: intendedAi.facing }
    : intendedAi;
  const nextHuman = collision
    ? { ...oldHuman, facing: intendedHuman.facing }
    : intendedHuman;
  const events: GameEvent[] = [];

  if (collision && (aiAction in DIRECTION_DELTAS || humanAction in DIRECTION_DELTAS)) {
    events.push(event('movement_collision', 'Both chefs stayed in place because their moves collided.'));
  } else {
    for (const [actor, action, before, after] of [
      ['ai', aiAction, oldAi, nextAi],
      ['human', humanAction, oldHuman, nextHuman],
    ] as const) {
      if (!(action in DIRECTION_DELTAS)) continue;
      if (isSamePoint(before, after)) {
        events.push(event('movement_blocked', `${actorName(actor)} faced a blocked tile.`, actor));
      } else {
        events.push(
          event('move', `${actorName(actor)} moved to ${after.x},${after.y}.`, actor, {
            position: { x: after.x, y: after.y },
          }),
        );
      }
    }
  }

  return {
    state: { ...state, partner: nextAi, player: nextHuman },
    events,
  };
}

function applyEnvironmentEffects(
  state: GameState,
): { state: GameState; events: GameEvent[] } {
  const events: GameEvent[] = [];
  let pot = state.pot;
  if (pot.stage === 'filling' && pot.tomatoes + pot.onions === 3) {
    pot = { ...pot, stage: 'cooking', secondsRemaining: COOK_TIME_STEPS };
    events.push(event('cooking_started', 'The full pot started cooking automatically.'));
  }
  if (pot.stage === 'cooking') {
    if (pot.secondsRemaining <= 1) {
      pot = { ...pot, stage: 'ready', secondsRemaining: 0 };
      events.push(event('soup_ready', 'The soup is ready.'));
    } else {
      pot = { ...pot, secondsRemaining: pot.secondsRemaining - 1 };
    }
  }

  const tick = Math.min(ROUND_STEPS, state.tick + 1);
  const stepsLeft = Math.max(0, ROUND_STEPS - tick);
  const status: GameStatus = stepsLeft === 0 ? 'finished' : state.status;
  if (status === 'finished') {
    events.push(event('round_finished', 'The 800-step round is complete.'));
  }
  return {
    state: {
      ...state,
      tick,
      secondsLeft: Math.ceil(stepsLeft / STEPS_PER_SECOND),
      status,
      pot,
    },
    events,
  };
}

function aiObjectTransfer(events: readonly GameEvent[]): AiObjectTransfer | null {
  const aiEvent = events.find(
    (item) =>
      item.actor === 'ai' &&
      ['pick_tomato', 'pick_onion', 'pick_dish', 'pick_soup', 'pickup_counter', 'drop_counter'].includes(
        item.code,
      ),
  );
  if (!aiEvent?.item) return null;
  return {
    kind: aiEvent.code === 'drop_counter' ? 'stash' : 'get',
    item: aiEvent.item,
  };
}

function updateAiPolicyLiveness(
  previous: GameState,
  aiAction: GameAction,
  events: readonly GameEvent[],
): AiPolicyLiveness {
  const prior = previous.aiPolicyLiveness ?? {
    contractVersion: AI_LIVENESS_CONTRACT_VERSION,
    consecutiveWaitSteps: 0,
    recentObjectTransfer: null,
    cycleBlockedStashItem: null,
  };
  const taskProgressed = events.some((item) =>
    [
      'pot_tomato',
      'pot_onion',
      'pick_soup',
      'serve_correct_soup',
      'serve_wrong_soup',
      'cooking_started',
      'soup_ready',
    ].includes(item.code),
  );
  const transfer = aiObjectTransfer(events);
  const completedGetStashCycle = Boolean(
    transfer?.kind === 'stash' &&
      prior.recentObjectTransfer?.kind === 'get' &&
      transfer.item === prior.recentObjectTransfer.item,
  );

  return {
    contractVersion: AI_LIVENESS_CONTRACT_VERSION,
    consecutiveWaitSteps:
      aiAction === 'stay' && !taskProgressed
        ? prior.consecutiveWaitSteps + 1
        : 0,
    recentObjectTransfer: taskProgressed
      ? null
      : transfer ?? prior.recentObjectTransfer,
    cycleBlockedStashItem: taskProgressed
      ? null
      : completedGetStashCycle
        ? transfer!.item
        : prior.cycleBlockedStashItem,
  };
}

/**
 * Apply one authoritative Overcooked environment step.
 *
 * The argument order intentionally matches Pygame's env.multi_step call:
 * player 0 / AI first, then player 1 / human. Interactions resolve in that
 * order, movements resolve jointly, and old-dynamics cooking advances last.
 */
export function stepGame(
  state: GameState,
  aiAction: GameAction,
  humanAction: GameAction,
): GameState {
  if (state.status !== 'running') return state;
  let next = state;
  const events: GameEvent[] = [];

  if (aiAction === 'interact') {
    const result = applyActorInteraction(next, 'partner', 'ai');
    next = result.state;
    events.push(result.event);
  }
  if (humanAction === 'interact') {
    const result = applyActorInteraction(next, 'player', 'human');
    next = result.state;
    events.push(result.event);
  }

  const movement = resolveMovements(next, aiAction, humanAction);
  next = movement.state;
  events.push(...movement.events);

  const environment = applyEnvironmentEffects(next);
  next = environment.state;
  events.push(...environment.events);
  next = {
    ...next,
    aiPolicyLiveness: updateAiPolicyLiveness(state, aiAction, events),
  };
  return withEvents(next, events);
}

/**
 * Compatibility wrapper used by the existing Web controls. The canonical
 * fixed-rate integration should call stepGame once every 500 ms.
 */
export function movePlayer(state: GameState, direction: Direction): GameState {
  if (state.status !== 'running') return state;
  const movement = resolveMovements(state, 'stay', direction);
  return withEvents(movement.state, movement.events);
}

/** Compatibility wrapper used by the existing Web controls. */
export function interactPlayer(state: GameState): GameState {
  if (state.status !== 'running') return state;
  const result = applyActorInteraction(state, 'player', 'human');
  return withEvents(result.state, [result.event]);
}

interface InteractionGoal {
  target: Point;
  access: Point;
  facing: Direction;
}

function pathDistance(start: Point, target: Point, blocked: Point | null): number {
  if (isSamePoint(start, target)) return 0;
  const key = (point: Point) => counterKey(point);
  const queue: Array<{ point: Point; distance: number }> = [{ point: start, distance: 0 }];
  const visited = new Set([key(start)]);
  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const [, delta] of DIRECTIONS) {
      const point = { x: current.point.x + delta.x, y: current.point.y + delta.y };
      if (
        !isWalkable(point) ||
        (blocked !== null && isSamePoint(point, blocked)) ||
        visited.has(key(point))
      ) continue;
      if (isSamePoint(point, target)) return current.distance + 1;
      visited.add(key(point));
      queue.push({ point, distance: current.distance + 1 });
    }
  }
  return Number.POSITIVE_INFINITY;
}

function interactionGoal(
  state: GameState,
  target: Point,
): InteractionGoal | null {
  const candidates: InteractionGoal[] = [];
  for (const [facing, delta] of DIRECTIONS) {
    const access = { x: target.x - delta.x, y: target.y - delta.y };
    if (!isWalkable(access)) continue;
    candidates.push({ target, access, facing });
  }
  candidates.sort(
    (left, right) =>
      pathDistance(state.partner, left.access, state.player) -
      pathDistance(state.partner, right.access, state.player),
  );
  return candidates.find(
    (candidate) => Number.isFinite(pathDistance(state.partner, candidate.access, state.player)),
  ) ?? null;
}

function stationGoal(state: GameState, kind: StationKind): InteractionGoal | null {
  const station = STATIONS.find((item) => item.kind === kind);
  return station ? interactionGoal(state, station.position) : null;
}

function counterGoals(
  state: GameState,
  predicate: (object: CounterObject | undefined) => boolean,
): InteractionGoal[] {
  const goals: InteractionGoal[] = [];
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const point = { x, y };
      if (!isCounter(point) || !predicate(state.counterObjects[counterKey(point)])) continue;
      const goal = interactionGoal(state, point);
      if (goal) goals.push(goal);
    }
  }
  return goals.sort(
    (left, right) => {
      const distanceDifference =
        pathDistance(state.partner, left.access, state.player) -
        pathDistance(state.partner, right.access, state.player);
      if (distanceDifference !== 0) return distanceDifference;
      const leftTurnCost =
        isSamePoint(state.partner, left.access) && state.partner.facing === left.facing ? 0 : 1;
      const rightTurnCost =
        isSamePoint(state.partner, right.access) && state.partner.facing === right.facing ? 0 : 1;
      return leftTurnCost - rightTurnCost;
    },
  );
}

function actionTowardGoal(state: GameState, goal: InteractionGoal | null): GameAction {
  if (!goal) return 'stay';
  if (isSamePoint(state.partner, goal.access)) {
    return state.partner.facing === goal.facing ? 'interact' : goal.facing;
  }

  const key = (point: Point) => counterKey(point);
  const queue: Array<{ point: Point; first?: Direction }> = [{ point: state.partner }];
  const visited = new Set([key(state.partner)]);
  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const [direction, delta] of DIRECTIONS) {
      const point = { x: current.point.x + delta.x, y: current.point.y + delta.y };
      if (!isWalkable(point) || isSamePoint(point, state.player) || visited.has(key(point))) continue;
      const first = current.first ?? direction;
      if (isSamePoint(point, goal.access)) return first;
      visited.add(key(point));
      queue.push({ point, first });
    }
  }
  return 'stay';
}

function firstCounterGoal(
  state: GameState,
  predicate: (object: CounterObject | undefined) => boolean,
): InteractionGoal | null {
  return counterGoals(state, predicate)[0] ?? null;
}

function stationPosition(kind: StationKind): Point | null {
  return STATIONS.find((station) => station.kind === kind)?.position ?? null;
}

function interactionDistance(
  start: Point,
  target: Point,
  blocked: Point | null,
): number {
  let best = Number.POSITIVE_INFINITY;
  for (const [, delta] of DIRECTIONS) {
    const access = { x: target.x - delta.x, y: target.y - delta.y };
    if (!isWalkable(access) || (blocked !== null && isSamePoint(access, blocked))) continue;
    best = Math.min(best, pathDistance(start, access, blocked));
  }
  return best;
}

function targetForAiSubgoal(state: GameState, subgoal: Subgoal): Point | null {
  switch (subgoal) {
    case 'GET_TOMATO':
      return (
        firstCounterGoal(state, (object) => object?.item === 'tomato')?.target ??
        stationPosition('tomato')
      );
    case 'PUT_TOMATO_IN_POT':
    case 'PUT_ONION_IN_POT':
      return stationPosition('pot');
    case 'GET_ONION':
      return (
        firstCounterGoal(state, (object) => object?.item === 'onion')?.target ??
        stationPosition('onion')
      );
    case 'GET_DISH':
      return (
        firstCounterGoal(state, (object) => object?.item === 'dish')?.target ??
        stationPosition('dish')
      );
    case 'PICKUP_SOUP':
      return state.partner.held === 'dish'
        ? stationPosition('pot')
        : firstCounterGoal(state, (object) => object?.item === 'soup')?.target ?? null;
    case 'SERVE_SOUP':
      return stationPosition('serve');
    case 'STASH_HELD_OBJECT':
      return firstCounterGoal(state, (object) => object === undefined)?.target ?? null;
    case 'YIELD_PATH':
    case 'WAIT':
      return null;
  }
}

function humanTargetFromState(state: GameState): { resource: string; target: Point } | null {
  switch (state.player.held) {
    case 'tomato':
      return { resource: 'tomato', target: stationPosition('pot')! };
    case 'onion':
      return { resource: 'onion', target: stationPosition('pot')! };
    case 'dish':
      return { resource: 'dish', target: stationPosition('pot')! };
    case 'soup':
      return { resource: 'soup', target: stationPosition('serve')! };
    default: {
      // With no action history in GameState, only infer pickup intent when the
      // human is already adjacent to and facing an immediately interactable
      // object. This makes target-stealing live without guessing from position
      // alone or treating a stale movement direction as intent.
      const target = pointAhead(state.player);
      const counterObject = isCounter(target)
        ? state.counterObjects[counterKey(target)]
        : undefined;
      if (counterObject) return { resource: counterObject.item, target };

      const station = STATIONS.find((item) => isSamePoint(item.position, target));
      if (
        station?.kind === 'tomato' ||
        station?.kind === 'onion' ||
        station?.kind === 'dish'
      ) {
        return { resource: station.kind, target };
      }
      return null;
    }
  }
}

function intendedSoloPosition(state: GameState, action: GameAction): Point {
  if (!(action in DIRECTION_DELTAS)) return state.partner;
  const delta = DIRECTION_DELTAS[action as Direction];
  const target = { x: state.partner.x + delta.x, y: state.partner.y + delta.y };
  return isWalkable(target) && !isSamePoint(target, state.player)
    ? target
    : state.partner;
}

function computeYieldPathAction(state: GameState): Direction | null {
  const humanTarget = humanTargetFromState(state);
  if (!humanTarget) return null;
  const currentDistance = interactionDistance(
    state.player,
    humanTarget.target,
    state.partner,
  );
  const candidates = DIRECTIONS.flatMap(([direction, delta]) => {
    const destination = {
      x: state.partner.x + delta.x,
      y: state.partner.y + delta.y,
    };
    if (!isWalkable(destination) || isSamePoint(destination, state.player)) return [];
    const resultingDistance = interactionDistance(
      state.player,
      humanTarget.target,
      destination,
    );
    // YIELD_PATH is the shortest-path clearing subgoal. Merely stepping out of
    // the cell the human faces is not enough to claim it; the move must reduce
    // the human's actual route distance to the inferred interaction target.
    if (!(resultingDistance < currentDistance)) return [];
    return [{ direction, resultingDistance }];
  });
  candidates.sort(
    (left, right) => left.resultingDistance - right.resultingDistance,
  );
  return candidates[0]?.direction ?? null;
}

export interface LivePolicyGeometry {
  candidatePathEffects: Readonly<Partial<Record<Subgoal, PathEffect>>>;
  candidateTargetOverlapsHuman: Readonly<Partial<Record<Subgoal, boolean>>>;
  humanIntent: string | null;
  yieldAction: Direction | null;
}

/** Pure state geometry shared by live feasibility and reward featurization. */
export function computeLivePolicyGeometry(state: GameState): LivePolicyGeometry {
  const humanTarget = humanTargetFromState(state);
  const baselineDistance = humanTarget
    ? interactionDistance(state.player, humanTarget.target, null)
    : Number.POSITIVE_INFINITY;
  const currentDistance = humanTarget
    ? interactionDistance(state.player, humanTarget.target, state.partner)
    : Number.POSITIVE_INFINITY;
  const candidatePathEffects: Partial<Record<Subgoal, PathEffect>> = {};
  const candidateTargetOverlapsHuman: Partial<Record<Subgoal, boolean>> = {};
  const yieldAction = computeYieldPathAction(state);

  for (const subgoal of SUBGOALS) {
    const target = targetForAiSubgoal(state, subgoal);
    if (humanTarget && target) {
      candidateTargetOverlapsHuman[subgoal] = isSamePoint(target, humanTarget.target);
    }
    const action = subgoal === 'YIELD_PATH'
      ? yieldAction ?? 'stay'
      : actionForAiSubgoal(state, subgoal);
    const nextPosition = intendedSoloPosition(state, action);
    if (humanTarget) {
      const nextDistance = interactionDistance(
        state.player,
        humanTarget.target,
        nextPosition,
      );
      if (nextDistance < currentDistance) {
        candidatePathEffects[subgoal] = 'clears';
      } else if (nextDistance > currentDistance) {
        candidatePathEffects[subgoal] = currentDistance <= baselineDistance
          ? 'enters'
          : 'blocks';
      } else if (currentDistance > baselineDistance) {
        candidatePathEffects[subgoal] = 'blocks';
      }
    }
  }
  if (yieldAction) candidatePathEffects.YIELD_PATH = 'clears';

  return {
    candidatePathEffects,
    candidateTargetOverlapsHuman,
    humanIntent: humanTarget?.resource ?? null,
    yieldAction,
  };
}

export function actionForAiSubgoal(state: GameState, subgoal: Subgoal): GameAction {
  switch (subgoal) {
    case 'GET_TOMATO':
      return actionTowardGoal(
        state,
        firstCounterGoal(state, (object) => object?.item === 'tomato') ??
          stationGoal(state, 'tomato'),
      );
    case 'PUT_TOMATO_IN_POT':
    case 'PUT_ONION_IN_POT':
      return actionTowardGoal(state, stationGoal(state, 'pot'));
    case 'GET_ONION':
      return actionTowardGoal(
        state,
        firstCounterGoal(state, (object) => object?.item === 'onion') ??
          stationGoal(state, 'onion'),
      );
    case 'GET_DISH':
      return actionTowardGoal(
        state,
        firstCounterGoal(state, (object) => object?.item === 'dish') ??
          stationGoal(state, 'dish'),
      );
    case 'PICKUP_SOUP':
      return actionTowardGoal(
        state,
        state.partner.held === 'dish'
          ? stationGoal(state, 'pot')
          : firstCounterGoal(state, (object) => object?.item === 'soup'),
      );
    case 'SERVE_SOUP':
      return actionTowardGoal(state, stationGoal(state, 'serve'));
    case 'STASH_HELD_OBJECT':
      return actionTowardGoal(
        state,
        firstCounterGoal(state, (object) => object === undefined),
      );
    case 'YIELD_PATH':
      return computeYieldPathAction(state) ?? 'stay';
    case 'WAIT':
      return 'stay';
  }
}

export interface AiDecision extends RewardSubgoalDecision {
  action: GameAction;
  candidateActions: Readonly<Partial<Record<Subgoal, GameAction>>>;
  candidatePathEffects: Readonly<Partial<Record<Subgoal, PathEffect>>>;
  candidateTargetOverlapsHuman: Readonly<Partial<Record<Subgoal, boolean>>>;
  yieldAction: Direction | null;
  motionFeasibilityFilterApplied: boolean;
  motionUnexecutableSubgoals: readonly Subgoal[];
  waitGuardApplied: boolean;
  livenessReason:
    | 'stalled_wait_infeasible'
    | 'passive_cooking_wait_exempt'
    | 'get_stash_cycle_infeasible'
    | 'stall_threshold_not_reached'
    | 'no_executable_productive_alternative';
  livenessRemovedSubgoals: readonly Subgoal[];
  getStashCycleGuardApplied: boolean;
  getStashCycleBlockedItem: Exclude<HeldItem, null> | null;
}

/** Task-valid subgoals are ranked by the active posterior mean via w · phi. */
export function chooseAiDecision(
  state: GameState,
  weights: RewardWeights = {},
): AiDecision {
  const geometry = computeLivePolicyGeometry(state);
  const contextOverrides: Partial<SubgoalContext> = {
    humanIntent: geometry.humanIntent,
    candidatePathEffects: geometry.candidatePathEffects,
    candidateTargetOverlapsHuman: geometry.candidateTargetOverlapsHuman,
  };
  const context = buildSubgoalContext(state, contextOverrides);
  const taskFeasible = enumerateFeasibleSubgoals(context);
  const candidateActions: Partial<Record<Subgoal, GameAction>> = {};
  for (const subgoal of taskFeasible) {
    candidateActions[subgoal] = subgoal === 'YIELD_PATH'
      ? geometry.yieldAction ?? 'stay'
      : actionForAiSubgoal(state, subgoal);
  }

  const motionUnexecutableSubgoals = taskFeasible.filter(
    (subgoal) => subgoal !== 'WAIT' && candidateActions[subgoal] === 'stay',
  );
  let feasible = taskFeasible.filter(
    (subgoal) => subgoal === 'WAIT' || candidateActions[subgoal] !== 'stay',
  );
  if (feasible.length === 0) feasible = ['WAIT'];

  const livenessRemoved = new Set<Subgoal>(motionUnexecutableSubgoals);
  const blockedItem = state.aiPolicyLiveness?.cycleBlockedStashItem ?? null;
  const requiredPutSubgoal = blockedItem === 'tomato'
    ? 'PUT_TOMATO_IN_POT'
    : blockedItem === 'onion'
      ? 'PUT_ONION_IN_POT'
      : null;
  const getStashCycleGuardApplied = Boolean(
    blockedItem !== null &&
      state.partner.held === blockedItem &&
      requiredPutSubgoal !== null &&
      feasible.includes('STASH_HELD_OBJECT') &&
      feasible.includes(requiredPutSubgoal) &&
      candidateActions[requiredPutSubgoal] !== 'stay',
  );
  if (getStashCycleGuardApplied) {
    feasible = feasible.filter(
      (subgoal) => subgoal !== 'STASH_HELD_OBJECT' && subgoal !== 'WAIT',
    );
    livenessRemoved.add('STASH_HELD_OBJECT');
    livenessRemoved.add('WAIT');
  }

  const executableProductive = feasible.filter(
    (subgoal) => subgoal !== 'WAIT' && candidateActions[subgoal] !== 'stay',
  );
  const stalledWaitThresholdReached =
    (state.aiPolicyLiveness?.consecutiveWaitSteps ?? 0) >=
    MAX_CONSECUTIVE_WAIT_STEPS;
  const passiveCookingWait =
    passiveCookingWaitIsValid(context) &&
    geometry.candidatePathEffects.WAIT !== 'blocks';
  const waitGuardApplied = Boolean(
    !getStashCycleGuardApplied &&
      stalledWaitThresholdReached &&
      executableProductive.length > 0 &&
      !passiveCookingWait &&
      feasible.includes('WAIT'),
  );
  if (waitGuardApplied) {
    feasible = feasible.filter((subgoal) => subgoal !== 'WAIT');
    livenessRemoved.add('WAIT');
  }

  const rewardDecision = chooseRewardSubgoal(state, weights, {
    contextOverrides,
    feasibleSubgoals: feasible,
  });
  const action = state.status === 'running'
    ? candidateActions[rewardDecision.chosenSubgoal] ??
      actionForAiSubgoal(state, rewardDecision.chosenSubgoal)
    : 'stay';
  const livenessReason = getStashCycleGuardApplied
    ? 'get_stash_cycle_infeasible'
    : waitGuardApplied
      ? 'stalled_wait_infeasible'
      : passiveCookingWait && stalledWaitThresholdReached
        ? 'passive_cooking_wait_exempt'
        : executableProductive.length === 0
          ? 'no_executable_productive_alternative'
          : 'stall_threshold_not_reached';
  return {
    ...rewardDecision,
    action,
    candidateActions,
    candidatePathEffects: geometry.candidatePathEffects,
    candidateTargetOverlapsHuman: geometry.candidateTargetOverlapsHuman,
    yieldAction: geometry.yieldAction,
    motionFeasibilityFilterApplied: motionUnexecutableSubgoals.length > 0,
    motionUnexecutableSubgoals,
    waitGuardApplied,
    livenessReason,
    livenessRemovedSubgoals: [...livenessRemoved],
    getStashCycleGuardApplied,
    getStashCycleBlockedItem: blockedItem,
  };
}

export function chooseAiAction(
  state: GameState,
  weights: RewardWeights = {},
): GameAction {
  return chooseAiDecision(state, weights).action;
}

/**
 * One wall-clock second for the current page timer: two authoritative 2 Hz
 * environment decisions with a waiting human. New integrations should prefer
 * stepGame directly at a 500 ms cadence so human input is sampled jointly.
 */
export function tickGame(state: GameState): GameState {
  if (state.status !== 'running') return state;
  let next = state;
  for (let step = 0; step < STEPS_PER_SECOND && next.status === 'running'; step += 1) {
    next = stepGame(next, chooseAiAction(next), 'stay');
  }
  return next;
}

export function itemLabel(item: HeldItem): string {
  switch (item) {
    case 'tomato':
      return 'tomato';
    case 'onion':
      return 'onion';
    case 'dish':
      return 'dish';
    case 'soup':
      return 'tomato-onion soup';
    default:
      return 'empty-handed';
  }
}
