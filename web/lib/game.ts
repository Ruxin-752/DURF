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

function pathDistance(start: Point, target: Point, blocked: Point): number {
  if (isSamePoint(start, target)) return 0;
  const key = (point: Point) => counterKey(point);
  const queue: Array<{ point: Point; distance: number }> = [{ point: start, distance: 0 }];
  const visited = new Set([key(start)]);
  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const [, delta] of DIRECTIONS) {
      const point = { x: current.point.x + delta.x, y: current.point.y + delta.y };
      if (!isWalkable(point) || isSamePoint(point, blocked) || visited.has(key(point))) continue;
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

export function chooseAiAction(state: GameState): GameAction {
  if (state.status !== 'running') return 'stay';
  const held = state.partner.held;
  const totalIngredients = state.pot.tomatoes + state.pot.onions;
  const needsTomato = totalIngredients < 3 && state.pot.tomatoes < 2;
  const needsOnion = totalIngredients < 3 && state.pot.onions < 1;

  if (held === 'soup') return actionTowardGoal(state, stationGoal(state, 'serve'));
  if (held === 'dish') {
    if (state.pot.stage === 'ready') {
      return actionTowardGoal(state, stationGoal(state, 'pot'));
    }
    if (state.pot.stage === 'cooking') return 'stay';
    return actionTowardGoal(state, firstCounterGoal(state, (object) => object === undefined));
  }
  if (held === 'tomato' || held === 'onion') {
    const ingredientNeeded = held === 'tomato' ? needsTomato : needsOnion;
    if (
      ingredientNeeded &&
      (state.pot.stage === 'empty' || state.pot.stage === 'filling')
    ) {
      return actionTowardGoal(state, stationGoal(state, 'pot'));
    }
    return actionTowardGoal(state, firstCounterGoal(state, (object) => object === undefined));
  }

  const soupOnCounter = firstCounterGoal(state, (object) => object?.item === 'soup');
  if (soupOnCounter) return actionTowardGoal(state, soupOnCounter);
  if (state.pot.stage === 'ready') {
    const dishOnCounter = firstCounterGoal(state, (object) => object?.item === 'dish');
    return actionTowardGoal(state, dishOnCounter ?? stationGoal(state, 'dish'));
  }
  if (state.pot.stage === 'cooking') return 'stay';

  if (needsTomato) {
    const tomatoOnCounter = firstCounterGoal(state, (object) => object?.item === 'tomato');
    return actionTowardGoal(state, tomatoOnCounter ?? stationGoal(state, 'tomato'));
  }
  if (needsOnion) {
    const onionOnCounter = firstCounterGoal(state, (object) => object?.item === 'onion');
    return actionTowardGoal(state, onionOnCounter ?? stationGoal(state, 'onion'));
  }
  return 'stay';
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
