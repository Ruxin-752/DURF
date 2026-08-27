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
export const ROUND_SECONDS = 90;

export type GameStatus = 'waiting' | 'running' | 'paused' | 'finished';
export type HeldItem = 'tomato' | 'onion' | 'dish' | 'soup' | null;
export type Direction = 'up' | 'down' | 'left' | 'right';
export type StationKind = 'tomato' | 'onion' | 'dish' | 'pot' | 'serve';

export interface Point {
  x: number;
  y: number;
}

export const PLAYER_STARTS = {
  player: { x: 3, y: 1 },
  partner: { x: 6, y: 1 },
} as const;

export const STATIONS: ReadonlyArray<{
  kind: StationKind;
  terrain: 'T' | 'O' | 'D' | 'P' | 'S';
  label: string;
  shortLabel: string;
  position: Point;
}> = [
  { kind: 'pot', terrain: 'P', label: '汤锅', shortLabel: '锅', position: { x: 4, y: 0 } },
  { kind: 'dish', terrain: 'D', label: '盘子架', shortLabel: '盘', position: { x: 0, y: 1 } },
  { kind: 'serve', terrain: 'S', label: '出餐口', shortLabel: '出餐', position: { x: 9, y: 3 } },
  { kind: 'tomato', terrain: 'T', label: '番茄食材架', shortLabel: '番茄', position: { x: 0, y: 4 } },
  { kind: 'onion', terrain: 'O', label: '洋葱食材架', shortLabel: '洋葱', position: { x: 4, y: 5 } },
];

export interface Chef extends Point {
  held: HeldItem;
  facing: Direction;
}

export interface PotState {
  stage: 'empty' | 'filling' | 'cooking' | 'ready';
  secondsRemaining: number;
  tomatoes: number;
  onions: number;
}

export interface KitchenOrder {
  id: string;
  recipe: 'two-tomato-one-onion-soup';
  reward: number;
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
  order: KitchenOrder;
  lastAction: string;
}

function newOrder(index: number): KitchenOrder {
  return { id: `two-tomato-one-onion-soup-${index}`, recipe: 'two-tomato-one-onion-soup', reward: 100 };
}

function emptyPot(): PotState {
  return { stage: 'empty', secondsRemaining: 0, tomatoes: 0, onions: 0 };
}

export function createGameState(status: GameStatus = 'waiting'): GameState {
  return {
    status,
    score: 0,
    secondsLeft: ROUND_SECONDS,
    ordersCompleted: 0,
    tick: 0,
    player: { ...PLAYER_STARTS.player, held: null, facing: 'left' },
    partner: { ...PLAYER_STARTS.partner, held: null, facing: 'right' },
    pot: emptyPot(),
    order: newOrder(1),
    lastAction: '等待研究同意',
  };
}

export function startGame(state: GameState): GameState {
  if (state.status !== 'waiting') return state;
  return { ...state, status: 'running', lastAction: '厨房开工！' };
}

export function togglePause(state: GameState): GameState {
  if (state.status === 'running') {
    return { ...state, status: 'paused', lastAction: '游戏已暂停' };
  }
  if (state.status === 'paused') {
    return { ...state, status: 'running', lastAction: '继续做菜' };
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

function isSamePoint(left: Point, right: Point): boolean {
  return left.x === right.x && left.y === right.y;
}

export function movePlayer(state: GameState, direction: Direction): GameState {
  if (state.status !== 'running') return state;
  const delta: Record<Direction, Point> = {
    up: { x: 0, y: -1 },
    down: { x: 0, y: 1 },
    left: { x: -1, y: 0 },
    right: { x: 1, y: 0 },
  };
  const target = {
    x: state.player.x + delta[direction].x,
    y: state.player.y + delta[direction].y,
  };
  if (!isWalkable(target)) {
    return { ...state, player: { ...state.player, facing: direction }, lastAction: '前面是固定设施' };
  }
  if (isSamePoint(target, state.partner)) {
    return {
      ...state,
      player: { ...state.player, facing: direction },
      lastAction: 'AI伙伴挡在前面，请换一条路',
    };
  }
  const player = { ...state.player, ...target, facing: direction };
  return { ...state, player, lastAction: `玩家移动到 ${player.x},${player.y}` };
}

function pointAhead(chef: Chef): Point {
  const delta: Record<Direction, Point> = {
    up: { x: 0, y: -1 },
    down: { x: 0, y: 1 },
    left: { x: -1, y: 0 },
    right: { x: 1, y: 0 },
  };
  return {
    x: chef.x + delta[chef.facing].x,
    y: chef.y + delta[chef.facing].y,
  };
}

function stationAhead(chef: Chef): StationKind | null {
  const ahead = pointAhead(chef);
  return STATIONS.find((station) => isSamePoint(ahead, station.position))?.kind ?? null;
}

interface InteractionResult {
  chef: Chef;
  pot: PotState;
  served: boolean;
  message: string;
}

function interactChef(chef: Chef, pot: PotState, actor: string): InteractionResult {
  const station = stationAhead(chef);
  if (station === 'tomato' && chef.held === null) {
    return { chef: { ...chef, held: 'tomato' }, pot, served: false, message: `${actor}拿了番茄` };
  }
  if (station === 'onion' && chef.held === null) {
    return { chef: { ...chef, held: 'onion' }, pot, served: false, message: `${actor}拿了洋葱` };
  }
  if (station === 'dish' && chef.held === null) {
    return { chef: { ...chef, held: 'dish' }, pot, served: false, message: `${actor}拿了盘子` };
  }
  if (
    station === 'pot' &&
    chef.held === 'tomato' &&
    pot.tomatoes < 2 &&
    (pot.stage === 'empty' || pot.stage === 'filling')
  ) {
    const tomatoes = pot.tomatoes + 1;
    const complete = tomatoes === 2 && pot.onions === 1;
    return {
      chef: { ...chef, held: null },
      pot: {
        ...pot,
        tomatoes,
        stage: complete ? 'cooking' : 'filling',
        secondsRemaining: complete ? 4 : 0,
      },
      served: false,
      message: `${actor}往锅里放了番茄（${tomatoes}/2）`,
    };
  }
  if (
    station === 'pot' &&
    chef.held === 'onion' &&
    pot.onions < 1 &&
    (pot.stage === 'empty' || pot.stage === 'filling')
  ) {
    const complete = pot.tomatoes === 2;
    return {
      chef: { ...chef, held: null },
      pot: {
        ...pot,
        onions: 1,
        stage: complete ? 'cooking' : 'filling',
        secondsRemaining: complete ? 4 : 0,
      },
      served: false,
      message: `${actor}往锅里放了洋葱（1/1）`,
    };
  }
  if (station === 'pot' && chef.held === 'dish' && pot.stage === 'ready') {
    return {
      chef: { ...chef, held: 'soup' },
      pot: emptyPot(),
      served: false,
      message: `${actor}用盘子盛起了汤`,
    };
  }
  if (station === 'serve' && chef.held === 'soup') {
    return {
      chef: { ...chef, held: null },
      pot,
      served: true,
      message: `${actor}完成一份番茄洋葱汤！`,
    };
  }
  return { chef, pot, served: false, message: `${actor}这里暂时不能操作` };
}

function applyInteraction(state: GameState, chefKey: 'player' | 'partner'): GameState {
  const actor = chefKey === 'player' ? '玩家' : 'AI伙伴';
  const result = interactChef(state[chefKey], state.pot, actor);
  const completed = state.ordersCompleted + (result.served ? 1 : 0);
  return {
    ...state,
    [chefKey]: result.chef,
    pot: result.pot,
    score: state.score + (result.served ? state.order.reward : 0),
    ordersCompleted: completed,
    order: result.served ? newOrder(completed + 1) : state.order,
    secondsLeft: result.served ? Math.min(99, state.secondsLeft + 5) : state.secondsLeft,
    lastAction: result.message,
  };
}

export function interactPlayer(state: GameState): GameState {
  return state.status === 'running' ? applyInteraction(state, 'player') : state;
}

const ACCESS_POINTS: Record<StationKind, Point> = {
  pot: { x: 4, y: 1 },
  dish: { x: 1, y: 1 },
  serve: { x: 8, y: 3 },
  tomato: { x: 1, y: 4 },
  onion: { x: 4, y: 4 },
};

const ACCESS_FACING: Record<StationKind, Direction> = {
  pot: 'up',
  dish: 'left',
  serve: 'right',
  tomato: 'left',
  onion: 'down',
};

function targetForPartner(state: GameState): Point {
  if (state.partner.held === 'tomato' || state.partner.held === 'onion') return ACCESS_POINTS.pot;
  if (state.partner.held === 'dish') return ACCESS_POINTS.pot;
  if (state.partner.held === 'soup') return ACCESS_POINTS.serve;
  if (state.pot.stage === 'ready') return ACCESS_POINTS.dish;
  if (state.pot.stage === 'cooking') return { x: 6, y: 1 };
  if (state.pot.tomatoes < 2) return ACCESS_POINTS.tomato;
  if (state.pot.onions < 1) return ACCESS_POINTS.onion;
  return { x: 6, y: 1 };
}

const DIRECTIONS: ReadonlyArray<[Direction, Point]> = [
  ['up', { x: 0, y: -1 }],
  ['left', { x: -1, y: 0 }],
  ['right', { x: 1, y: 0 }],
  ['down', { x: 0, y: 1 }],
];

function nextPathStep(start: Chef, target: Point, blocked: Point): Chef {
  if (start.x === target.x && start.y === target.y) return start;
  const key = (point: Point) => `${point.x},${point.y}`;
  const queue: Array<{ point: Point; first?: Direction }> = [{ point: start }];
  const visited = new Set([key(start)]);
  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const [direction, delta] of DIRECTIONS) {
      const point = { x: current.point.x + delta.x, y: current.point.y + delta.y };
      if (!isWalkable(point) || isSamePoint(point, blocked) || visited.has(key(point))) continue;
      const first = current.first ?? direction;
      if (point.x === target.x && point.y === target.y) {
        const firstDelta = DIRECTIONS.find(([candidate]) => candidate === first)![1];
        return {
          ...start,
          x: start.x + firstDelta.x,
          y: start.y + firstDelta.y,
          facing: first,
        };
      }
      visited.add(key(point));
      queue.push({ point, first });
    }
  }
  return start;
}

function advancePartner(state: GameState): GameState {
  const target = targetForPartner(state);
  if (state.partner.x === target.x && state.partner.y === target.y) {
    if (state.pot.stage === 'cooking' && state.partner.held === null) {
      return { ...state, lastAction: 'AI伙伴在等汤煮好' };
    }
    const station = (Object.keys(ACCESS_POINTS) as StationKind[]).find((kind) =>
      isSamePoint(ACCESS_POINTS[kind], target),
    );
    if (station && state.partner.facing !== ACCESS_FACING[station]) {
      return {
        ...state,
        partner: { ...state.partner, facing: ACCESS_FACING[station] },
        lastAction: `AI伙伴转身面向${STATIONS.find((item) => item.kind === station)?.label ?? '设施'}`,
      };
    }
    return applyInteraction(state, 'partner');
  }
  const partner = nextPathStep(state.partner, target, state.player);
  return {
    ...state,
    partner,
    lastAction: isSamePoint(partner, state.partner)
      ? 'AI伙伴在等玩家让路'
      : 'AI伙伴正在沿原地图环路协作',
  };
}

export function tickGame(state: GameState): GameState {
  if (state.status !== 'running') return state;
  const secondsLeft = Math.max(0, state.secondsLeft - 1);
  const pot =
    state.pot.stage === 'cooking'
      ? state.pot.secondsRemaining <= 1
        ? ({ ...state.pot, stage: 'ready', secondsRemaining: 0 } as const)
        : ({ ...state.pot, secondsRemaining: state.pot.secondsRemaining - 1 } as const)
      : state.pot;
  const timed = {
    ...state,
    tick: state.tick + 1,
    secondsLeft,
    pot,
    status: secondsLeft === 0 ? ('finished' as const) : state.status,
    lastAction: secondsLeft === 0 ? '本轮结束' : state.lastAction,
  };
  return secondsLeft === 0 ? timed : advancePartner(timed);
}

export function itemLabel(item: HeldItem): string {
  return (
    { tomato: '番茄', onion: '洋葱', dish: '盘子', soup: '番茄洋葱汤' } as const
  )[item as Exclude<HeldItem, null>] ?? '空手';
}
