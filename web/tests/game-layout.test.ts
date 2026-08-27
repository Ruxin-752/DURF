import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  COOK_TIME_STEPS,
  CORRECT_SOUP_REWARD,
  PLAYER_STARTS,
  ROUND_SECONDS,
  ROUND_STEPS,
  SOURCE_LAYOUT_ROWS,
  STATIONS,
  STEPS_PER_SECOND,
  TERRAIN_ROWS,
  chooseAiAction,
  counterKey,
  createGameState,
  interactPlayer,
  isCounter,
  isWalkable,
  movePlayer,
  stepGame,
  tickGame,
  togglePause,
} from '../lib/game';

const webRoot = join(import.meta.dirname, '..');
const fixturePath = join(
  webRoot,
  'tests',
  'fixtures',
  'ring_tomato_onion_10x6_h0_full_task.layout',
);
const repositoryLayoutPath = join(
  webRoot,
  '..',
  'src',
  'overcooked_ai_py',
  'data',
  'layouts',
  'ring_tomato_onion_10x6_h0_full_task.layout',
);

function normalizeNewlines(value: string): string {
  return value.replace(/\r\n/gu, '\n').trimEnd();
}

describe('immutable original Overcooked layout geometry', () => {
  it('keeps the deployed fixture content-equivalent apart from platform newlines', () => {
    expect(normalizeNewlines(readFileSync(fixturePath, 'utf8'))).toBe(
      normalizeNewlines(readFileSync(repositoryLayoutPath, 'utf8')),
    );
  });

  it('keeps every source and terrain row character in the original 10x6 position', () => {
    expect(SOURCE_LAYOUT_ROWS).toEqual([
      'XXXXPXXXXX',
      'D  1  2  X',
      'X XXXXXX X',
      'X XXXXXX S',
      'T        X',
      'XXXXOXXXXX',
    ]);
    expect(TERRAIN_ROWS).toEqual([
      'XXXXPXXXXX',
      'D        X',
      'X XXXXXX X',
      'X XXXXXX S',
      'T        X',
      'XXXXOXXXXX',
    ]);
    expect(BOARD_WIDTH).toBe(10);
    expect(BOARD_HEIGHT).toBe(6);
    expect(TERRAIN_ROWS.every((row) => row.length === BOARD_WIDTH)).toBe(true);
  });

  it('locks Pygame player identities, starts, facings, and every facility coordinate', () => {
    expect(PLAYER_STARTS).toEqual({
      partner: { x: 3, y: 1 },
      player: { x: 6, y: 1 },
    });
    const state = createGameState();
    expect(state.partner.facing).toBe('up');
    expect(state.player.facing).toBe('up');
    expect(
      Object.fromEntries(STATIONS.map((station) => [station.kind, station.position])),
    ).toEqual({
      pot: { x: 4, y: 0 },
      dish: { x: 0, y: 1 },
      serve: { x: 9, y: 3 },
      tomato: { x: 0, y: 4 },
      onion: { x: 4, y: 5 },
    });
    for (const station of STATIONS) {
      expect(SOURCE_LAYOUT_ROWS[station.position.y][station.position.x]).toBe(
        station.terrain,
      );
    }
    expect(isWalkable(PLAYER_STARTS.player)).toBe(true);
    expect(isWalkable(PLAYER_STARTS.partner)).toBe(true);
  });

  it('keeps the Pygame episode timing and recipe constants', () => {
    expect(ROUND_STEPS).toBe(800);
    expect(STEPS_PER_SECOND).toBe(2);
    expect(ROUND_SECONDS).toBe(400);
    expect(COOK_TIME_STEPS).toBe(20);
    expect(CORRECT_SOUP_REWARD).toBe(20);
  });
});

describe('Pygame-compatible joint environment steps', () => {
  it('resolves same-cell and swap collisions by keeping both positions', () => {
    const initial = createGameState('running');
    const sameCell = stepGame(
      {
        ...initial,
        partner: { ...initial.partner, x: 3, y: 1 },
        player: { ...initial.player, x: 5, y: 1 },
      },
      'right',
      'left',
    );
    expect([sameCell.partner.x, sameCell.partner.y]).toEqual([3, 1]);
    expect([sameCell.player.x, sameCell.player.y]).toEqual([5, 1]);
    expect(sameCell.partner.facing).toBe('right');
    expect(sameCell.player.facing).toBe('left');
    expect(sameCell.lastEventCode).toBe('movement_collision');

    const swapped = stepGame(
      {
        ...initial,
        partner: { ...initial.partner, x: 3, y: 1 },
        player: { ...initial.player, x: 4, y: 1 },
      },
      'right',
      'left',
    );
    expect([swapped.partner.x, swapped.partner.y]).toEqual([3, 1]);
    expect([swapped.player.x, swapped.player.y]).toEqual([4, 1]);
    expect(swapped.lastEventCode).toBe('movement_collision');
  });

  it('allows a chef to enter a cell vacated in the same non-colliding step', () => {
    const initial = createGameState('running');
    const state = stepGame(
      {
        ...initial,
        partner: { ...initial.partner, x: 3, y: 1 },
        player: { ...initial.player, x: 4, y: 1 },
      },
      'right',
      'right',
    );
    expect([state.partner.x, state.partner.y]).toEqual([4, 1]);
    expect([state.player.x, state.player.y]).toEqual([5, 1]);
  });

  it('requires facing the fixed facility and preserves compatibility controls', () => {
    const initial = createGameState('running');
    const besideDishFacingAway = {
      ...initial,
      player: { ...initial.player, x: 1, y: 1, facing: 'right' as const },
    };

    const ignored = interactPlayer(besideDishFacingAway);
    expect(ignored.player.held).toBeNull();
    expect(ignored.lastEventCode).toBe('interaction_failed');

    const facingDish = movePlayer(ignored, 'left');
    expect({ x: facingDish.player.x, y: facingDish.player.y }).toEqual({ x: 1, y: 1 });
    expect(facingDish.player.facing).toBe('left');

    const pickedUp = interactPlayer(facingDish);
    expect(pickedUp.player.held).toBe('dish');
    expect(pickedUp.lastEventCode).toBe('pick_dish');
  });

  it('freezes steps, timer, pot, and AI throughout pause', () => {
    const running = {
      ...createGameState('running'),
      tick: 125,
      secondsLeft: 338,
      pot: {
        stage: 'cooking' as const,
        secondsRemaining: 7,
        tomatoes: 2,
        onions: 1,
      },
    };
    const paused = togglePause(running);
    const afterJointStep = stepGame(paused, 'left', 'right');
    const afterTimer = tickGame(afterJointStep);
    expect(afterJointStep).toBe(paused);
    expect(afterTimer).toBe(paused);
    expect(afterTimer.tick).toBe(125);
    expect(afterTimer.secondsLeft).toBe(338);
    expect(afterTimer.pot.secondsRemaining).toBe(7);
    expect(afterTimer.partner).toEqual(paused.partner);

    const resumed = togglePause(afterTimer);
    const advanced = stepGame(resumed, 'stay', 'stay');
    expect(advanced.tick).toBe(126);
    expect(advanced.pot.secondsRemaining).toBe(6);
  });
});

describe('counter objects and interaction order', () => {
  it.each([
    ['ai', 'tomato'],
    ['ai', 'onion'],
    ['ai', 'dish'],
    ['ai', 'soup'],
    ['human', 'tomato'],
    ['human', 'onion'],
    ['human', 'dish'],
    ['human', 'soup'],
  ] as const)('lets %s drop and recover %s on an X counter', (actor, item) => {
    const initial = createGameState('running');
    const chefKey = actor === 'ai' ? 'partner' : 'player';
    const otherChefKey = actor === 'ai' ? 'player' : 'partner';
    const actionIndex = actor === 'ai'
      ? (action: 'interact') => [action, 'stay'] as const
      : (action: 'interact') => ['stay', action] as const;
    const counter = { x: 3, y: 2 };
    const state = {
      ...initial,
      [chefKey]: {
        ...initial[chefKey],
        x: 3,
        y: 1,
        facing: 'down' as const,
        held: item,
        heldSoup: item === 'soup' ? { tomatoes: 1, onions: 2 } : undefined,
      },
      [otherChefKey]: { ...initial[otherChefKey], x: 6, y: 1 },
    };

    const [dropAi, dropHuman] = actionIndex('interact');
    const dropped = stepGame(state, dropAi, dropHuman);
    expect(dropped[chefKey].held).toBeNull();
    expect(dropped.counterObjects[counterKey(counter)]?.item).toBe(item);

    const [pickAi, pickHuman] = actionIndex('interact');
    const recovered = stepGame(dropped, pickAi, pickHuman);
    expect(recovered[chefKey].held).toBe(item);
    expect(recovered.counterObjects[counterKey(counter)]).toBeUndefined();
    if (item === 'soup') {
      expect(recovered[chefKey].heldSoup).toEqual({ tomatoes: 1, onions: 2 });
    }
  });

  it('lets either chef drop and pick up objects on any X counter', () => {
    const initial = createGameState('running');
    const counter = { x: 3, y: 2 };
    expect(isCounter(counter)).toBe(true);
    const carrying = {
      ...initial,
      player: { ...initial.player, x: 3, y: 1, facing: 'down' as const, held: 'tomato' as const },
      partner: { ...initial.partner, x: 6, y: 1 },
    };

    const dropped = stepGame(carrying, 'stay', 'interact');
    expect(dropped.player.held).toBeNull();
    expect(dropped.counterObjects[counterKey(counter)]?.item).toBe('tomato');
    expect(dropped.lastStepEvents.some((item) => item.code === 'drop_counter')).toBe(true);

    const pickedUp = stepGame(dropped, 'stay', 'interact');
    expect(pickedUp.player.held).toBe('tomato');
    expect(pickedUp.counterObjects[counterKey(counter)]).toBeUndefined();
    expect(pickedUp.lastStepEvents.some((item) => item.code === 'pickup_counter')).toBe(true);
  });

  it('preserves soup contents when soup moves through a counter', () => {
    const initial = createGameState('running');
    const counter = { x: 3, y: 2 };
    const carryingSoup = {
      ...initial,
      player: {
        ...initial.player,
        x: 3,
        y: 1,
        facing: 'down' as const,
        held: 'soup' as const,
        heldSoup: { tomatoes: 1, onions: 2 },
      },
      partner: { ...initial.partner, x: 6, y: 1 },
    };
    const dropped = stepGame(carryingSoup, 'stay', 'interact');
    expect(dropped.counterObjects[counterKey(counter)]).toEqual({
      item: 'soup',
      soupContents: { tomatoes: 1, onions: 2 },
    });
    const pickedUp = stepGame(dropped, 'stay', 'interact');
    expect(pickedUp.player.held).toBe('soup');
    expect(pickedUp.player.heldSoup).toEqual({ tomatoes: 1, onions: 2 });
  });

  it('resolves AI interaction before human interaction', () => {
    const initial = createGameState('running');
    const counter = { x: 3, y: 2 };
    const state = stepGame(
      {
        ...initial,
        partner: {
          ...initial.partner,
          x: 3,
          y: 1,
          facing: 'down',
          held: 'onion',
        },
        player: {
          ...initial.player,
          x: 3,
          y: 3,
          facing: 'up',
          held: null,
        },
      },
      'interact',
      'interact',
    );

    expect(state.partner.held).toBeNull();
    expect(state.player.held).toBe('onion');
    expect(state.counterObjects[counterKey(counter)]).toBeUndefined();
    expect(state.lastStepEvents.slice(0, 2).map((item) => item.code)).toEqual([
      'drop_counter',
      'pickup_counter',
    ]);
  });
});

describe('old-dynamics cooking, scoring, and horizon', () => {
  it.each([
    [['tomato', 'tomato', 'tomato'], 3, 0],
    [['tomato', 'tomato', 'onion'], 2, 1],
    [['tomato', 'onion', 'tomato'], 2, 1],
    [['tomato', 'onion', 'onion'], 1, 2],
    [['onion', 'tomato', 'tomato'], 2, 1],
    [['onion', 'tomato', 'onion'], 1, 2],
    [['onion', 'onion', 'tomato'], 1, 2],
    [['onion', 'onion', 'onion'], 0, 3],
  ] as const)('accepts old-dynamics ingredient sequence %j', (ingredients, tomatoes, onions) => {
    let state = createGameState('running');
    for (const ingredient of ingredients) {
      state = stepGame(
        {
          ...state,
          player: {
            ...state.player,
            x: 4,
            y: 1,
            facing: 'up',
            held: ingredient,
          },
        },
        'stay',
        'interact',
      );
    }
    expect(state.pot.stage).toBe('cooking');
    expect(state.pot.tomatoes).toBe(tomatoes);
    expect(state.pot.onions).toBe(onions);
    expect(state.pot.secondsRemaining).toBe(COOK_TIME_STEPS - 1);
  });

  it('starts a full pot automatically and cooks for exactly 20 environment steps', () => {
    const initial = createGameState('running');
    let state = stepGame(
      {
        ...initial,
        player: {
          ...initial.player,
          x: 4,
          y: 1,
          facing: 'up',
          held: 'onion',
        },
        pot: {
          stage: 'filling',
          secondsRemaining: 0,
          tomatoes: 2,
          onions: 0,
        },
      },
      'stay',
      'interact',
    );
    expect(state.pot).toEqual({
      stage: 'cooking',
      secondsRemaining: 19,
      tomatoes: 2,
      onions: 1,
    });
    expect(state.lastStepEvents.some((item) => item.code === 'cooking_started')).toBe(true);

    for (let step = 1; step < COOK_TIME_STEPS; step += 1) {
      state = stepGame(state, 'stay', 'stay');
    }
    expect(state.pot.stage).toBe('ready');
    expect(state.pot.secondsRemaining).toBe(0);
  });

  it('awards 20 for the correct soup and never adds time', () => {
    const initial = createGameState('running');
    const state = stepGame(
      {
        ...initial,
        tick: 554,
        secondsLeft: 123,
        player: {
          ...initial.player,
          x: 8,
          y: 3,
          facing: 'right',
          held: 'soup',
          heldSoup: { tomatoes: 2, onions: 1 },
        },
      },
      'stay',
      'interact',
    );
    expect(state.score).toBe(20);
    expect(state.ordersCompleted).toBe(1);
    expect(state.secondsLeft).toBe(123);
    expect(state.lastStepEvents.some((item) => item.code === 'serve_correct_soup')).toBe(true);
  });

  it('gives no sparse reward for a soup that does not match the fixed order', () => {
    const initial = createGameState('running');
    const state = stepGame(
      {
        ...initial,
        player: {
          ...initial.player,
          x: 8,
          y: 3,
          facing: 'right',
          held: 'soup',
          heldSoup: { tomatoes: 1, onions: 2 },
        },
      },
      'stay',
      'interact',
    );
    expect(state.score).toBe(0);
    expect(state.ordersCompleted).toBe(0);
    expect(state.player.held).toBeNull();
    expect(state.lastStepEvents.some((item) => item.code === 'serve_wrong_soup')).toBe(true);
  });

  it('ends at exactly 800 decisions and tickGame represents one two-step second', () => {
    let state = createGameState('running');
    const afterOneSecond = tickGame(state);
    expect(afterOneSecond.tick).toBe(2);
    expect(afterOneSecond.secondsLeft).toBe(399);

    state = { ...state, tick: 799, secondsLeft: 1 };
    state = stepGame(state, 'stay', 'stay');
    expect(state.tick).toBe(800);
    expect(state.secondsLeft).toBe(0);
    expect(state.status).toBe('finished');
    expect(state.lastEventCode).toBe('round_finished');
  });
});

describe('counter-aware AI partner', () => {
  it('uses a needed ingredient placed on a counter by the human', () => {
    const initial = createGameState('running');
    const counter = { x: 3, y: 2 };
    const state = {
      ...initial,
      partner: { ...initial.partner, x: 3, y: 1, facing: 'down' as const },
      counterObjects: { [counterKey(counter)]: { item: 'tomato' as const } },
    };
    expect(chooseAiAction(state)).toBe('interact');
    const next = stepGame(state, chooseAiAction(state), 'stay');
    expect(next.partner.held).toBe('tomato');
    expect(next.counterObjects[counterKey(counter)]).toBeUndefined();
  });

  it('can put an unusable held object on an empty counter', () => {
    const initial = createGameState('running');
    const counter = { x: 3, y: 2 };
    const state = {
      ...initial,
      partner: {
        ...initial.partner,
        x: 3,
        y: 1,
        facing: 'down' as const,
        held: 'onion' as const,
      },
      pot: {
        stage: 'cooking' as const,
        secondsRemaining: 10,
        tomatoes: 2,
        onions: 1,
      },
    };
    expect(chooseAiAction(state)).toBe('interact');
    const next = stepGame(state, chooseAiAction(state), 'stay');
    expect(next.partner.held).toBeNull();
    expect(next.counterObjects[counterKey(counter)]?.item).toBe('onion');
  });

  it('can finish the fixed recipe without walking through counters', () => {
    const initial = createGameState('running');
    let state = {
      ...initial,
      player: { ...initial.player, x: 7, y: 4 },
    };
    const visited: Array<{ x: number; y: number }> = [];
    for (let second = 0; second < ROUND_SECONDS && state.ordersCompleted === 0; second += 1) {
      state = tickGame(state);
      visited.push({ x: state.partner.x, y: state.partner.y });
      expect([state.partner.x, state.partner.y]).not.toEqual([
        state.player.x,
        state.player.y,
      ]);
    }
    expect(visited.every(isWalkable)).toBe(true);
    expect(state.ordersCompleted).toBeGreaterThanOrEqual(1);
    expect(state.score).toBeGreaterThanOrEqual(CORRECT_SOUP_REWARD);
  });
});
