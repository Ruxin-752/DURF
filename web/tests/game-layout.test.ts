import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  PLAYER_STARTS,
  SOURCE_LAYOUT_ROWS,
  STATIONS,
  TERRAIN_ROWS,
  createGameState,
  isWalkable,
  tickGame,
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
  it('keeps the deployed fixture byte-for-byte equivalent to the repository layout', () => {
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

  it('locks player starts and every pot/ingredient/dish/serving coordinate', () => {
    expect(PLAYER_STARTS).toEqual({
      player: { x: 3, y: 1 },
      partner: { x: 6, y: 1 },
    });
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
});

describe('AI partner on the locked ring', () => {
  it('can complete the exact two-tomato-one-onion recipe without crossing counters', () => {
    let state = createGameState('running');
    const visited: Array<{ x: number; y: number }> = [];
    for (let second = 0; second < 89 && state.ordersCompleted === 0; second += 1) {
      state = tickGame(state);
      visited.push({ x: state.partner.x, y: state.partner.y });
    }
    expect(visited.every(isWalkable)).toBe(true);
    expect(state.ordersCompleted).toBeGreaterThanOrEqual(1);
    expect(state.score).toBeGreaterThanOrEqual(100);
  });
});
