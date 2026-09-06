import type { GameState, HeldItem, Ingredient } from './game';

export const SUBGOALS = [
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
] as const;

export type Subgoal = (typeof SUBGOALS)[number];

export const REWARD_FEATURES = [
  'ingredient_tomato',
  'ingredient_onion',
  'pick_tomato',
  'pick_onion',
  'pick_dish',
  'pick_ready_soup',
  'recipe_needs_first_tomato',
  'recipe_needs_second_tomato',
  'recipe_needs_onion',
  'adds_needed_tomato',
  'adds_needed_onion',
  'adds_extra_tomato',
  'adds_extra_onion',
  'wrong_ingredient',
  'breaks_recipe',
  'completes_recipe',
  'matches_current_order',
  'pot_empty',
  'pot_has_one_tomato',
  'pot_has_two_tomatoes',
  'pot_has_two_tomatoes_one_onion',
  'pot_cooking',
  'soup_ready',
  'dish_needed_for_ready_soup',
  'serve_ready_soup',
  'supports_serving',
  'delays_serving',
  'near_tomato_dispenser',
  'near_onion_dispenser',
  'near_dish_dispenser',
  'near_pot',
  'near_serving_counter',
  'moves_toward_needed_object',
  'moves_away_from_needed_object',
  'blocks_partner_on_ring',
  'distance_cost',
  'time_cost',
  'respects_human_intent',
  'avoids_duplicate_human_task',
  'clears_human_shortest_path',
  'clears_human_path',
  'clears_serving_access',
  'blocks_human_path',
  'blocks_serving_route',
  'human_wait_cost',
  'duplicate_human_task',
  'steals_human_target',
  'crowds_human_target',
  'frustrates_human',
  'collision_risk',
  'complementary_to_human',
  'avoids_human_shortest_path',
  'cuts_in_front_of_human',
] as const;

export type RewardFeature = (typeof REWARD_FEATURES)[number];
export type RewardWeights = Readonly<Partial<Record<RewardFeature, number>>>;
export type FeatureVector = Partial<Record<RewardFeature, number>>;
export type PathEffect = 'blocks' | 'enters' | 'clears';

/**
 * Versioned mirror of Python's live-policy feature contract.
 *
 * Only policy-active dimensions may change candidate ordering. Context-only
 * dimensions are observable state facts shared by every candidate, while
 * unsupported dimensions do not have a live Web policy implementation yet.
 */
export const POLICY_FEATURE_CONTRACT_VERSION = 'durf-53d-policy-contract-v1';

export const POLICY_ACTIVE_FEATURES = [
  'adds_needed_onion',
  'adds_needed_tomato',
  'avoids_human_shortest_path',
  'avoids_duplicate_human_task',
  'blocks_human_path',
  'blocks_serving_route',
  'clears_human_path',
  'clears_human_shortest_path',
  'clears_serving_access',
  'complementary_to_human',
  'completes_recipe',
  'crowds_human_target',
  'cuts_in_front_of_human',
  'delays_serving',
  'dish_needed_for_ready_soup',
  'distance_cost',
  'duplicate_human_task',
  'frustrates_human',
  'human_wait_cost',
  'ingredient_onion',
  'ingredient_tomato',
  'matches_current_order',
  'moves_toward_needed_object',
  'pick_dish',
  'pick_onion',
  'pick_ready_soup',
  'pick_tomato',
  'recipe_needs_onion',
  'respects_human_intent',
  'serve_ready_soup',
  'steals_human_target',
  'supports_serving',
  'time_cost',
] as const satisfies readonly RewardFeature[];

export const POLICY_CONTEXT_ONLY_FEATURES = [
  'pot_cooking',
  'pot_empty',
  'pot_has_one_tomato',
  'pot_has_two_tomatoes',
  'pot_has_two_tomatoes_one_onion',
  'soup_ready',
] as const satisfies readonly RewardFeature[];

export const POLICY_UNSUPPORTED_FEATURES = [
  'recipe_needs_first_tomato',
  'recipe_needs_second_tomato',
  'adds_extra_tomato',
  'adds_extra_onion',
  'wrong_ingredient',
  'breaks_recipe',
  'near_tomato_dispenser',
  'near_onion_dispenser',
  'near_dish_dispenser',
  'near_pot',
  'near_serving_counter',
  'moves_away_from_needed_object',
  'blocks_partner_on_ring',
  'collision_risk',
] as const satisfies readonly RewardFeature[];

export const POLICY_FEATURE_CONTRACT = Object.freeze({
  schemaVersion: POLICY_FEATURE_CONTRACT_VERSION,
  featureCount: REWARD_FEATURES.length,
  policyActive: POLICY_ACTIVE_FEATURES,
  contextOnly: POLICY_CONTEXT_ONLY_FEATURES,
  unsupported: POLICY_UNSUPPORTED_FEATURES,
});

type Resource = Ingredient | 'dish' | 'soup' | 'serving';
type StagedResource = Exclude<HeldItem, null>;

export interface PotSnapshot {
  ingredients: readonly Ingredient[];
  status: string;
}

export interface SubgoalContext {
  recipe: readonly Ingredient[];
  potIngredients: readonly Ingredient[];
  potStatus: string;
  agentHolding: HeldItem;
  humanHolding: HeldItem;
  humanIntent: string | null;
  humanCommittedUnits: number | null;
  candidatePathEffects: Readonly<Partial<Record<Subgoal, PathEffect>>>;
  candidateTargetOverlapsHuman: Readonly<Partial<Record<Subgoal, boolean>>>;
  potSnapshots: readonly PotSnapshot[] | null;
  stagedInventory: Readonly<Partial<Record<StagedResource, number>>>;
}

export interface FeatureContribution {
  feature: RewardFeature;
  value: number;
  weight: number;
  contribution: number;
}

export interface RankedSubgoal {
  subgoal: Subgoal;
  score: number;
  features: FeatureVector;
}

export type DecisionSource =
  | 'reward_argmax'
  | 'h0_tie_fallback'
  | 'unresolved_reward_tie'
  | 'decision_null_weights_h0_fallback'
  | 'invalid_weights_h0_fallback';

export interface RewardSubgoalDecision {
  chosenSubgoal: Subgoal;
  feasibleSubgoals: readonly Subgoal[];
  ranking: readonly RankedSubgoal[];
  score: number;
  decisionSource: DecisionSource;
  rewardMargin: number | null;
  topContributions: readonly FeatureContribution[];
  usedLearnedWeights: boolean;
  h0Fallback: Subgoal;
  invalidWeightFeatures: readonly string[];
  policyFeatureContractVersion: string;
  activeWeightFeatures: readonly RewardFeature[];
  ignoredDecisionNullWeightFeatures: readonly RewardFeature[];
}

export interface RewardSubgoalOptions {
  contextOverrides?: Partial<SubgoalContext>;
  /** A live-feasibility subset; entries outside task-valid enumeration are ignored. */
  feasibleSubgoals?: readonly Subgoal[];
  h0Fallback?: Subgoal;
  tieTolerance?: number;
  topContributionLimit?: number;
}

const DEFAULT_RECIPE: readonly Ingredient[] = ['tomato', 'tomato', 'onion'];
const REWARD_FEATURE_SET = new Set<string>(REWARD_FEATURES);
const POLICY_ACTIVE_FEATURE_SET = new Set<RewardFeature>(POLICY_ACTIVE_FEATURES);
const POLICY_DECISION_NULL_FEATURE_SET = new Set<RewardFeature>([
  ...POLICY_CONTEXT_ONLY_FEATURES,
  ...POLICY_UNSUPPORTED_FEATURES,
]);

const SUBGOAL_RESOURCE: Readonly<Record<Subgoal, Resource | null>> = {
  GET_TOMATO: 'tomato',
  PUT_TOMATO_IN_POT: 'tomato',
  GET_ONION: 'onion',
  PUT_ONION_IN_POT: 'onion',
  GET_DISH: 'dish',
  PICKUP_SOUP: 'soup',
  SERVE_SOUP: 'serving',
  STASH_HELD_OBJECT: null,
  YIELD_PATH: null,
  WAIT: null,
};

function itemCounts(items: readonly string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const item of items) counts.set(item, (counts.get(item) ?? 0) + 1);
  return counts;
}

function countItem(items: readonly string[], item: string): number {
  return items.reduce((count, value) => count + Number(value === item), 0);
}

function distinct<T>(items: readonly T[]): T[] {
  return [...new Set(items)];
}

function nonNegativeInteger(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.max(0, Math.trunc(value))
    : 0;
}

function missingIngredients(
  recipe: readonly Ingredient[],
  current: readonly Ingredient[],
): Ingredient[] {
  const remaining = itemCounts(recipe);
  for (const ingredient of current) {
    remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - 1);
  }
  const missing: Ingredient[] = [];
  for (const ingredient of recipe) {
    if ((remaining.get(ingredient) ?? 0) > 0) {
      missing.push(ingredient);
      remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - 1);
    }
  }
  return missing;
}

function statePotIngredients(state: GameState): Ingredient[] {
  return [
    ...Array.from({ length: state.pot.tomatoes }, () => 'tomato' as const),
    ...Array.from({ length: state.pot.onions }, () => 'onion' as const),
  ];
}

function stateStagedInventory(
  state: GameState,
): Partial<Record<StagedResource, number>> {
  const inventory: Partial<Record<StagedResource, number>> = {};
  for (const object of Object.values(state.counterObjects)) {
    inventory[object.item] = (inventory[object.item] ?? 0) + 1;
  }
  return inventory;
}

function optionalHolding(
  override: HeldItem | undefined,
  fallback: HeldItem,
): HeldItem {
  return override === undefined ? fallback : override;
}

export function buildSubgoalContext(
  state: GameState,
  overrides: Partial<SubgoalContext> = {},
): SubgoalContext {
  const potIngredients = statePotIngredients(state);
  return {
    recipe: [...(overrides.recipe ?? DEFAULT_RECIPE)],
    potIngredients: [...(overrides.potIngredients ?? potIngredients)],
    potStatus: overrides.potStatus ?? state.pot.stage,
    agentHolding: optionalHolding(overrides.agentHolding, state.partner.held),
    humanHolding: optionalHolding(overrides.humanHolding, state.player.held),
    humanIntent: overrides.humanIntent ?? null,
    humanCommittedUnits: overrides.humanCommittedUnits ?? null,
    candidatePathEffects: { ...overrides.candidatePathEffects },
    candidateTargetOverlapsHuman: {
      ...overrides.candidateTargetOverlapsHuman,
    },
    potSnapshots: overrides.potSnapshots
      ? overrides.potSnapshots.map((snapshot) => ({
          ingredients: [...snapshot.ingredients],
          status: snapshot.status,
        }))
      : null,
    stagedInventory: {
      ...(overrides.stagedInventory ?? stateStagedInventory(state)),
    },
  };
}

function normalizedPotSnapshots(context: SubgoalContext): PotSnapshot[] {
  if (context.potSnapshots && context.potSnapshots.length > 0) {
    return context.potSnapshots.map((snapshot) => ({
      ingredients: [...snapshot.ingredients],
      status: String(snapshot.status || 'empty').toLowerCase(),
    }));
  }
  return [
    {
      ingredients: [...context.potIngredients],
      status: String(context.potStatus || 'empty').toLowerCase(),
    },
  ];
}

function readySoupCount(context: SubgoalContext): number {
  return normalizedPotSnapshots(context).filter(({ status }) => status === 'ready').length;
}

function cookingSoupCount(context: SubgoalContext): number {
  return normalizedPotSnapshots(context).filter(({ status }) => status === 'cooking').length;
}

function soupReady(context: SubgoalContext): boolean {
  return readySoupCount(context) > 0;
}

function soupCooking(context: SubgoalContext): boolean {
  return cookingSoupCount(context) > 0;
}

/** Python-parity exemption: waiting is productive only while a complete soup cooks. */
export function passiveCookingWaitIsValid(context: SubgoalContext): boolean {
  if (
    !soupCooking(context) ||
    soupReady(context) ||
    openMissingIngredients(context).length > 0 ||
    ![null, 'dish'].includes(context.agentHolding)
  ) {
    return false;
  }
  const recipe = itemCounts(context.recipe);
  return normalizedPotSnapshots(context).some(({ ingredients, status }) => {
    if (status !== 'cooking') return false;
    const contents = itemCounts(ingredients);
    return [...recipe].every(
      ([ingredient, count]) => (contents.get(ingredient) ?? 0) >= count,
    );
  });
}

function openMissingIngredients(context: SubgoalContext): Ingredient[] {
  const missing: Ingredient[] = [];
  for (const snapshot of normalizedPotSnapshots(context)) {
    if (snapshot.status === 'ready' || snapshot.status === 'cooking') continue;
    missing.push(...missingIngredients(context.recipe, snapshot.ingredients));
  }
  return missing;
}

function stagedUnits(context: SubgoalContext, resource: StagedResource): number {
  return nonNegativeInteger(context.stagedInventory[resource]);
}

function humanCoversHeldIngredient(
  context: SubgoalContext,
  held: Ingredient,
  missing: readonly Ingredient[],
): boolean {
  const remaining = countItem(missing, held);
  if (remaining <= 0 || context.humanHolding !== held) return false;
  const committed = context.humanCommittedUnits ?? 1;
  return nonNegativeInteger(committed) >= remaining;
}

function prefetchIngredientDeficits(context: SubgoalContext): Ingredient[] {
  const remaining = itemCounts(context.recipe);
  for (const ingredient of distinct(context.recipe)) {
    const committed =
      stagedUnits(context, ingredient) +
      Number(context.agentHolding === ingredient) +
      Number(context.humanHolding === ingredient);
    remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - committed);
  }
  const deficits: Ingredient[] = [];
  for (const ingredient of context.recipe) {
    if ((remaining.get(ingredient) ?? 0) > 0) {
      deficits.push(ingredient);
      remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - 1);
    }
  }
  return deficits;
}

function openWorkNotCoveredByHuman(
  context: SubgoalContext,
  missing: readonly Ingredient[],
): Ingredient[] {
  const remaining = itemCounts(missing);
  const held = context.humanHolding;
  if (held === 'tomato' || held === 'onion') {
    const committed = nonNegativeInteger(context.humanCommittedUnits ?? 1);
    remaining.set(held, Math.max(0, (remaining.get(held) ?? 0) - committed));
  }
  const uncovered: Ingredient[] = [];
  for (const ingredient of missing) {
    if ((remaining.get(ingredient) ?? 0) > 0) {
      uncovered.push(ingredient);
      remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - 1);
    }
  }
  return uncovered;
}

function dishSupply(context: SubgoalContext): number {
  return (
    stagedUnits(context, 'dish') +
    Number(context.agentHolding === 'dish') +
    Number(context.humanHolding === 'dish')
  );
}

function activeSoupsNeedAnotherDish(context: SubgoalContext): boolean {
  return readySoupCount(context) + cookingSoupCount(context) > dishSupply(context);
}

/**
 * Mirrors Python's task-valid enumeration. YIELD_PATH follows the live Python
 * wrapper: it is inserted only when geometry marks a legal clearing move.
 */
export function enumerateFeasibleSubgoals(context: SubgoalContext): Subgoal[] {
  const missing = openMissingIngredients(context);
  const held = context.agentHolding;
  const candidates: Subgoal[] = [];

  if (held === 'tomato' || held === 'onion') {
    if (missing.includes(held)) {
      candidates.push(held === 'tomato' ? 'PUT_TOMATO_IN_POT' : 'PUT_ONION_IN_POT');
      if (humanCoversHeldIngredient(context, held, missing)) {
        candidates.push('STASH_HELD_OBJECT');
      }
    } else {
      candidates.push('STASH_HELD_OBJECT');
    }
  } else if (held === 'dish') {
    candidates.push(soupReady(context) ? 'PICKUP_SOUP' : 'STASH_HELD_OBJECT');
  } else if (held === 'soup') {
    candidates.push('SERVE_SOUP');
  } else {
    const uncoveredOpenWork = openWorkNotCoveredByHuman(context, missing);
    const imminentSoupCount = Number(missing.length > 0 && uncoveredOpenWork.length === 0);
    const activeSoupCount = readySoupCount(context) + cookingSoupCount(context);
    const dishWorkUncovered =
      (soupReady(context) || imminentSoupCount > 0) &&
      activeSoupCount + imminentSoupCount > dishSupply(context);
    if (dishWorkUncovered) candidates.push('GET_DISH');
    for (const ingredient of distinct(uncoveredOpenWork)) {
      candidates.push(ingredient === 'tomato' ? 'GET_TOMATO' : 'GET_ONION');
    }

    if (soupCooking(context) && !soupReady(context) && missing.length === 0) {
      if (activeSoupsNeedAnotherDish(context)) candidates.push('GET_DISH');
      for (const ingredient of distinct(prefetchIngredientDeficits(context))) {
        candidates.push(ingredient === 'tomato' ? 'GET_TOMATO' : 'GET_ONION');
      }
    }
  }

  const feasible = distinct(candidates);
  if (context.candidatePathEffects.YIELD_PATH === 'clears') {
    feasible.push('YIELD_PATH');
  }
  feasible.push('WAIT');
  return distinct(feasible);
}

function h0PreferredSubgoal(context: SubgoalContext): Subgoal {
  const held = context.agentHolding;
  const missing = openMissingIngredients(context);
  if (held === 'tomato' || held === 'onion') {
    if (!missing.includes(held)) return 'STASH_HELD_OBJECT';
    return held === 'tomato' ? 'PUT_TOMATO_IN_POT' : 'PUT_ONION_IN_POT';
  }
  if (held === 'dish') return soupReady(context) ? 'PICKUP_SOUP' : 'STASH_HELD_OBJECT';
  if (held === 'soup') return 'SERVE_SOUP';
  if (soupReady(context)) return 'GET_DISH';
  if (missing[0] === 'tomato') return 'GET_TOMATO';
  if (missing[0] === 'onion') return 'GET_ONION';
  return 'WAIT';
}

export function chooseH0Subgoal(
  context: SubgoalContext,
  feasibleSubgoals: readonly Subgoal[] = enumerateFeasibleSubgoals(context),
): Subgoal {
  const preferred = h0PreferredSubgoal(context);
  if (feasibleSubgoals.includes(preferred)) return preferred;
  return feasibleSubgoals[0] ?? 'WAIT';
}

function setFeature(features: FeatureVector, feature: RewardFeature): void {
  features[feature] = 1;
}

function addStateFeatures(context: SubgoalContext, features: FeatureVector): void {
  for (const snapshot of normalizedPotSnapshots(context)) {
    if (snapshot.ingredients.length === 0) setFeature(features, 'pot_empty');
    const tomatoes = countItem(snapshot.ingredients, 'tomato');
    const onions = countItem(snapshot.ingredients, 'onion');
    if (tomatoes >= 1) setFeature(features, 'pot_has_one_tomato');
    if (tomatoes >= 2) setFeature(features, 'pot_has_two_tomatoes');
    if (tomatoes >= 2 && onions >= 1) {
      setFeature(features, 'pot_has_two_tomatoes_one_onion');
    }
  }
  if (soupCooking(context)) setFeature(features, 'pot_cooking');
  if (soupReady(context)) setFeature(features, 'soup_ready');
}

function addTaskFeatures(
  context: SubgoalContext,
  subgoal: Subgoal,
  features: FeatureVector,
): void {
  const missing = openMissingIngredients(context);
  if (subgoal === 'GET_TOMATO' || subgoal === 'PUT_TOMATO_IN_POT') {
    setFeature(features, 'ingredient_tomato');
    if (subgoal === 'GET_TOMATO') setFeature(features, 'pick_tomato');
    if (missing.includes('tomato')) {
      setFeature(features, 'adds_needed_tomato');
      setFeature(features, 'moves_toward_needed_object');
      setFeature(features, 'matches_current_order');
    } else if (subgoal === 'GET_TOMATO' && soupCooking(context)) {
      setFeature(features, 'moves_toward_needed_object');
    } else {
      setFeature(features, 'adds_extra_tomato');
      setFeature(features, 'wrong_ingredient');
      setFeature(features, 'breaks_recipe');
    }
  } else if (subgoal === 'GET_ONION' || subgoal === 'PUT_ONION_IN_POT') {
    setFeature(features, 'ingredient_onion');
    if (subgoal === 'GET_ONION') setFeature(features, 'pick_onion');
    if (missing.includes('onion')) {
      setFeature(features, 'adds_needed_onion');
      setFeature(features, 'recipe_needs_onion');
      setFeature(features, 'moves_toward_needed_object');
      setFeature(features, 'matches_current_order');
    } else if (subgoal === 'GET_ONION' && soupCooking(context)) {
      setFeature(features, 'moves_toward_needed_object');
    } else {
      setFeature(features, 'adds_extra_onion');
      setFeature(features, 'wrong_ingredient');
      setFeature(features, 'breaks_recipe');
    }
  } else if (subgoal === 'GET_DISH') {
    setFeature(features, 'pick_dish');
    if (soupReady(context)) {
      setFeature(features, 'dish_needed_for_ready_soup');
      setFeature(features, 'supports_serving');
    } else if (!soupCooking(context)) {
      setFeature(features, 'time_cost');
      setFeature(features, 'delays_serving');
    }
  } else if (subgoal === 'PICKUP_SOUP') {
    setFeature(features, 'pick_ready_soup');
    if (soupReady(context)) setFeature(features, 'supports_serving');
    else setFeature(features, 'time_cost');
  } else if (subgoal === 'SERVE_SOUP') {
    if (context.agentHolding === 'soup' || soupReady(context)) {
      setFeature(features, 'serve_ready_soup');
      setFeature(features, 'supports_serving');
      setFeature(features, 'completes_recipe');
    } else {
      setFeature(features, 'time_cost');
    }
  } else if (subgoal === 'STASH_HELD_OBJECT') {
    setFeature(features, soupReady(context) ? 'supports_serving' : 'time_cost');
  } else if (subgoal === 'YIELD_PATH') {
    setFeature(features, 'time_cost');
    setFeature(features, 'distance_cost');
  } else if (subgoal === 'WAIT') {
    const passiveCookingOnly =
      soupCooking(context) &&
      !soupReady(context) &&
      openMissingIngredients(context).length === 0;
    if (!passiveCookingOnly || context.agentHolding !== null) {
      setFeature(features, 'time_cost');
    }
    if (soupReady(context) || missing.length > 0) setFeature(features, 'delays_serving');
    if (context.agentHolding !== null) setFeature(features, 'delays_serving');
  }
}

function resourceFromText(text: string | null): Resource | null {
  if (!text) return null;
  const lowered = text.toLowerCase();
  const mappings: ReadonlyArray<readonly [readonly string[], Resource]> = [
    [['dish', 'plate'], 'dish'],
    [['serve', 'serving', 'deliver'], 'serving'],
    [['soup'], 'soup'],
    [['tomato'], 'tomato'],
    [['onion'], 'onion'],
  ];
  for (const [keywords, resource] of mappings) {
    if (keywords.some((keyword) => lowered.includes(keyword))) return resource;
  }
  return null;
}

function humanTargetResource(context: SubgoalContext): Resource | null {
  return resourceFromText(context.humanIntent) ?? resourceFromText(context.humanHolding);
}

function addCoordinationFeatures(
  context: SubgoalContext,
  subgoal: Subgoal,
  features: FeatureVector,
): void {
  const humanTarget = humanTargetResource(context);
  const pathEffect = context.candidatePathEffects[subgoal];
  const blocksPath = pathEffect === 'blocks' || pathEffect === 'enters';
  const clearsPath = pathEffect === 'clears';
  if (humanTarget === null) {
    if (subgoal === 'YIELD_PATH' && clearsPath) {
      setFeature(features, 'respects_human_intent');
      setFeature(features, 'clears_human_path');
      setFeature(features, 'clears_human_shortest_path');
      setFeature(features, 'avoids_human_shortest_path');
    }
    return;
  }

  const subgoalResource = SUBGOAL_RESOURCE[subgoal];
  const servingChain = new Set<Resource>(['dish', 'soup', 'serving']);
  const missing = openMissingIngredients(context);
  const remainingUnits = (resource: Resource): number => {
    if (resource === 'tomato' || resource === 'onion') {
      const units = countItem(missing, resource);
      return units === 0 && soupCooking(context)
        ? countItem(context.recipe, resource)
        : units;
    }
    if (resource === 'dish' || resource === 'soup') return readySoupCount(context);
    return (
      readySoupCount(context) +
      Number(context.agentHolding === 'soup') +
      Number(context.humanHolding === 'soup')
    );
  };

  const committedUnits = nonNegativeInteger(context.humanCommittedUnits ?? 1);
  const sameResource = subgoalResource !== null && subgoalResource === humanTarget;
  const targetOverlap =
    context.candidateTargetOverlapsHuman[subgoal] ?? sameResource;
  let sameWork =
    sameResource ||
    (targetOverlap &&
      servingChain.has(humanTarget) &&
      subgoalResource !== null &&
      servingChain.has(subgoalResource));
  if (
    subgoal === 'SERVE_SOUP' &&
    context.agentHolding === 'soup' &&
    context.humanHolding === 'soup'
  ) {
    sameWork = false;
  }
  const humanCoversWork =
    sameWork &&
    remainingUnits(humanTarget) > 0 &&
    committedUnits >= remainingUnits(humanTarget);
  const releasesCoveredDuplicate =
    subgoal === 'STASH_HELD_OBJECT' &&
    (context.agentHolding === 'tomato' || context.agentHolding === 'onion') &&
    context.agentHolding === context.humanHolding &&
    context.agentHolding === humanTarget &&
    remainingUnits(humanTarget) > 0 &&
    committedUnits >= remainingUnits(humanTarget);

  if (releasesCoveredDuplicate) {
    setFeature(features, 'respects_human_intent');
    setFeature(features, 'avoids_duplicate_human_task');
  } else if (humanCoversWork) {
    setFeature(features, 'duplicate_human_task');
    if (targetOverlap) setFeature(features, 'crowds_human_target');
    if (
      targetOverlap &&
      context.humanHolding === null &&
      ['GET_DISH', 'GET_TOMATO', 'GET_ONION', 'PICKUP_SOUP'].includes(subgoal)
    ) {
      setFeature(features, 'steals_human_target');
    }
  } else if (subgoalResource !== null) {
    setFeature(features, 'complementary_to_human');
    setFeature(features, 'respects_human_intent');
    setFeature(features, 'avoids_duplicate_human_task');
  } else if (subgoal === 'WAIT') {
    const targetUnits = remainingUnits(humanTarget);
    if (targetUnits > 0 && committedUnits >= targetUnits && !blocksPath) {
      setFeature(features, 'respects_human_intent');
      setFeature(features, 'avoids_duplicate_human_task');
    }
  }

  if (blocksPath) {
    setFeature(features, 'blocks_human_path');
    setFeature(features, 'human_wait_cost');
    setFeature(features, 'frustrates_human');
    if (pathEffect === 'enters') setFeature(features, 'cuts_in_front_of_human');
    if (servingChain.has(humanTarget)) setFeature(features, 'blocks_serving_route');
  } else if (clearsPath) {
    setFeature(features, 'clears_human_path');
    setFeature(features, 'clears_human_shortest_path');
    setFeature(features, 'avoids_human_shortest_path');
    if (subgoal === 'YIELD_PATH') setFeature(features, 'respects_human_intent');
    if (servingChain.has(humanTarget)) setFeature(features, 'clears_serving_access');
  }
}

export function featurizeSubgoal(
  context: SubgoalContext,
  subgoal: Subgoal,
): FeatureVector {
  const features: FeatureVector = {};
  addStateFeatures(context, features);
  addTaskFeatures(context, subgoal, features);
  addCoordinationFeatures(context, subgoal, features);
  return features;
}

interface NormalizedWeights {
  values: Record<RewardFeature, number>;
  invalidFeatures: string[];
  activeFeatures: RewardFeature[];
  ignoredDecisionNullFeatures: RewardFeature[];
}

function normalizeWeights(weights: RewardWeights): NormalizedWeights {
  const raw = weights as Readonly<Record<string, unknown>>;
  const invalidFeatures = Object.entries(raw)
    .filter(
      ([feature, value]) =>
        !REWARD_FEATURE_SET.has(feature) ||
        typeof value !== 'number' ||
        !Number.isFinite(value),
    )
    .map(([feature]) => feature)
    .sort();
  const safe = invalidFeatures.length === 0;
  const values = Object.fromEntries(
    REWARD_FEATURES.map((feature) => [
      feature,
      safe &&
      POLICY_ACTIVE_FEATURE_SET.has(feature) &&
      typeof raw[feature] === 'number'
        ? raw[feature]
        : 0,
    ]),
  ) as Record<RewardFeature, number>;
  const activeFeatures = safe
    ? POLICY_ACTIVE_FEATURES.filter(
        (feature) => typeof raw[feature] === 'number' && raw[feature] !== 0,
      )
    : [];
  const ignoredDecisionNullFeatures = safe
    ? REWARD_FEATURES.filter(
        (feature) =>
          POLICY_DECISION_NULL_FEATURE_SET.has(feature) &&
          typeof raw[feature] === 'number' &&
          raw[feature] !== 0,
      )
    : [];
  return {
    values,
    invalidFeatures,
    activeFeatures,
    ignoredDecisionNullFeatures,
  };
}

function scoreFeatureVector(
  weights: Readonly<Record<RewardFeature, number>>,
  features: FeatureVector,
): number {
  return POLICY_ACTIVE_FEATURES.reduce(
    (score, feature) => score + weights[feature] * (features[feature] ?? 0),
    0,
  );
}

function rankWithWeights(
  context: SubgoalContext,
  feasible: readonly Subgoal[],
  weights: Readonly<Record<RewardFeature, number>>,
): RankedSubgoal[] {
  return feasible
    .map((subgoal, order) => {
      const features = featurizeSubgoal(context, subgoal);
      return { subgoal, score: scoreFeatureVector(weights, features), features, order };
    })
    .sort((left, right) => right.score - left.score || left.order - right.order)
    .map(({ order: _order, ...ranked }) => ranked);
}

function selectedContributions(
  features: FeatureVector,
  weights: Readonly<Record<RewardFeature, number>>,
  limit: number,
): FeatureContribution[] {
  return POLICY_ACTIVE_FEATURES.flatMap((feature) => {
    const value = features[feature] ?? 0;
    const weight = weights[feature];
    const contribution = value * weight;
    return contribution === 0 ? [] : [{ feature, value, weight, contribution }];
  })
    .sort(
      (left, right) =>
        Math.abs(right.contribution) - Math.abs(left.contribution) ||
        left.feature.localeCompare(right.feature),
    )
    .slice(0, limit);
}

export function chooseRewardSubgoal(
  state: GameState,
  weights: RewardWeights,
  options: RewardSubgoalOptions = {},
): RewardSubgoalDecision {
  const context = buildSubgoalContext(state, options.contextOverrides);
  const taskFeasible = enumerateFeasibleSubgoals(context);
  const requestedFeasible = options.feasibleSubgoals
    ? new Set(options.feasibleSubgoals)
    : null;
  const feasible = requestedFeasible
    ? taskFeasible.filter((subgoal) => requestedFeasible.has(subgoal))
    : taskFeasible;
  if (feasible.length === 0) feasible.push('WAIT');
  const normalized = normalizeWeights(weights);
  let ranking = rankWithWeights(context, feasible, normalized.values);
  const scoreOverflow = ranking.some(({ score }) => !Number.isFinite(score));
  if (scoreOverflow) {
    normalized.invalidFeatures.push('<score-overflow>');
    for (const feature of REWARD_FEATURES) normalized.values[feature] = 0;
    normalized.activeFeatures = [];
    ranking = rankWithWeights(context, feasible, normalized.values);
  }

  const requestedFallback = options.h0Fallback;
  const h0Fallback =
    requestedFallback && feasible.includes(requestedFallback)
      ? requestedFallback
      : chooseH0Subgoal(context, feasible);
  const tolerance =
    typeof options.tieTolerance === 'number' &&
    Number.isFinite(options.tieTolerance) &&
    options.tieTolerance >= 0
      ? options.tieTolerance
      : 1e-9;
  const topScore = ranking[0]?.score ?? 0;
  const tied = ranking.filter(({ score }) => Math.abs(score - topScore) <= tolerance);
  const tieFallback = tied.find(({ subgoal }) => subgoal === h0Fallback);
  const selected = tieFallback ?? ranking[0];
  const invalidWeights = normalized.invalidFeatures.length > 0;
  let decisionSource: DecisionSource;
  if (invalidWeights) decisionSource = 'invalid_weights_h0_fallback';
  else if (
    normalized.activeFeatures.length === 0 &&
    normalized.ignoredDecisionNullFeatures.length > 0
  ) {
    decisionSource = 'decision_null_weights_h0_fallback';
  }
  else if (tied.length === 1) decisionSource = 'reward_argmax';
  else if (tieFallback) decisionSource = 'h0_tie_fallback';
  else decisionSource = 'unresolved_reward_tie';
  const contributionLimit =
    typeof options.topContributionLimit === 'number' &&
    Number.isFinite(options.topContributionLimit)
      ? Math.max(0, Math.trunc(options.topContributionLimit))
      : 5;

  return {
    chosenSubgoal: selected.subgoal,
    feasibleSubgoals: feasible,
    ranking,
    score: selected.score,
    decisionSource,
    rewardMargin:
      ranking.length > 1 ? ranking[0].score - ranking[1].score : null,
    topContributions: selectedContributions(
      selected.features,
      normalized.values,
      contributionLimit,
    ),
    usedLearnedWeights:
      !invalidWeights && normalized.activeFeatures.length > 0 && tied.length === 1,
    h0Fallback,
    invalidWeightFeatures: normalized.invalidFeatures,
    policyFeatureContractVersion: POLICY_FEATURE_CONTRACT_VERSION,
    activeWeightFeatures: normalized.activeFeatures,
    ignoredDecisionNullWeightFeatures: normalized.ignoredDecisionNullFeatures,
  };
}
