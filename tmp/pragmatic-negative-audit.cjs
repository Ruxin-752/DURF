// tmp/pragmatic-negative-audit.ts
var import_node_fs = require("node:fs");
var import_node_crypto = require("node:crypto");

// web/lib/browser-models.ts
function createFullGaussianPrior(features, variance = 25) {
  if (!(variance > 0)) throw new Error("prior variance must be positive");
  return {
    features: [...features],
    mean: features.map(() => 0),
    covariance: features.map(
      (_, row) => features.map((__, column) => row === column ? variance : 0)
    )
  };
}
var FORM_DEFAULT = {
  evaluative: "trajectory",
  imperative: "action_spatial",
  descriptive: "feature"
};
var FORM_SUBTYPES = {
  evaluative: /* @__PURE__ */ new Set(["trajectory", "action_behavioral"]),
  imperative: /* @__PURE__ */ new Set(["action_spatial"]),
  descriptive: /* @__PURE__ */ new Set(["feature"])
};
function normalizedReference(features, values) {
  const selected = {};
  const vector = features.map((feature) => {
    const value = Number(values[feature] ?? 0);
    if (!Number.isFinite(value)) throw new Error(`Route1 feature ${feature} is not finite`);
    if (value !== 0) selected[feature] = value;
    return value;
  });
  const total = vector.reduce((sum, value) => sum + Math.abs(value), 0);
  return {
    vector: total ? vector.map((value) => value / total) : vector,
    selected
  };
}
function applyRoute1PaperUpdate(state, observation) {
  const threshold = observation.feedbackFormThreshold ?? 0.55;
  const requested = observation.requestedReferenceSubtype;
  const defaultSubtype = FORM_DEFAULT[observation.feedbackForm];
  const conflict = Boolean(requested && requested !== "other" && !FORM_SUBTYPES[observation.feedbackForm].has(requested));
  const effectiveSubtype = requested === "other" ? "other" : requested && FORM_SUBTYPES[observation.feedbackForm].has(requested) ? requested : defaultSubtype;
  const source = observation.feedbackForm === "evaluative" ? observation.trajectoryFeatures ?? {} : observation.feedbackForm === "imperative" ? observation.actionFeatures ?? {} : observation.namedFeatures ?? {};
  const { vector, selected } = normalizedReference(state.features, source);
  const confidence = observation.feedbackFormConfidence;
  const effectivePrecision = observation.basePrecision ?? 2;
  if (!(effectivePrecision > 0)) {
    throw new Error("Route1 observation precision must be positive");
  }
  const rejectedReason = observation.feedbackFormAbstained || !(confidence >= threshold) ? "feedback_form_low_confidence" : effectiveSubtype === "other" ? "reference_type_other" : vector.every((value) => value === 0) ? "empty_target_features" : null;
  if (rejectedReason) {
    return {
      state,
      status: "rejected",
      reason: rejectedReason,
      feedbackForm: observation.feedbackForm,
      effectiveReferenceSubtype: effectiveSubtype,
      referenceConflict: conflict,
      targetFeatures: selected,
      effectivePrecision,
      delta: {}
    };
  }
  const covarianceR = state.covariance.map(
    (row) => row.reduce((sum, value, column) => sum + value * vector[column], 0)
  );
  const rCovarianceR = vector.reduce(
    (sum, value, index) => sum + value * covarianceR[index],
    0
  );
  const denominator = 1 + effectivePrecision * rCovarianceR;
  const activeTarget = vector.reduce((sum, value) => sum + value * value, 0) * observation.valence;
  const projectedMean = vector.reduce(
    (sum, value, index) => sum + value * state.mean[index],
    0
  );
  const innovation = activeTarget - projectedMean;
  const gain = covarianceR.map((value) => effectivePrecision * value / denominator);
  const nextMean = state.mean.map((value, index) => value + gain[index] * innovation);
  const nextCovariance = state.covariance.map(
    (row, rowIndex) => row.map(
      (value, columnIndex) => value - effectivePrecision * covarianceR[rowIndex] * covarianceR[columnIndex] / denominator
    )
  );
  for (let row = 0; row < nextCovariance.length; row += 1) {
    for (let column = row + 1; column < nextCovariance.length; column += 1) {
      const average = (nextCovariance[row][column] + nextCovariance[column][row]) / 2;
      nextCovariance[row][column] = average;
      nextCovariance[column][row] = average;
    }
  }
  return {
    state: {
      features: [...state.features],
      mean: nextMean,
      covariance: nextCovariance
    },
    status: "updated",
    feedbackForm: observation.feedbackForm,
    effectiveReferenceSubtype: effectiveSubtype,
    referenceConflict: conflict,
    targetFeatures: selected,
    effectivePrecision,
    delta: Object.fromEntries(
      state.features.map((feature, index) => [feature, nextMean[index] - state.mean[index]]).filter(([, value]) => Math.abs(value) > 1e-12)
    )
  };
}

// web/lib/subgoal-policy.ts
var SUBGOALS = [
  "GET_TOMATO",
  "PUT_TOMATO_IN_POT",
  "GET_ONION",
  "PUT_ONION_IN_POT",
  "GET_DISH",
  "PICKUP_SOUP",
  "SERVE_SOUP",
  "STASH_HELD_OBJECT",
  "YIELD_PATH",
  "WAIT"
];
var REWARD_FEATURES = [
  "ingredient_tomato",
  "ingredient_onion",
  "pick_tomato",
  "pick_onion",
  "pick_dish",
  "pick_ready_soup",
  "recipe_needs_first_tomato",
  "recipe_needs_second_tomato",
  "recipe_needs_onion",
  "adds_needed_tomato",
  "adds_needed_onion",
  "adds_extra_tomato",
  "adds_extra_onion",
  "wrong_ingredient",
  "breaks_recipe",
  "completes_recipe",
  "matches_current_order",
  "pot_empty",
  "pot_has_one_tomato",
  "pot_has_two_tomatoes",
  "pot_has_two_tomatoes_one_onion",
  "pot_cooking",
  "soup_ready",
  "dish_needed_for_ready_soup",
  "serve_ready_soup",
  "supports_serving",
  "delays_serving",
  "near_tomato_dispenser",
  "near_onion_dispenser",
  "near_dish_dispenser",
  "near_pot",
  "near_serving_counter",
  "moves_toward_needed_object",
  "moves_away_from_needed_object",
  "blocks_partner_on_ring",
  "distance_cost",
  "time_cost",
  "respects_human_intent",
  "avoids_duplicate_human_task",
  "clears_human_shortest_path",
  "clears_human_path",
  "clears_serving_access",
  "blocks_human_path",
  "blocks_serving_route",
  "human_wait_cost",
  "duplicate_human_task",
  "steals_human_target",
  "crowds_human_target",
  "frustrates_human",
  "collision_risk",
  "complementary_to_human",
  "avoids_human_shortest_path",
  "cuts_in_front_of_human"
];
var POLICY_FEATURE_CONTRACT_VERSION = "durf-53d-policy-contract-v1";
var POLICY_ACTIVE_FEATURES = [
  "adds_needed_onion",
  "adds_needed_tomato",
  "avoids_human_shortest_path",
  "avoids_duplicate_human_task",
  "blocks_human_path",
  "blocks_serving_route",
  "clears_human_path",
  "clears_human_shortest_path",
  "clears_serving_access",
  "complementary_to_human",
  "completes_recipe",
  "crowds_human_target",
  "cuts_in_front_of_human",
  "delays_serving",
  "dish_needed_for_ready_soup",
  "distance_cost",
  "duplicate_human_task",
  "frustrates_human",
  "human_wait_cost",
  "ingredient_onion",
  "ingredient_tomato",
  "matches_current_order",
  "moves_toward_needed_object",
  "pick_dish",
  "pick_onion",
  "pick_ready_soup",
  "pick_tomato",
  "recipe_needs_onion",
  "respects_human_intent",
  "serve_ready_soup",
  "steals_human_target",
  "supports_serving",
  "time_cost"
];
var POLICY_CONTEXT_ONLY_FEATURES = [
  "pot_cooking",
  "pot_empty",
  "pot_has_one_tomato",
  "pot_has_two_tomatoes",
  "pot_has_two_tomatoes_one_onion",
  "soup_ready"
];
var POLICY_UNSUPPORTED_FEATURES = [
  "recipe_needs_first_tomato",
  "recipe_needs_second_tomato",
  "adds_extra_tomato",
  "adds_extra_onion",
  "wrong_ingredient",
  "breaks_recipe",
  "near_tomato_dispenser",
  "near_onion_dispenser",
  "near_dish_dispenser",
  "near_pot",
  "near_serving_counter",
  "moves_away_from_needed_object",
  "blocks_partner_on_ring",
  "collision_risk"
];
var POLICY_FEATURE_CONTRACT = Object.freeze({
  schemaVersion: POLICY_FEATURE_CONTRACT_VERSION,
  featureCount: REWARD_FEATURES.length,
  policyActive: POLICY_ACTIVE_FEATURES,
  contextOnly: POLICY_CONTEXT_ONLY_FEATURES,
  unsupported: POLICY_UNSUPPORTED_FEATURES
});
var DEFAULT_RECIPE = ["tomato", "tomato", "onion"];
var REWARD_FEATURE_SET = new Set(REWARD_FEATURES);
var POLICY_ACTIVE_FEATURE_SET = new Set(POLICY_ACTIVE_FEATURES);
var POLICY_DECISION_NULL_FEATURE_SET = /* @__PURE__ */ new Set([
  ...POLICY_CONTEXT_ONLY_FEATURES,
  ...POLICY_UNSUPPORTED_FEATURES
]);
var SUBGOAL_RESOURCE = {
  GET_TOMATO: "tomato",
  PUT_TOMATO_IN_POT: "tomato",
  GET_ONION: "onion",
  PUT_ONION_IN_POT: "onion",
  GET_DISH: "dish",
  PICKUP_SOUP: "soup",
  SERVE_SOUP: "serving",
  STASH_HELD_OBJECT: null,
  YIELD_PATH: null,
  WAIT: null
};
function itemCounts(items) {
  const counts = /* @__PURE__ */ new Map();
  for (const item of items) counts.set(item, (counts.get(item) ?? 0) + 1);
  return counts;
}
function countItem(items, item) {
  return items.reduce((count, value) => count + Number(value === item), 0);
}
function distinct(items) {
  return [...new Set(items)];
}
function nonNegativeInteger(value) {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : 0;
}
function missingIngredients(recipe, current) {
  const remaining = itemCounts(recipe);
  for (const ingredient of current) {
    remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - 1);
  }
  const missing = [];
  for (const ingredient of recipe) {
    if ((remaining.get(ingredient) ?? 0) > 0) {
      missing.push(ingredient);
      remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - 1);
    }
  }
  return missing;
}
function statePotIngredients(state) {
  return [
    ...Array.from({ length: state.pot.tomatoes }, () => "tomato"),
    ...Array.from({ length: state.pot.onions }, () => "onion")
  ];
}
function stateStagedInventory(state) {
  const inventory = {};
  for (const object of Object.values(state.counterObjects)) {
    inventory[object.item] = (inventory[object.item] ?? 0) + 1;
  }
  return inventory;
}
function optionalHolding(override, fallback) {
  return override === void 0 ? fallback : override;
}
function buildSubgoalContext(state, overrides = {}) {
  const potIngredients = statePotIngredients(state);
  return {
    recipe: [...overrides.recipe ?? DEFAULT_RECIPE],
    potIngredients: [...overrides.potIngredients ?? potIngredients],
    potStatus: overrides.potStatus ?? state.pot.stage,
    agentHolding: optionalHolding(overrides.agentHolding, state.partner.held),
    humanHolding: optionalHolding(overrides.humanHolding, state.player.held),
    humanIntent: overrides.humanIntent ?? null,
    humanCommittedUnits: overrides.humanCommittedUnits ?? null,
    candidatePathEffects: { ...overrides.candidatePathEffects },
    candidateTargetOverlapsHuman: {
      ...overrides.candidateTargetOverlapsHuman
    },
    potSnapshots: overrides.potSnapshots ? overrides.potSnapshots.map((snapshot) => ({
      ingredients: [...snapshot.ingredients],
      status: snapshot.status
    })) : null,
    stagedInventory: {
      ...overrides.stagedInventory ?? stateStagedInventory(state)
    }
  };
}
function normalizedPotSnapshots(context) {
  if (context.potSnapshots && context.potSnapshots.length > 0) {
    return context.potSnapshots.map((snapshot) => ({
      ingredients: [...snapshot.ingredients],
      status: String(snapshot.status || "empty").toLowerCase()
    }));
  }
  return [
    {
      ingredients: [...context.potIngredients],
      status: String(context.potStatus || "empty").toLowerCase()
    }
  ];
}
function readySoupCount(context) {
  return normalizedPotSnapshots(context).filter(({ status }) => status === "ready").length;
}
function cookingSoupCount(context) {
  return normalizedPotSnapshots(context).filter(({ status }) => status === "cooking").length;
}
function soupReady(context) {
  return readySoupCount(context) > 0;
}
function soupCooking(context) {
  return cookingSoupCount(context) > 0;
}
function passiveCookingWaitIsValid(context) {
  if (!soupCooking(context) || soupReady(context) || openMissingIngredients(context).length > 0 || ![null, "dish"].includes(context.agentHolding)) {
    return false;
  }
  const recipe = itemCounts(context.recipe);
  return normalizedPotSnapshots(context).some(({ ingredients, status }) => {
    if (status !== "cooking") return false;
    const contents = itemCounts(ingredients);
    return [...recipe].every(
      ([ingredient, count]) => (contents.get(ingredient) ?? 0) >= count
    );
  });
}
function openMissingIngredients(context) {
  const missing = [];
  for (const snapshot of normalizedPotSnapshots(context)) {
    if (snapshot.status === "ready" || snapshot.status === "cooking") continue;
    missing.push(...missingIngredients(context.recipe, snapshot.ingredients));
  }
  return missing;
}
function stagedUnits(context, resource) {
  return nonNegativeInteger(context.stagedInventory[resource]);
}
function humanCoversHeldIngredient(context, held, missing) {
  const remaining = countItem(missing, held);
  if (remaining <= 0 || context.humanHolding !== held) return false;
  const committed = context.humanCommittedUnits ?? 1;
  return nonNegativeInteger(committed) >= remaining;
}
function prefetchIngredientDeficits(context) {
  const remaining = itemCounts(context.recipe);
  for (const ingredient of distinct(context.recipe)) {
    const committed = stagedUnits(context, ingredient) + Number(context.agentHolding === ingredient) + Number(context.humanHolding === ingredient);
    remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - committed);
  }
  const deficits = [];
  for (const ingredient of context.recipe) {
    if ((remaining.get(ingredient) ?? 0) > 0) {
      deficits.push(ingredient);
      remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - 1);
    }
  }
  return deficits;
}
function openWorkNotCoveredByHuman(context, missing) {
  const remaining = itemCounts(missing);
  const held = context.humanHolding;
  if (held === "tomato" || held === "onion") {
    const committed = nonNegativeInteger(context.humanCommittedUnits ?? 1);
    remaining.set(held, Math.max(0, (remaining.get(held) ?? 0) - committed));
  }
  const uncovered = [];
  for (const ingredient of missing) {
    if ((remaining.get(ingredient) ?? 0) > 0) {
      uncovered.push(ingredient);
      remaining.set(ingredient, (remaining.get(ingredient) ?? 0) - 1);
    }
  }
  return uncovered;
}
function dishSupply(context) {
  return stagedUnits(context, "dish") + Number(context.agentHolding === "dish") + Number(context.humanHolding === "dish");
}
function activeSoupsNeedAnotherDish(context) {
  return readySoupCount(context) + cookingSoupCount(context) > dishSupply(context);
}
function enumerateFeasibleSubgoals(context) {
  const missing = openMissingIngredients(context);
  const held = context.agentHolding;
  const candidates = [];
  if (held === "tomato" || held === "onion") {
    if (missing.includes(held)) {
      candidates.push(held === "tomato" ? "PUT_TOMATO_IN_POT" : "PUT_ONION_IN_POT");
      if (humanCoversHeldIngredient(context, held, missing)) {
        candidates.push("STASH_HELD_OBJECT");
      }
    } else {
      candidates.push("STASH_HELD_OBJECT");
    }
  } else if (held === "dish") {
    candidates.push(soupReady(context) ? "PICKUP_SOUP" : "STASH_HELD_OBJECT");
  } else if (held === "soup") {
    candidates.push("SERVE_SOUP");
  } else {
    const uncoveredOpenWork = openWorkNotCoveredByHuman(context, missing);
    const imminentSoupCount = Number(missing.length > 0 && uncoveredOpenWork.length === 0);
    const activeSoupCount = readySoupCount(context) + cookingSoupCount(context);
    const dishWorkUncovered = (soupReady(context) || imminentSoupCount > 0) && activeSoupCount + imminentSoupCount > dishSupply(context);
    if (dishWorkUncovered) candidates.push("GET_DISH");
    for (const ingredient of distinct(uncoveredOpenWork)) {
      candidates.push(ingredient === "tomato" ? "GET_TOMATO" : "GET_ONION");
    }
    if (soupCooking(context) && !soupReady(context) && missing.length === 0) {
      if (activeSoupsNeedAnotherDish(context)) candidates.push("GET_DISH");
      for (const ingredient of distinct(prefetchIngredientDeficits(context))) {
        candidates.push(ingredient === "tomato" ? "GET_TOMATO" : "GET_ONION");
      }
    }
  }
  const feasible = distinct(candidates);
  if (context.candidatePathEffects.YIELD_PATH === "clears") {
    feasible.push("YIELD_PATH");
  }
  feasible.push("WAIT");
  return distinct(feasible);
}
function h0PreferredSubgoal(context) {
  const held = context.agentHolding;
  const missing = openMissingIngredients(context);
  if (held === "tomato" || held === "onion") {
    if (!missing.includes(held)) return "STASH_HELD_OBJECT";
    return held === "tomato" ? "PUT_TOMATO_IN_POT" : "PUT_ONION_IN_POT";
  }
  if (held === "dish") return soupReady(context) ? "PICKUP_SOUP" : "STASH_HELD_OBJECT";
  if (held === "soup") return "SERVE_SOUP";
  if (soupReady(context)) return "GET_DISH";
  if (missing[0] === "tomato") return "GET_TOMATO";
  if (missing[0] === "onion") return "GET_ONION";
  return "WAIT";
}
function chooseH0Subgoal(context, feasibleSubgoals = enumerateFeasibleSubgoals(context)) {
  const preferred = h0PreferredSubgoal(context);
  if (feasibleSubgoals.includes(preferred)) return preferred;
  return feasibleSubgoals[0] ?? "WAIT";
}
function setFeature(features, feature) {
  features[feature] = 1;
}
function addStateFeatures(context, features) {
  for (const snapshot of normalizedPotSnapshots(context)) {
    if (snapshot.ingredients.length === 0) setFeature(features, "pot_empty");
    const tomatoes = countItem(snapshot.ingredients, "tomato");
    const onions = countItem(snapshot.ingredients, "onion");
    if (tomatoes >= 1) setFeature(features, "pot_has_one_tomato");
    if (tomatoes >= 2) setFeature(features, "pot_has_two_tomatoes");
    if (tomatoes >= 2 && onions >= 1) {
      setFeature(features, "pot_has_two_tomatoes_one_onion");
    }
  }
  if (soupCooking(context)) setFeature(features, "pot_cooking");
  if (soupReady(context)) setFeature(features, "soup_ready");
}
function addTaskFeatures(context, subgoal, features) {
  const missing = openMissingIngredients(context);
  if (subgoal === "GET_TOMATO" || subgoal === "PUT_TOMATO_IN_POT") {
    setFeature(features, "ingredient_tomato");
    if (subgoal === "GET_TOMATO") setFeature(features, "pick_tomato");
    if (missing.includes("tomato")) {
      setFeature(features, "adds_needed_tomato");
      setFeature(features, "moves_toward_needed_object");
      setFeature(features, "matches_current_order");
    } else if (subgoal === "GET_TOMATO" && soupCooking(context)) {
      setFeature(features, "moves_toward_needed_object");
    } else {
      setFeature(features, "adds_extra_tomato");
      setFeature(features, "wrong_ingredient");
      setFeature(features, "breaks_recipe");
    }
  } else if (subgoal === "GET_ONION" || subgoal === "PUT_ONION_IN_POT") {
    setFeature(features, "ingredient_onion");
    if (subgoal === "GET_ONION") setFeature(features, "pick_onion");
    if (missing.includes("onion")) {
      setFeature(features, "adds_needed_onion");
      setFeature(features, "recipe_needs_onion");
      setFeature(features, "moves_toward_needed_object");
      setFeature(features, "matches_current_order");
    } else if (subgoal === "GET_ONION" && soupCooking(context)) {
      setFeature(features, "moves_toward_needed_object");
    } else {
      setFeature(features, "adds_extra_onion");
      setFeature(features, "wrong_ingredient");
      setFeature(features, "breaks_recipe");
    }
  } else if (subgoal === "GET_DISH") {
    setFeature(features, "pick_dish");
    if (soupReady(context)) {
      setFeature(features, "dish_needed_for_ready_soup");
      setFeature(features, "supports_serving");
    } else if (!soupCooking(context)) {
      setFeature(features, "time_cost");
      setFeature(features, "delays_serving");
    }
  } else if (subgoal === "PICKUP_SOUP") {
    setFeature(features, "pick_ready_soup");
    if (soupReady(context)) setFeature(features, "supports_serving");
    else setFeature(features, "time_cost");
  } else if (subgoal === "SERVE_SOUP") {
    if (context.agentHolding === "soup" || soupReady(context)) {
      setFeature(features, "serve_ready_soup");
      setFeature(features, "supports_serving");
      setFeature(features, "completes_recipe");
    } else {
      setFeature(features, "time_cost");
    }
  } else if (subgoal === "STASH_HELD_OBJECT") {
    setFeature(features, soupReady(context) ? "supports_serving" : "time_cost");
  } else if (subgoal === "YIELD_PATH") {
    setFeature(features, "time_cost");
    setFeature(features, "distance_cost");
  } else if (subgoal === "WAIT") {
    const passiveCookingOnly = soupCooking(context) && !soupReady(context) && openMissingIngredients(context).length === 0;
    if (!passiveCookingOnly || context.agentHolding !== null) {
      setFeature(features, "time_cost");
    }
    if (soupReady(context) || missing.length > 0) setFeature(features, "delays_serving");
    if (context.agentHolding !== null) setFeature(features, "delays_serving");
  }
}
function resourceFromText(text) {
  if (!text) return null;
  const lowered = text.toLowerCase();
  const mappings = [
    [["dish", "plate"], "dish"],
    [["serve", "serving", "deliver"], "serving"],
    [["soup"], "soup"],
    [["tomato"], "tomato"],
    [["onion"], "onion"]
  ];
  for (const [keywords, resource] of mappings) {
    if (keywords.some((keyword) => lowered.includes(keyword))) return resource;
  }
  return null;
}
function humanTargetResource(context) {
  return resourceFromText(context.humanIntent) ?? resourceFromText(context.humanHolding);
}
function addCoordinationFeatures(context, subgoal, features) {
  const humanTarget = humanTargetResource(context);
  const pathEffect = context.candidatePathEffects[subgoal];
  const blocksPath = pathEffect === "blocks" || pathEffect === "enters";
  const clearsPath = pathEffect === "clears";
  if (humanTarget === null) {
    if (subgoal === "YIELD_PATH" && clearsPath) {
      setFeature(features, "respects_human_intent");
      setFeature(features, "clears_human_path");
      setFeature(features, "clears_human_shortest_path");
      setFeature(features, "avoids_human_shortest_path");
    }
    return;
  }
  const subgoalResource = SUBGOAL_RESOURCE[subgoal];
  const servingChain = /* @__PURE__ */ new Set(["dish", "soup", "serving"]);
  const missing = openMissingIngredients(context);
  const remainingUnits = (resource) => {
    if (resource === "tomato" || resource === "onion") {
      const units = countItem(missing, resource);
      return units === 0 && soupCooking(context) ? countItem(context.recipe, resource) : units;
    }
    if (resource === "dish" || resource === "soup") return readySoupCount(context);
    return readySoupCount(context) + Number(context.agentHolding === "soup") + Number(context.humanHolding === "soup");
  };
  const committedUnits = nonNegativeInteger(context.humanCommittedUnits ?? 1);
  const sameResource = subgoalResource !== null && subgoalResource === humanTarget;
  const targetOverlap = context.candidateTargetOverlapsHuman[subgoal] ?? sameResource;
  let sameWork = sameResource || targetOverlap && servingChain.has(humanTarget) && subgoalResource !== null && servingChain.has(subgoalResource);
  if (subgoal === "SERVE_SOUP" && context.agentHolding === "soup" && context.humanHolding === "soup") {
    sameWork = false;
  }
  const humanCoversWork = sameWork && remainingUnits(humanTarget) > 0 && committedUnits >= remainingUnits(humanTarget);
  const releasesCoveredDuplicate = subgoal === "STASH_HELD_OBJECT" && (context.agentHolding === "tomato" || context.agentHolding === "onion") && context.agentHolding === context.humanHolding && context.agentHolding === humanTarget && remainingUnits(humanTarget) > 0 && committedUnits >= remainingUnits(humanTarget);
  if (releasesCoveredDuplicate) {
    setFeature(features, "respects_human_intent");
    setFeature(features, "avoids_duplicate_human_task");
  } else if (humanCoversWork) {
    setFeature(features, "duplicate_human_task");
    if (targetOverlap) setFeature(features, "crowds_human_target");
    if (targetOverlap && context.humanHolding === null && ["GET_DISH", "GET_TOMATO", "GET_ONION", "PICKUP_SOUP"].includes(subgoal)) {
      setFeature(features, "steals_human_target");
    }
  } else if (subgoalResource !== null) {
    setFeature(features, "complementary_to_human");
    setFeature(features, "respects_human_intent");
    setFeature(features, "avoids_duplicate_human_task");
  } else if (subgoal === "WAIT") {
    const targetUnits = remainingUnits(humanTarget);
    if (targetUnits > 0 && committedUnits >= targetUnits && !blocksPath) {
      setFeature(features, "respects_human_intent");
      setFeature(features, "avoids_duplicate_human_task");
    }
  }
  if (blocksPath) {
    setFeature(features, "blocks_human_path");
    setFeature(features, "human_wait_cost");
    setFeature(features, "frustrates_human");
    if (pathEffect === "enters") setFeature(features, "cuts_in_front_of_human");
    if (servingChain.has(humanTarget)) setFeature(features, "blocks_serving_route");
  } else if (clearsPath) {
    setFeature(features, "clears_human_path");
    setFeature(features, "clears_human_shortest_path");
    setFeature(features, "avoids_human_shortest_path");
    if (subgoal === "YIELD_PATH") setFeature(features, "respects_human_intent");
    if (servingChain.has(humanTarget)) setFeature(features, "clears_serving_access");
  }
}
function featurizeSubgoal(context, subgoal) {
  const features = {};
  addStateFeatures(context, features);
  addTaskFeatures(context, subgoal, features);
  addCoordinationFeatures(context, subgoal, features);
  return features;
}
function normalizeWeights(weights2) {
  const raw = weights2;
  const invalidFeatures = Object.entries(raw).filter(
    ([feature, value]) => !REWARD_FEATURE_SET.has(feature) || typeof value !== "number" || !Number.isFinite(value)
  ).map(([feature]) => feature).sort();
  const safe = invalidFeatures.length === 0;
  const values = Object.fromEntries(
    REWARD_FEATURES.map((feature) => [
      feature,
      safe && POLICY_ACTIVE_FEATURE_SET.has(feature) && typeof raw[feature] === "number" ? raw[feature] : 0
    ])
  );
  const activeFeatures2 = safe ? POLICY_ACTIVE_FEATURES.filter(
    (feature) => typeof raw[feature] === "number" && raw[feature] !== 0
  ) : [];
  const ignoredDecisionNullFeatures = safe ? REWARD_FEATURES.filter(
    (feature) => POLICY_DECISION_NULL_FEATURE_SET.has(feature) && typeof raw[feature] === "number" && raw[feature] !== 0
  ) : [];
  return {
    values,
    invalidFeatures,
    activeFeatures: activeFeatures2,
    ignoredDecisionNullFeatures
  };
}
function scoreFeatureVector(weights2, features) {
  return POLICY_ACTIVE_FEATURES.reduce(
    (score, feature) => score + weights2[feature] * (features[feature] ?? 0),
    0
  );
}
function rankWithWeights(context, feasible, weights2) {
  return feasible.map((subgoal, order) => {
    const features = featurizeSubgoal(context, subgoal);
    return { subgoal, score: scoreFeatureVector(weights2, features), features, order };
  }).sort((left, right) => right.score - left.score || left.order - right.order).map(({ order: _order, ...ranked }) => ranked);
}
function selectedContributions(features, weights2, limit) {
  return POLICY_ACTIVE_FEATURES.flatMap((feature) => {
    const value = features[feature] ?? 0;
    const weight = weights2[feature];
    const contribution = value * weight;
    return contribution === 0 ? [] : [{ feature, value, weight, contribution }];
  }).sort(
    (left, right) => Math.abs(right.contribution) - Math.abs(left.contribution) || left.feature.localeCompare(right.feature)
  ).slice(0, limit);
}
function chooseRewardSubgoal(state, weights2, options = {}) {
  const context = buildSubgoalContext(state, options.contextOverrides);
  const taskFeasible = enumerateFeasibleSubgoals(context);
  const requestedFeasible = options.feasibleSubgoals ? new Set(options.feasibleSubgoals) : null;
  const feasible = requestedFeasible ? taskFeasible.filter((subgoal) => requestedFeasible.has(subgoal)) : taskFeasible;
  if (feasible.length === 0) feasible.push("WAIT");
  const normalized = normalizeWeights(weights2);
  let ranking = rankWithWeights(context, feasible, normalized.values);
  const scoreOverflow = ranking.some(({ score }) => !Number.isFinite(score));
  if (scoreOverflow) {
    normalized.invalidFeatures.push("<score-overflow>");
    for (const feature of REWARD_FEATURES) normalized.values[feature] = 0;
    normalized.activeFeatures = [];
    ranking = rankWithWeights(context, feasible, normalized.values);
  }
  const requestedFallback = options.h0Fallback;
  const h0Fallback = requestedFallback && feasible.includes(requestedFallback) ? requestedFallback : chooseH0Subgoal(context, feasible);
  const tolerance = typeof options.tieTolerance === "number" && Number.isFinite(options.tieTolerance) && options.tieTolerance >= 0 ? options.tieTolerance : 1e-9;
  const topScore = ranking[0]?.score ?? 0;
  const tied = ranking.filter(({ score }) => Math.abs(score - topScore) <= tolerance);
  const tieFallback = tied.find(({ subgoal }) => subgoal === h0Fallback);
  const selected = tieFallback ?? ranking[0];
  const invalidWeights = normalized.invalidFeatures.length > 0;
  let decisionSource;
  if (invalidWeights) decisionSource = "invalid_weights_h0_fallback";
  else if (normalized.activeFeatures.length === 0 && normalized.ignoredDecisionNullFeatures.length > 0) {
    decisionSource = "decision_null_weights_h0_fallback";
  } else if (tied.length === 1) decisionSource = "reward_argmax";
  else if (tieFallback) decisionSource = "h0_tie_fallback";
  else decisionSource = "unresolved_reward_tie";
  const contributionLimit = typeof options.topContributionLimit === "number" && Number.isFinite(options.topContributionLimit) ? Math.max(0, Math.trunc(options.topContributionLimit)) : 5;
  return {
    chosenSubgoal: selected.subgoal,
    feasibleSubgoals: feasible,
    ranking,
    score: selected.score,
    decisionSource,
    rewardMargin: ranking.length > 1 ? ranking[0].score - ranking[1].score : null,
    topContributions: selectedContributions(
      selected.features,
      normalized.values,
      contributionLimit
    ),
    usedLearnedWeights: !invalidWeights && normalized.activeFeatures.length > 0 && tied.length === 1,
    h0Fallback,
    invalidWeightFeatures: normalized.invalidFeatures,
    policyFeatureContractVersion: POLICY_FEATURE_CONTRACT_VERSION,
    activeWeightFeatures: normalized.activeFeatures,
    ignoredDecisionNullWeightFeatures: normalized.ignoredDecisionNullFeatures
  };
}

// web/lib/game.ts
var SOURCE_LAYOUT_ROWS = [
  "XXXXPXXXXX",
  "D  1  2  X",
  "X XXXXXX X",
  "X XXXXXX S",
  "T        X",
  "XXXXOXXXXX"
];
var TERRAIN_ROWS = SOURCE_LAYOUT_ROWS.map(
  (row) => row.replace(/[12]/gu, " ")
);
var BOARD_WIDTH = 10;
var BOARD_HEIGHT = 6;
var ROUND_STEPS = 800;
var STEPS_PER_SECOND = 2;
var GAME_STEP_INTERVAL_MS = 1e3 / STEPS_PER_SECOND;
var ROUND_SECONDS = ROUND_STEPS / STEPS_PER_SECOND;
var COOK_TIME_STEPS = 20;
var CORRECT_SOUP_REWARD = 20;
var AI_LIVENESS_CONTRACT_VERSION = "durf-web-ai-liveness-v1";
var MAX_CONSECUTIVE_WAIT_STEPS = 3;
var PLAYER_STARTS = {
  partner: { x: 3, y: 1 },
  player: { x: 6, y: 1 }
};
var STATIONS = [
  { kind: "pot", terrain: "P", label: "Soup pot", shortLabel: "Pot", position: { x: 4, y: 0 } },
  { kind: "dish", terrain: "D", label: "Dish dispenser", shortLabel: "Dish", position: { x: 0, y: 1 } },
  { kind: "serve", terrain: "S", label: "Serving window", shortLabel: "Serve", position: { x: 9, y: 3 } },
  { kind: "tomato", terrain: "T", label: "Tomato dispenser", shortLabel: "Tomato", position: { x: 0, y: 4 } },
  { kind: "onion", terrain: "O", label: "Onion dispenser", shortLabel: "Onion", position: { x: 4, y: 5 } }
];
var DIRECTION_DELTAS = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 }
};
var DIRECTIONS = [
  ["up", DIRECTION_DELTAS.up],
  ["left", DIRECTION_DELTAS.left],
  ["right", DIRECTION_DELTAS.right],
  ["down", DIRECTION_DELTAS.down]
];
function newOrder(index) {
  return {
    id: `two-tomato-one-onion-soup-${index}`,
    recipe: "two-tomato-one-onion-soup",
    reward: CORRECT_SOUP_REWARD
  };
}
function emptyPot() {
  return { stage: "empty", secondsRemaining: 0, tomatoes: 0, onions: 0 };
}
function event(code, message, actor = null, extra = {}) {
  return { code, actor, message, ...extra };
}
function withEvents(state, events) {
  const effectiveEvents = events.length > 0 ? events : [event("wait", "Both chefs waited.")];
  return {
    ...state,
    lastAction: effectiveEvents.map((item) => item.message).join(" "),
    lastEventCode: effectiveEvents[effectiveEvents.length - 1].code,
    lastStepEvents: effectiveEvents
  };
}
function createGameState(status = "waiting") {
  const initialEvent = event("waiting", "Waiting for research consent.");
  return {
    status,
    score: 0,
    secondsLeft: ROUND_SECONDS,
    ordersCompleted: 0,
    tick: 0,
    player: { ...PLAYER_STARTS.player, held: null, facing: "up" },
    partner: { ...PLAYER_STARTS.partner, held: null, facing: "up" },
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
      cycleBlockedStashItem: null
    }
  };
}
function isWalkable(point) {
  return point.x >= 0 && point.x < BOARD_WIDTH && point.y >= 0 && point.y < BOARD_HEIGHT && TERRAIN_ROWS[point.y][point.x] === " ";
}
function isCounter(point) {
  return point.x >= 0 && point.x < BOARD_WIDTH && point.y >= 0 && point.y < BOARD_HEIGHT && TERRAIN_ROWS[point.y][point.x] === "X";
}
function counterKey(point) {
  return `${point.x},${point.y}`;
}
function isSamePoint(left, right) {
  return left.x === right.x && left.y === right.y;
}
function pointAhead(chef) {
  const delta = DIRECTION_DELTAS[chef.facing];
  return { x: chef.x + delta.x, y: chef.y + delta.y };
}
function stationAhead(chef) {
  const ahead = pointAhead(chef);
  return STATIONS.find((station) => isSamePoint(ahead, station.position))?.kind ?? null;
}
function actorName(actor) {
  return actor === "ai" ? "AI" : "Human";
}
function cloneSoupContents(contents) {
  return contents ? { ...contents } : void 0;
}
function failedInteraction(chef, pot, counterObjects, actor) {
  return {
    chef,
    pot,
    counterObjects,
    reward: 0,
    completedOrder: false,
    event: event("interaction_failed", `${actorName(actor)} cannot interact here.`, actor)
  };
}
function interactChef(chef, pot, counterObjects, actor) {
  const ahead = pointAhead(chef);
  const name = actorName(actor);
  if (isCounter(ahead)) {
    const key = counterKey(ahead);
    const counterObject = counterObjects[key];
    if (chef.held !== null && counterObject === void 0) {
      const nextCounters = { ...counterObjects };
      nextCounters[key] = {
        item: chef.held,
        soupContents: chef.held === "soup" ? cloneSoupContents(chef.heldSoup) : void 0
      };
      const droppedItem = chef.held;
      return {
        chef: { ...chef, held: null, heldSoup: void 0 },
        pot,
        counterObjects: nextCounters,
        reward: 0,
        completedOrder: false,
        event: event(
          "drop_counter",
          `${name} placed ${itemLabel(droppedItem)} on a counter.`,
          actor,
          { item: droppedItem, position: ahead }
        )
      };
    }
    if (chef.held === null && counterObject !== void 0) {
      const nextCounters = { ...counterObjects };
      delete nextCounters[key];
      return {
        chef: {
          ...chef,
          held: counterObject.item,
          heldSoup: counterObject.item === "soup" ? cloneSoupContents(counterObject.soupContents) : void 0
        },
        pot,
        counterObjects: nextCounters,
        reward: 0,
        completedOrder: false,
        event: event(
          "pickup_counter",
          `${name} picked up ${itemLabel(counterObject.item)} from a counter.`,
          actor,
          { item: counterObject.item, position: ahead }
        )
      };
    }
    return failedInteraction(chef, pot, counterObjects, actor);
  }
  const station = stationAhead(chef);
  if (station === "tomato" && chef.held === null) {
    return {
      chef: { ...chef, held: "tomato", heldSoup: void 0 },
      pot,
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event("pick_tomato", `${name} picked up a tomato.`, actor, { item: "tomato" })
    };
  }
  if (station === "onion" && chef.held === null) {
    return {
      chef: { ...chef, held: "onion", heldSoup: void 0 },
      pot,
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event("pick_onion", `${name} picked up an onion.`, actor, { item: "onion" })
    };
  }
  if (station === "dish" && chef.held === null) {
    return {
      chef: { ...chef, held: "dish", heldSoup: void 0 },
      pot,
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event("pick_dish", `${name} picked up a dish.`, actor, { item: "dish" })
    };
  }
  const ingredientCount = pot.tomatoes + pot.onions;
  if (station === "pot" && (chef.held === "tomato" || chef.held === "onion") && ingredientCount < 3 && (pot.stage === "empty" || pot.stage === "filling")) {
    const ingredient = chef.held;
    const nextPot = {
      ...pot,
      stage: "filling",
      tomatoes: pot.tomatoes + Number(ingredient === "tomato"),
      onions: pot.onions + Number(ingredient === "onion")
    };
    return {
      chef: { ...chef, held: null, heldSoup: void 0 },
      pot: nextPot,
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event(
        ingredient === "tomato" ? "pot_tomato" : "pot_onion",
        `${name} added ${itemLabel(ingredient)} to the pot.`,
        actor,
        { item: ingredient }
      )
    };
  }
  if (station === "pot" && chef.held === "dish" && pot.stage === "ready") {
    return {
      chef: {
        ...chef,
        held: "soup",
        heldSoup: { tomatoes: pot.tomatoes, onions: pot.onions }
      },
      pot: emptyPot(),
      counterObjects,
      reward: 0,
      completedOrder: false,
      event: event("pick_soup", `${name} plated the cooked soup.`, actor, { item: "soup" })
    };
  }
  if (station === "serve" && chef.held === "soup") {
    const contents = chef.heldSoup ?? { tomatoes: 2, onions: 1 };
    const correct = contents.tomatoes === 2 && contents.onions === 1;
    return {
      chef: { ...chef, held: null, heldSoup: void 0 },
      pot,
      counterObjects,
      reward: correct ? CORRECT_SOUP_REWARD : 0,
      completedOrder: correct,
      event: event(
        correct ? "serve_correct_soup" : "serve_wrong_soup",
        correct ? `${name} delivered the correct tomato-onion soup.` : `${name} delivered a soup that does not match the order.`,
        actor,
        { item: "soup" }
      )
    };
  }
  return failedInteraction(chef, pot, counterObjects, actor);
}
function applyActorInteraction(state, chefKey, actor) {
  const result = interactChef(
    state[chefKey],
    state.pot,
    state.counterObjects,
    actor
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
      order: result.completedOrder ? newOrder(completed + 1) : state.order
    },
    event: result.event
  };
}
function intendedChef(chef, action) {
  if (!(action in DIRECTION_DELTAS)) return chef;
  const direction = action;
  const delta = DIRECTION_DELTAS[direction];
  const target = { x: chef.x + delta.x, y: chef.y + delta.y };
  return {
    ...chef,
    ...isWalkable(target) ? target : { x: chef.x, y: chef.y },
    facing: direction
  };
}
function resolveMovements(state, aiAction, humanAction) {
  const oldAi = state.partner;
  const oldHuman = state.player;
  const intendedAi = intendedChef(oldAi, aiAction);
  const intendedHuman = intendedChef(oldHuman, humanAction);
  const collision = isSamePoint(intendedAi, intendedHuman) || isSamePoint(intendedAi, oldHuman) && isSamePoint(intendedHuman, oldAi);
  const nextAi = collision ? { ...oldAi, facing: intendedAi.facing } : intendedAi;
  const nextHuman = collision ? { ...oldHuman, facing: intendedHuman.facing } : intendedHuman;
  const events = [];
  if (collision && (aiAction in DIRECTION_DELTAS || humanAction in DIRECTION_DELTAS)) {
    events.push(event("movement_collision", "Both chefs stayed in place because their moves collided."));
  } else {
    for (const [actor, action, before, after] of [
      ["ai", aiAction, oldAi, nextAi],
      ["human", humanAction, oldHuman, nextHuman]
    ]) {
      if (!(action in DIRECTION_DELTAS)) continue;
      if (isSamePoint(before, after)) {
        events.push(event("movement_blocked", `${actorName(actor)} faced a blocked tile.`, actor));
      } else {
        events.push(
          event("move", `${actorName(actor)} moved to ${after.x},${after.y}.`, actor, {
            position: { x: after.x, y: after.y }
          })
        );
      }
    }
  }
  return {
    state: { ...state, partner: nextAi, player: nextHuman },
    events
  };
}
function applyEnvironmentEffects(state) {
  const events = [];
  let pot = state.pot;
  if (pot.stage === "filling" && pot.tomatoes + pot.onions === 3) {
    pot = { ...pot, stage: "cooking", secondsRemaining: COOK_TIME_STEPS };
    events.push(event("cooking_started", "The full pot started cooking automatically."));
  }
  if (pot.stage === "cooking") {
    if (pot.secondsRemaining <= 1) {
      pot = { ...pot, stage: "ready", secondsRemaining: 0 };
      events.push(event("soup_ready", "The soup is ready."));
    } else {
      pot = { ...pot, secondsRemaining: pot.secondsRemaining - 1 };
    }
  }
  const tick = Math.min(ROUND_STEPS, state.tick + 1);
  const stepsLeft = Math.max(0, ROUND_STEPS - tick);
  const status = stepsLeft === 0 ? "finished" : state.status;
  if (status === "finished") {
    events.push(event("round_finished", "The 800-step round is complete."));
  }
  return {
    state: {
      ...state,
      tick,
      secondsLeft: Math.ceil(stepsLeft / STEPS_PER_SECOND),
      status,
      pot
    },
    events
  };
}
function aiObjectTransfer(events) {
  const aiEvent = events.find(
    (item) => item.actor === "ai" && ["pick_tomato", "pick_onion", "pick_dish", "pick_soup", "pickup_counter", "drop_counter"].includes(
      item.code
    )
  );
  if (!aiEvent?.item) return null;
  return {
    kind: aiEvent.code === "drop_counter" ? "stash" : "get",
    item: aiEvent.item
  };
}
function updateAiPolicyLiveness(previous, aiAction, events) {
  const prior2 = previous.aiPolicyLiveness ?? {
    contractVersion: AI_LIVENESS_CONTRACT_VERSION,
    consecutiveWaitSteps: 0,
    recentObjectTransfer: null,
    cycleBlockedStashItem: null
  };
  const taskProgressed = events.some(
    (item) => [
      "pot_tomato",
      "pot_onion",
      "pick_soup",
      "serve_correct_soup",
      "serve_wrong_soup",
      "cooking_started",
      "soup_ready"
    ].includes(item.code)
  );
  const transfer = aiObjectTransfer(events);
  const completedGetStashCycle = Boolean(
    transfer?.kind === "stash" && prior2.recentObjectTransfer?.kind === "get" && transfer.item === prior2.recentObjectTransfer.item
  );
  return {
    contractVersion: AI_LIVENESS_CONTRACT_VERSION,
    consecutiveWaitSteps: aiAction === "stay" && !taskProgressed ? prior2.consecutiveWaitSteps + 1 : 0,
    recentObjectTransfer: taskProgressed ? null : transfer ?? prior2.recentObjectTransfer,
    cycleBlockedStashItem: taskProgressed ? null : completedGetStashCycle ? transfer.item : prior2.cycleBlockedStashItem
  };
}
function stepGame(state, aiAction, humanAction) {
  if (state.status !== "running") return state;
  let next = state;
  const events = [];
  if (aiAction === "interact") {
    const result = applyActorInteraction(next, "partner", "ai");
    next = result.state;
    events.push(result.event);
  }
  if (humanAction === "interact") {
    const result = applyActorInteraction(next, "player", "human");
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
    aiPolicyLiveness: updateAiPolicyLiveness(state, aiAction, events)
  };
  return withEvents(next, events);
}
function pathDistance(start, target, blocked) {
  if (isSamePoint(start, target)) return 0;
  const key = (point) => counterKey(point);
  const queue = [{ point: start, distance: 0 }];
  const visited = /* @__PURE__ */ new Set([key(start)]);
  while (queue.length > 0) {
    const current = queue.shift();
    for (const [, delta] of DIRECTIONS) {
      const point = { x: current.point.x + delta.x, y: current.point.y + delta.y };
      if (!isWalkable(point) || blocked !== null && isSamePoint(point, blocked) || visited.has(key(point))) continue;
      if (isSamePoint(point, target)) return current.distance + 1;
      visited.add(key(point));
      queue.push({ point, distance: current.distance + 1 });
    }
  }
  return Number.POSITIVE_INFINITY;
}
function interactionGoal(state, target) {
  const candidates = [];
  for (const [facing, delta] of DIRECTIONS) {
    const access = { x: target.x - delta.x, y: target.y - delta.y };
    if (!isWalkable(access)) continue;
    candidates.push({ target, access, facing });
  }
  candidates.sort(
    (left, right) => pathDistance(state.partner, left.access, state.player) - pathDistance(state.partner, right.access, state.player)
  );
  return candidates.find(
    (candidate) => Number.isFinite(pathDistance(state.partner, candidate.access, state.player))
  ) ?? null;
}
function stationGoal(state, kind) {
  const station = STATIONS.find((item) => item.kind === kind);
  return station ? interactionGoal(state, station.position) : null;
}
function counterGoals(state, predicate) {
  const goals = [];
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
      const distanceDifference = pathDistance(state.partner, left.access, state.player) - pathDistance(state.partner, right.access, state.player);
      if (distanceDifference !== 0) return distanceDifference;
      const leftTurnCost = isSamePoint(state.partner, left.access) && state.partner.facing === left.facing ? 0 : 1;
      const rightTurnCost = isSamePoint(state.partner, right.access) && state.partner.facing === right.facing ? 0 : 1;
      return leftTurnCost - rightTurnCost;
    }
  );
}
function actionTowardGoal(state, goal) {
  if (!goal) return "stay";
  if (isSamePoint(state.partner, goal.access)) {
    return state.partner.facing === goal.facing ? "interact" : goal.facing;
  }
  const key = (point) => counterKey(point);
  const queue = [{ point: state.partner }];
  const visited = /* @__PURE__ */ new Set([key(state.partner)]);
  while (queue.length > 0) {
    const current = queue.shift();
    for (const [direction, delta] of DIRECTIONS) {
      const point = { x: current.point.x + delta.x, y: current.point.y + delta.y };
      if (!isWalkable(point) || isSamePoint(point, state.player) || visited.has(key(point))) continue;
      const first = current.first ?? direction;
      if (isSamePoint(point, goal.access)) return first;
      visited.add(key(point));
      queue.push({ point, first });
    }
  }
  return "stay";
}
function firstCounterGoal(state, predicate) {
  return counterGoals(state, predicate)[0] ?? null;
}
function stationPosition(kind) {
  return STATIONS.find((station) => station.kind === kind)?.position ?? null;
}
function interactionDistance(start, target, blocked) {
  let best = Number.POSITIVE_INFINITY;
  for (const [, delta] of DIRECTIONS) {
    const access = { x: target.x - delta.x, y: target.y - delta.y };
    if (!isWalkable(access) || blocked !== null && isSamePoint(access, blocked)) continue;
    best = Math.min(best, pathDistance(start, access, blocked));
  }
  return best;
}
function targetForAiSubgoal(state, subgoal) {
  switch (subgoal) {
    case "GET_TOMATO":
      return firstCounterGoal(state, (object) => object?.item === "tomato")?.target ?? stationPosition("tomato");
    case "PUT_TOMATO_IN_POT":
    case "PUT_ONION_IN_POT":
      return stationPosition("pot");
    case "GET_ONION":
      return firstCounterGoal(state, (object) => object?.item === "onion")?.target ?? stationPosition("onion");
    case "GET_DISH":
      return firstCounterGoal(state, (object) => object?.item === "dish")?.target ?? stationPosition("dish");
    case "PICKUP_SOUP":
      return state.partner.held === "dish" ? stationPosition("pot") : firstCounterGoal(state, (object) => object?.item === "soup")?.target ?? null;
    case "SERVE_SOUP":
      return stationPosition("serve");
    case "STASH_HELD_OBJECT":
      return firstCounterGoal(state, (object) => object === void 0)?.target ?? null;
    case "YIELD_PATH":
    case "WAIT":
      return null;
  }
}
function humanTargetFromState(state) {
  switch (state.player.held) {
    case "tomato":
      return { resource: "tomato", target: stationPosition("pot") };
    case "onion":
      return { resource: "onion", target: stationPosition("pot") };
    case "dish":
      return { resource: "dish", target: stationPosition("pot") };
    case "soup":
      return { resource: "soup", target: stationPosition("serve") };
    default: {
      const target = pointAhead(state.player);
      const counterObject = isCounter(target) ? state.counterObjects[counterKey(target)] : void 0;
      if (counterObject) return { resource: counterObject.item, target };
      const station = STATIONS.find((item) => isSamePoint(item.position, target));
      if (station?.kind === "tomato" || station?.kind === "onion" || station?.kind === "dish") {
        return { resource: station.kind, target };
      }
      return null;
    }
  }
}
function intendedSoloPosition(state, action) {
  if (!(action in DIRECTION_DELTAS)) return state.partner;
  const delta = DIRECTION_DELTAS[action];
  const target = { x: state.partner.x + delta.x, y: state.partner.y + delta.y };
  return isWalkable(target) && !isSamePoint(target, state.player) ? target : state.partner;
}
function computeYieldPathAction(state) {
  const humanTarget = humanTargetFromState(state);
  if (!humanTarget) return null;
  const currentDistance = interactionDistance(
    state.player,
    humanTarget.target,
    state.partner
  );
  const candidates = DIRECTIONS.flatMap(([direction, delta]) => {
    const destination = {
      x: state.partner.x + delta.x,
      y: state.partner.y + delta.y
    };
    if (!isWalkable(destination) || isSamePoint(destination, state.player)) return [];
    const resultingDistance = interactionDistance(
      state.player,
      humanTarget.target,
      destination
    );
    if (!(resultingDistance < currentDistance)) return [];
    return [{ direction, resultingDistance }];
  });
  candidates.sort(
    (left, right) => left.resultingDistance - right.resultingDistance
  );
  return candidates[0]?.direction ?? null;
}
function computeLivePolicyGeometry(state) {
  const humanTarget = humanTargetFromState(state);
  const baselineDistance = humanTarget ? interactionDistance(state.player, humanTarget.target, null) : Number.POSITIVE_INFINITY;
  const currentDistance = humanTarget ? interactionDistance(state.player, humanTarget.target, state.partner) : Number.POSITIVE_INFINITY;
  const candidatePathEffects = {};
  const candidateTargetOverlapsHuman = {};
  const yieldAction = computeYieldPathAction(state);
  for (const subgoal of SUBGOALS) {
    const target = targetForAiSubgoal(state, subgoal);
    if (humanTarget && target) {
      candidateTargetOverlapsHuman[subgoal] = isSamePoint(target, humanTarget.target);
    }
    const action = subgoal === "YIELD_PATH" ? yieldAction ?? "stay" : actionForAiSubgoal(state, subgoal);
    const nextPosition = intendedSoloPosition(state, action);
    if (humanTarget) {
      const nextDistance = interactionDistance(
        state.player,
        humanTarget.target,
        nextPosition
      );
      if (nextDistance < currentDistance) {
        candidatePathEffects[subgoal] = "clears";
      } else if (nextDistance > currentDistance) {
        candidatePathEffects[subgoal] = currentDistance <= baselineDistance ? "enters" : "blocks";
      } else if (currentDistance > baselineDistance) {
        candidatePathEffects[subgoal] = "blocks";
      }
    }
  }
  if (yieldAction) candidatePathEffects.YIELD_PATH = "clears";
  return {
    candidatePathEffects,
    candidateTargetOverlapsHuman,
    humanIntent: humanTarget?.resource ?? null,
    yieldAction
  };
}
function actionForAiSubgoal(state, subgoal) {
  switch (subgoal) {
    case "GET_TOMATO":
      return actionTowardGoal(
        state,
        firstCounterGoal(state, (object) => object?.item === "tomato") ?? stationGoal(state, "tomato")
      );
    case "PUT_TOMATO_IN_POT":
    case "PUT_ONION_IN_POT":
      return actionTowardGoal(state, stationGoal(state, "pot"));
    case "GET_ONION":
      return actionTowardGoal(
        state,
        firstCounterGoal(state, (object) => object?.item === "onion") ?? stationGoal(state, "onion")
      );
    case "GET_DISH":
      return actionTowardGoal(
        state,
        firstCounterGoal(state, (object) => object?.item === "dish") ?? stationGoal(state, "dish")
      );
    case "PICKUP_SOUP":
      return actionTowardGoal(
        state,
        state.partner.held === "dish" ? stationGoal(state, "pot") : firstCounterGoal(state, (object) => object?.item === "soup")
      );
    case "SERVE_SOUP":
      return actionTowardGoal(state, stationGoal(state, "serve"));
    case "STASH_HELD_OBJECT":
      return actionTowardGoal(
        state,
        firstCounterGoal(state, (object) => object === void 0)
      );
    case "YIELD_PATH":
      return computeYieldPathAction(state) ?? "stay";
    case "WAIT":
      return "stay";
  }
}
function chooseAiDecision(state, weights2 = {}) {
  const geometry = computeLivePolicyGeometry(state);
  const contextOverrides = {
    humanIntent: geometry.humanIntent,
    candidatePathEffects: geometry.candidatePathEffects,
    candidateTargetOverlapsHuman: geometry.candidateTargetOverlapsHuman
  };
  const context = buildSubgoalContext(state, contextOverrides);
  const taskFeasible = enumerateFeasibleSubgoals(context);
  const candidateActions = {};
  for (const subgoal of taskFeasible) {
    candidateActions[subgoal] = subgoal === "YIELD_PATH" ? geometry.yieldAction ?? "stay" : actionForAiSubgoal(state, subgoal);
  }
  const motionUnexecutableSubgoals = taskFeasible.filter(
    (subgoal) => subgoal !== "WAIT" && candidateActions[subgoal] === "stay"
  );
  let feasible = taskFeasible.filter(
    (subgoal) => subgoal === "WAIT" || candidateActions[subgoal] !== "stay"
  );
  if (feasible.length === 0) feasible = ["WAIT"];
  const livenessRemoved = new Set(motionUnexecutableSubgoals);
  const blockedItem = state.aiPolicyLiveness?.cycleBlockedStashItem ?? null;
  const requiredPutSubgoal = blockedItem === "tomato" ? "PUT_TOMATO_IN_POT" : blockedItem === "onion" ? "PUT_ONION_IN_POT" : null;
  const getStashCycleGuardApplied = Boolean(
    blockedItem !== null && state.partner.held === blockedItem && requiredPutSubgoal !== null && feasible.includes("STASH_HELD_OBJECT") && feasible.includes(requiredPutSubgoal) && candidateActions[requiredPutSubgoal] !== "stay"
  );
  if (getStashCycleGuardApplied) {
    feasible = feasible.filter(
      (subgoal) => subgoal !== "STASH_HELD_OBJECT" && subgoal !== "WAIT"
    );
    livenessRemoved.add("STASH_HELD_OBJECT");
    livenessRemoved.add("WAIT");
  }
  const executableProductive = feasible.filter(
    (subgoal) => subgoal !== "WAIT" && candidateActions[subgoal] !== "stay"
  );
  const stalledWaitThresholdReached = (state.aiPolicyLiveness?.consecutiveWaitSteps ?? 0) >= MAX_CONSECUTIVE_WAIT_STEPS;
  const passiveCookingWait = passiveCookingWaitIsValid(context) && geometry.candidatePathEffects.WAIT !== "blocks";
  const waitGuardApplied = Boolean(
    !getStashCycleGuardApplied && stalledWaitThresholdReached && executableProductive.length > 0 && !passiveCookingWait && feasible.includes("WAIT")
  );
  if (waitGuardApplied) {
    feasible = feasible.filter((subgoal) => subgoal !== "WAIT");
    livenessRemoved.add("WAIT");
  }
  const rewardDecision = chooseRewardSubgoal(state, weights2, {
    contextOverrides,
    feasibleSubgoals: feasible
  });
  const action = state.status === "running" ? candidateActions[rewardDecision.chosenSubgoal] ?? actionForAiSubgoal(state, rewardDecision.chosenSubgoal) : "stay";
  const livenessReason = getStashCycleGuardApplied ? "get_stash_cycle_infeasible" : waitGuardApplied ? "stalled_wait_infeasible" : passiveCookingWait && stalledWaitThresholdReached ? "passive_cooking_wait_exempt" : executableProductive.length === 0 ? "no_executable_productive_alternative" : "stall_threshold_not_reached";
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
    getStashCycleBlockedItem: blockedItem
  };
}
function itemLabel(item) {
  switch (item) {
    case "tomato":
      return "tomato";
    case "onion":
      return "onion";
    case "dish":
      return "dish";
    case "soup":
      return "tomato-onion soup";
    default:
      return "empty-handed";
  }
}

// web/lib/route1-grounding.ts
var activeFeatures = new Set(POLICY_ACTIVE_FEATURES);
var rewardFeatures = new Set(REWARD_FEATURES);
function isUnresolvedQuestion(text, label) {
  if (label === "action" && /^(?:please\s+)?(?:can|could|would|will) you\s+(?:please\s+)?(?:get|grab|fetch|take|pick|collect|bring|add|put|place|serve|deliver|scoop|plate|ladle|move|go|walk|head|wait|stay|yield|stop|avoid|stash)\b/u.test(text)) return false;
  return /\?/u.test(text) || /^(?:who|what|when|where|why|how|is|are|was|were|am|does|did|can|could|would|will|should|shall|may|might|have|has|had)\b/u.test(text) || /^do\s+(?!not\b)/u.test(text);
}
function trajectoryApplicability(text, features) {
  const plain = text.replace(/[.!]+$/u, "").trim();
  const appraisal = "(?:good|great|nice|excellent|awesome|fantastic|terrific|perfect|bad|poor|terrible|awful|wrong|correct|helpful|unhelpful)";
  const modifier = "(?:(?:really|very|so|quite|not)\\s+)*";
  const generic = new RegExp(`^(?:${modifier}${appraisal}(?: (?:job|work|move|choice|decision|play|teamwork))?|well done|(?:you|we) did (?:well|great|badly)|(?:that|this|it) (?:was|is) ${modifier}${appraisal}|(?:that|this) (?:move|choice|decision|play) (?:was|is) ${modifier}${appraisal})$`, "u");
  if (generic.test(plain)) return null;
  const kitchenReference = /\b(?:onions?|tomato(?:es)?|soups?|dishes|dish|plates?|bowls?|pots?|recipe|serv(?:e|ed|ing)|pickup|pick(?:ed|ing)?|cook(?:ed|ing)?|block(?:ed|ing)?|paths?|routes?|waiting|teamwork)\b/u.test(text);
  const appraisalWord = new RegExp(`\\b(?:${appraisal}|well|better|worse|efficient|wasteful|slow|useful|quick|fast|mistake)\\b`, "u");
  const pastBehavior = /\b(?:you|we|the ai|the partner|my partner)\s+(?:(?:just|already|really|actually|have|had)\s+)*(?:picked|grabbed|fetched|took|got|added|put|placed|served|delivered|scooped|collected|waited|moved|blocked|cleared|wasted|helped|dropped|stashed)\b/u.test(text);
  const taskThanks = /^(?:thanks|thank you) for\b/u.test(text);
  if (!kitchenReference || !appraisalWord.test(text) && !pastBehavior && !taskThanks) {
    return "unresolved_trajectory_reference";
  }
  const objects = [
    [/\bonions?\b/u, ["ingredient_onion", "pick_onion", "adds_needed_onion", "adds_extra_onion"]],
    [/\btomato(?:es)?\b/u, ["ingredient_tomato", "pick_tomato", "adds_needed_tomato", "adds_extra_tomato"]],
    [/\b(?:dish|dishes|plates?|bowls?)\b/u, ["pick_dish", "dish_needed_for_ready_soup"]],
    [/\bsoups?\b/u, ["pick_ready_soup", "serve_ready_soup", "soup_ready", "supports_serving"]]
  ];
  for (const [mention, matching] of objects) {
    if (mention.test(text) && !matching.some((feature) => (features[feature] ?? 0) > 0)) return "trajectory_reference_not_observed";
  }
  return null;
}
function knownPositiveFeatures(source) {
  return Object.fromEntries(Object.entries(source).filter(
    ([feature, value]) => rewardFeatures.has(feature) && Number.isFinite(value) && value > 0
  ));
}
function actionReference(text, state) {
  const prohibitions = text.match(/\b(?:do not|don't|dont|never|avoid|stop|must not|should not|shouldn't)\b/gu) ?? [];
  const prohibit = prohibitions.length > 0;
  const polarity = prohibit ? "prohibit" : "affirm";
  if (prohibitions.length > 1 || /\b(?:not not|don't not|do not not)\b/u.test(text)) {
    return { subgoals: [], polarity, reason: "unsupported_action_negation" };
  }
  if (/\b(?:and|then|instead|rather|except|unless|or)\b/u.test(text)) {
    return { subgoals: [], polarity, reason: "ambiguous_action_reference" };
  }
  if (/\b(?:not|n't)\b/u.test(text) && !prohibit) {
    return { subgoals: [], polarity, reason: "unsupported_action_negation" };
  }
  const onion = /\bonions?\b/u.test(text);
  const tomato = /\btomato(?:es)?\b/u.test(text);
  const dish = /\b(?:dishes|dish|plates?|bowls?)\b/u.test(text);
  const soup = /\bsoups?\b/u.test(text);
  if ([onion, tomato, dish, soup].filter(Boolean).length > 1 && !(dish && soup && !onion && !tomato)) {
    return { subgoals: [], polarity, reason: "ambiguous_action_reference" };
  }
  const subgoals = [];
  const put = /\b(?:add(?:ing)?|put(?:ting)?|drop(?:ping)?|place|placing|load(?:ing)?|insert(?:ing)?|toss(?:ing)?)\b/u.test(text);
  const acquire = /\b(?:get(?:ting)?|grab(?:bing)?|fetch(?:ing)?|take|taking|pick(?:ing)?(?: up)?|collect(?:ing)?|bring(?:ing)?)\b/u.test(text);
  const approach = /\b(?:go|going|move|moving|head(?:ing)?|walk(?:ing)?)\b/u.test(text);
  if (put && /\b(?:onto|into|on|in|at)\b/u.test(text) && !/\b(?:pot|stove|counter)\b/u.test(text)) {
    return { subgoals: [], polarity, reason: "unresolved_action_reference" };
  }
  if (/\b(?:yield(?:ing)?|step(?:ping)? aside|move out of (?:my|the) way|clear(?:ing)? (?:my|the) path)\b/u.test(text)) {
    subgoals.push("YIELD_PATH");
  } else if (/\b(?:wait(?:ing)?|stay(?:ing)? (?:still|there|put)|hold still|pause)\b/u.test(text)) {
    subgoals.push("WAIT");
  } else if (/\b(?:stash(?:ing)?|set(?:ting)? down)\b/u.test(text) || put && /\bcounter\b/u.test(text)) {
    subgoals.push("STASH_HELD_OBJECT");
  } else if (/\b(?:serve|serving|deliver(?:ing)?|hand in)\b/u.test(text)) {
    subgoals.push("SERVE_SOUP");
  } else if (/\b(?:scoop(?:ing)?|plate|plating|ladle|ladling)\b/u.test(text) || soup && acquire) {
    subgoals.push("PICKUP_SOUP");
  } else if (put || approach && /\b(?:pot|stove)\b/u.test(text)) {
    const ingredient = onion ? "onion" : tomato ? "tomato" : state.partner.held;
    if (ingredient === "onion") subgoals.push("PUT_ONION_IN_POT");
    if (ingredient === "tomato") subgoals.push("PUT_TOMATO_IN_POT");
    if (ingredient === "dish" && approach) subgoals.push("PICKUP_SOUP");
  } else if (acquire || approach) {
    if (onion) subgoals.push("GET_ONION");
    if (tomato) subgoals.push("GET_TOMATO");
    if (dish) subgoals.push("GET_DISH");
  }
  return { subgoals, polarity };
}
function featureReference(text) {
  const features = {};
  if (/\bonions?\b/u.test(text)) features.ingredient_onion = 1;
  if (/\btomato(?:es)?\b/u.test(text)) features.ingredient_tomato = 1;
  if (/\b(?:block(?:s|ed|ing)?|obstruct(?:s|ed|ing)?)\b.*\b(?:path|route|way)\b/u.test(text)) features.blocks_human_path = 1;
  if (/\b(?:clear|open|free)\b.*\b(?:path|route|way)\b/u.test(text)) features.clears_human_path = 1;
  if (/\b(?:duplicate|duplicating|same)\b.*\b(?:task|work|job)\b/u.test(text)) features.duplicate_human_task = 1;
  if (/\b(?:delay|delays|delaying|slow)\b.*\bserv(?:e|ing|ice)\b/u.test(text)) features.delays_serving = 1;
  if (/\b(?:help(?:s|ful)?|support(?:s)?)\b.*\bserv(?:e|ing|ice)\b/u.test(text)) features.supports_serving = 1;
  if (/\b(?:wast(?:e|es|ing)|too much)\b.*\btime\b/u.test(text)) features.time_cost = 1;
  if (/\b(?:dispenser|window|counter|station)\b/u.test(text) || /\b(?:pot|stove)\b.*\b(?:has|contains|empty|cooking|ready|full)\b/u.test(text) || /\b(?:on (?:the )?(?:left|right)|next to|beside|near|north|south)\b/u.test(text)) return {};
  return features;
}
function groundRoute1Feedback(input) {
  const { grounding, state } = input;
  const threshold = grounding.threshold ?? 0.55;
  const result = {
    status: "rejected",
    label: grounding.label,
    confidence: grounding.confidence,
    threshold,
    modelHash: grounding.modelHash,
    targetFeatures: {},
    pragmaticAlternatives: {},
    selectedSubgoal: null,
    directivePolarity: null,
    adaptation: "paper-full-feature-complement-v1"
  };
  const reject = (reason) => ({ ...result, reason });
  if (grounding.abstained || !Number.isFinite(threshold) || threshold < 0 || threshold > 1 || !Number.isFinite(grounding.confidence) || grounding.confidence < threshold || grounding.confidence > 1) return reject("grounding_low_confidence");
  const text = input.text.trim().toLowerCase().replace(/[’‘]/gu, "'");
  if (!text || /[^\p{Script=Latin}\p{Number}\p{Punctuation}\p{Separator}\p{Symbol}\s]/u.test(text)) return reject("unsupported_language");
  if (/\b(?:movie|film|book|novel|song|music|chat(?:ting)?|weather|website|internet|vacation|holiday|yesterday|tomorrow)\b/u.test(text)) return reject("out_of_domain_reference");
  if (isUnresolvedQuestion(text, grounding.label)) return reject("unresolved_question_reference");
  const decision2 = chooseAiDecision(state);
  if (grounding.label === "trajectory") {
    result.targetFeatures = knownPositiveFeatures(input.trajectoryFeatures ?? {});
    const applicabilityReason = trajectoryApplicability(text, result.targetFeatures);
    if (applicabilityReason) return reject(applicabilityReason);
  } else if (grounding.label === "feature") {
    if (!/\b(?:is|are|was|were|has|have|contains?|needs?|requires?|prefer|important|useful|helpful|good|bad|wrong|correct|blocks?|blocked|blocking|clears?|clearing|delays?|delaying|wastes?|wasting)\b/u.test(text)) return reject("unresolved_feature_reference");
    result.targetFeatures = featureReference(text);
  } else {
    const reference = actionReference(text, state);
    result.directivePolarity = reference.polarity;
    if (reference.reason) return reject(reference.reason);
    if (reference.subgoals.length !== 1) return reject("unresolved_action_reference");
    result.selectedSubgoal = reference.subgoals[0];
    const candidate = decision2.ranking.find((item) => item.subgoal === result.selectedSubgoal);
    if (!candidate) return reject("infeasible_action_reference");
    result.targetFeatures = knownPositiveFeatures(candidate.features);
  }
  if (!Object.keys(result.targetFeatures).length) return reject("empty_target_features");
  if (!Object.keys(result.targetFeatures).some((feature) => activeFeatures.has(feature))) return reject("no_policy_active_target");
  if (result.directivePolarity !== "prohibit") {
    for (const feature of REWARD_FEATURES) {
      if (!(feature in result.targetFeatures)) result.pragmaticAlternatives[feature] = 1;
    }
  }
  return { ...result, status: "grounded" };
}

// web/lib/pragmatic-route1.ts
function applyPragmaticRoute1Update(state, observation) {
  return calculatePragmaticUpdate(state, observation);
}
function calculatePragmaticUpdate(state, observation, alternatives, explicitValence) {
  if (!Number.isFinite(observation.sentiment) || Math.abs(observation.sentiment) > 1) {
    throw new Error("Route1 sentiment must be a finite compound score in [-1, 1]");
  }
  const effectiveValence = explicitValence ?? (observation.sentiment === 0 ? 15 : 30 * observation.sentiment);
  const literal = applyRoute1PaperUpdate(state, {
    ...observation,
    valence: effectiveValence
  });
  if (literal.status === "rejected") {
    return { ...literal, effectiveValence, literal, pragmatic: null };
  }
  const inverseFeatures = alternatives ?? Object.fromEntries(
    state.features.filter((feature) => !(feature in literal.targetFeatures)).map((feature) => [feature, 1])
  );
  if (Object.keys(inverseFeatures).length === 0) {
    return { ...literal, effectiveValence, literal, pragmatic: null };
  }
  const pragmatic = applyRoute1PaperUpdate(literal.state, {
    feedbackForm: "descriptive",
    feedbackFormConfidence: observation.feedbackFormConfidence,
    feedbackFormThreshold: observation.feedbackFormThreshold,
    feedbackFormAbstained: observation.feedbackFormAbstained,
    namedFeatures: inverseFeatures,
    valence: -30,
    basePrecision: observation.basePrecision
  });
  return {
    ...literal,
    state: pragmatic.state,
    delta: Object.fromEntries(
      state.features.map((feature, index) => [feature, pragmatic.state.mean[index] - state.mean[index]]).filter(([, value]) => Math.abs(value) > 1e-12)
    ),
    effectiveValence,
    literal,
    pragmatic
  };
}
function applyGroundedPragmaticFeedback(state, observation) {
  const { grounding, sentiment } = observation;
  if (observation.basePrecision !== void 0 && (!Number.isFinite(observation.basePrecision) || observation.basePrecision <= 0)) {
    throw new Error("Route1 observation precision must be finite and positive");
  }
  const feedbackForm = grounding.label === "action" ? "imperative" : grounding.label === "trajectory" ? "evaluative" : "descriptive";
  const prohibit = grounding.directivePolarity === "prohibit";
  const result = calculatePragmaticUpdate(state, {
    feedbackForm,
    feedbackFormConfidence: grounding.confidence,
    feedbackFormThreshold: grounding.threshold,
    feedbackFormAbstained: grounding.status === "rejected",
    trajectoryFeatures: grounding.targetFeatures,
    actionFeatures: grounding.targetFeatures,
    namedFeatures: grounding.targetFeatures,
    sentiment,
    basePrecision: observation.basePrecision
  }, grounding.pragmaticAlternatives, prohibit ? -30 : void 0);
  return {
    ...result,
    ...grounding.status === "rejected" ? { reason: grounding.reason } : {},
    grounding,
    rawSentiment: sentiment,
    valenceSource: prohibit ? "explicit_prohibition" : sentiment === 0 ? "paper_neutral_prior" : "vader_compound"
  };
}

// web/lib/vendor/vader-data.json
var vader_data_default = { source: "vaderSentiment 3.3.2", source_sha256: "25cd814d23000b41c0e242d99960832c6620b3f1b844469173404500ce63206b", lexicon: { "$:": -1.5, "%)": -0.4, "%-)": -1.5, "&-:": -0.4, "&:": -0.7, "( '}{' )": 1.6, "(%": -0.9, "('-:": 2.2, "(':": 2.3, "((-:": 2.1, "(*": 1.1, "(-%": -0.7, "(-*": 1.3, "(-:": 1.6, "(-:0": 2.8, "(-:<": -0.4, "(-:o": 1.5, "(-:O": 1.5, "(-:{": -0.1, "(-:|>*": 1.9, "(-;": 1.3, "(-;|": 2.1, "(8": 2.6, "(:": 2.2, "(:0": 2.4, "(:<": -0.2, "(:o": 2.5, "(:O": 2.5, "(;": 1.1, "(;<": 0.3, "(=": 2.2, "(?:": 2.1, "(^:": 1.5, "(^;": 1.5, "(^;0": 2, "(^;o": 1.9, "(o:": 1.6, ")':": -2, ")-':": -2.1, ")-:": -2.1, ")-:<": -2.2, ")-:{": -2.1, "):": -1.8, "):<": -1.9, "):{": -2.3, ");<": -2.6, "*)": 0.6, "*-)": 0.3, "*-:": 2.1, "*-;": 2.4, "*:": 1.9, "*<|:-)": 1.6, "*\\0/*": 2.3, "*^:": 1.6, ",-:": 1.2, "---'-;-{@": 2.3, "--<--<@": 2.2, ".-:": -1.2, "..###-:": -1.7, "..###:": -1.9, "/-:": -1.3, "/:": -1.3, "/:<": -1.4, "/=": -0.9, "/^:": -1, "/o:": -1.4, "0-8": 0.1, "0-|": -1.2, "0:)": 1.9, "0:-)": 1.4, "0:-3": 1.5, "0:03": 1.9, "0;^)": 1.6, "0_o": -0.3, "10q": 2.1, "1337": 2.1, "143": 3.2, "1432": 2.6, "14aa41": 2.4, "182": -2.9, "187": -3.1, "2g2b4g": 2.8, "2g2bt": -0.1, "2qt": 2.1, "3:(": -2.2, "3:)": 0.5, "3:-(": -2.3, "3:-)": -1.4, "4col": -2.2, "4q": -3.1, "5fs": 1.5, "8)": 1.9, "8-d": 1.7, "8-o": -0.3, "86": -1.6, "8d": 2.9, ":###..": -2.4, ":$": -0.2, ":&": -0.6, ":'(": -2.2, ":')": 2.3, ":'-(": -2.4, ":'-)": 2.7, ":(": -1.9, ":)": 2, ":*": 2.5, ":-###..": -2.5, ":-&": -0.5, ":-(": -1.5, ":-)": 1.3, ":-))": 2.8, ":-*": 1.7, ":-,": 1.1, ":-.": -0.9, ":-/": -1.2, ":-<": -1.5, ":-d": 2.3, ":-D": 2.3, ":-o": 0.1, ":-p": 1.5, ":-[": -1.6, ":-\\": -0.9, ":-c": -1.3, ":-|": -0.7, ":-||": -2.5, ":-\xDE": 0.9, ":/": -1.4, ":3": 2.3, ":<": -2.1, ":>": 2.1, ":?)": 1.3, ":?c": -1.6, ":@": -2.5, ":d": 2.3, ":D": 2.3, ":l": -1.7, ":o": -0.4, ":p": 1, ":s": -1.2, ":[": -2, ":\\": -1.3, ":]": 2.2, ":^)": 2.1, ":^*": 2.6, ":^/": -1.2, ":^\\": -1, ":^|": -1, ":c": -2.1, ":c)": 2, ":o)": 2.1, ":o/": -1.4, ":o\\": -1.1, ":o|": -0.6, ":P": 1.4, ":{": -1.9, ":|": -0.4, ":}": 2.1, ":\xDE": 1.1, ";)": 0.9, ";-)": 1, ";-*": 2.2, ";-]": 0.7, ";d": 0.8, ";D": 0.8, ";]": 0.6, ";^)": 1.4, "</3": -3, "<3": 1.9, "<:": 2.1, "<:-|": -1.4, "=)": 2.2, "=-3": 2, "=-d": 2.4, "=-D": 2.4, "=/": -1.4, "=3": 2.1, "=d": 2.3, "=D": 2.3, "=l": -1.2, "=\\": -1.2, "=]": 1.6, "=p": 1.3, "=|": -0.8, ">-:": -2, ">.<": -1.3, ">:": -2.1, ">:(": -2.7, ">:)": 0.4, ">:-(": -2.7, ">:-)": -0.4, ">:/": -1.6, ">:o": -1.2, ">:p": 1, ">:[": -2.1, ">:\\": -1.7, ">;(": -2.9, ">;)": 0.1, ">_>^": 2.1, "@:": -2.1, "@>-->--": 2.1, "@}-;-'---": 2.2, aas: 2.5, aayf: 2.7, afu: -2.9, alol: 2.8, ambw: 2.9, aml: 3.4, atab: -1.9, awol: -1.3, ayc: 0.2, ayor: -1.2, "aug-00": 0.3, bfd: -2.7, bfe: -2.6, bff: 2.9, bffn: 1, bl: 2.3, bsod: -2.2, btd: -2.1, btdt: -0.1, bz: 0.4, "b^d": 2.6, cwot: -2.3, "d-':": -2.5, d8: -3.2, "d:": 1.2, "d:<": -3.2, "d;": -2.9, "d=": 1.5, doa: -2.3, dx: -3, ez: 1.5, fav: 2, fcol: -1.8, ff: 1.8, ffs: -2.8, fkm: -2.4, foaf: 1.8, ftw: 2, fu: -3.7, fubar: -3, fwb: 2.5, fyi: 0.8, fysa: 0.4, g1: 1.4, gg: 1.2, gga: 1.7, gigo: -0.6, gj: 2, gl: 1.3, gla: 2.5, gn: 1.2, gr8: 2.7, grrr: -0.4, gt: 1.1, "h&k": 2.3, hagd: 2.2, hagn: 2.2, hago: 1.2, hak: 1.9, hand: 2.2, heart: 3.2, hearts: 3.3, "hho1/2k": 1.4, hhoj: 2, hhok: 0.9, hugz: 2, hi5: 1.9, idk: -0.4, ijs: 0.7, ilu: 3.4, iluaaf: 2.7, ily: 3.4, ily2: 2.6, iou: 0.7, iyq: 2.3, "j/j": 2, "j/k": 1.6, "j/p": 1.4, "j/t": -0.2, "j/w": 1, j4f: 1.4, j4g: 1.7, jho: 0.8, jhomf: 1, jj: 1, jk: 0.9, jp: 0.8, jt: 0.9, jw: 1.6, jealz: -1.2, k4y: 2.3, kfy: 2.3, kia: -3.2, kk: 1.5, kmuf: 2.2, l: 2, "l&r": 2.2, laoj: 1.3, lmao: 2.9, lmbao: 1.8, lmfao: 2.5, lmso: 2.7, lol: 1.8, lolz: 2.7, lts: 1.6, ly: 2.6, ly4e: 2.7, lya: 3.3, lyb: 3, lyl: 3.1, lylab: 2.7, lylas: 2.6, lylb: 1.6, m8: 1.4, mia: -1.2, mml: 2, mofo: -2.4, muah: 2.3, mubar: -1, musm: 0.9, mwah: 2.5, n1: 1.9, nbd: 1.3, nbif: -0.5, nfc: -2.7, nfw: -2.4, nh: 2.2, nimby: -0.8, nimjd: -0.7, nimq: -0.2, nimy: -1.4, nitl: -1.5, nme: -2.1, noyb: -0.7, np: 1.4, ntmu: 1.4, "o-8": -0.5, "o-:": -0.3, "o-|": -1.1, "o.o": -0.8, "O.o": -0.6, "o.O": -0.6, "o:": -0.2, "o:)": 1.5, "o:-)": 2, "o:-3": 2.2, "o:3": 2.3, "o:<": -0.3, "o;^)": 1.6, ok: 1.2, o_o: -0.5, O_o: -0.5, o_O: -0.5, pita: -2.4, pls: 0.3, plz: 0.3, pmbi: 0.8, pmfji: 0.3, pmji: 0.7, po: -2.6, ptl: 2.6, pu: -1.1, qq: -2.2, qt: 1.8, "r&r": 2.4, rofl: 2.7, roflmao: 2.5, rotfl: 2.6, rotflmao: 2.8, rotflmfao: 2.5, rotflol: 3, rotgl: 2.9, rotglmao: 1.8, "s:": -1.1, sapfu: -1.1, sete: 2.8, sfete: 2.7, sgtm: 2.4, slap: 0.6, slaw: 2.1, smh: -1.3, snafu: -2.5, sob: -1, swak: 2.3, tgif: 2.3, thks: 1.4, thx: 1.5, tia: 2.3, tmi: -0.3, tnx: 1.1, true: 1.8, tx: 1.5, txs: 1.1, ty: 1.6, tyvm: 2.5, urw: 1.9, vbg: 2.1, vbs: 3.1, vip: 2.3, vwd: 2.6, vwp: 2.1, wag: -0.2, wd: 2.7, wilco: 0.9, wp: 1, wtf: -2.8, wtg: 2.1, wth: -2.4, "x-d": 2.6, "x-p": 1.7, xd: 2.8, xlnt: 3, xoxo: 3, xoxozzz: 2.3, xp: 1.6, xqzt: 1.6, xtc: 0.8, yolo: 1.1, yoyo: 0.4, yvw: 1.6, yw: 1.8, ywia: 2.5, zzz: -1.2, "[-;": 0.5, "[:": 1.3, "[;": 1, "[=": 1.7, "\\-:": -1, "\\:": -1, "\\:<": -1.7, "\\=": -1.1, "\\^:": -1.3, "\\o/": 2.2, "\\o:": -1.2, "]-:": -2.1, "]:": -1.6, "]:<": -2.5, "^<_<": 1.4, "^urs": -2.8, abandon: -1.9, abandoned: -2, abandoner: -1.9, abandoners: -1.9, abandoning: -1.6, abandonment: -2.4, abandonments: -1.7, abandons: -1.3, abducted: -2.3, abduction: -2.8, abductions: -2, abhor: -2, abhorred: -2.4, abhorrent: -3.1, abhors: -2.9, abilities: 1, ability: 1.3, aboard: 0.1, absentee: -1.1, absentees: -0.8, absolve: 1.2, absolved: 1.5, absolves: 1.3, absolving: 1.6, abuse: -3.2, abused: -2.3, abuser: -2.6, abusers: -2.6, abuses: -2.6, abusing: -2, abusive: -3.2, abusively: -2.8, abusiveness: -2.5, abusivenesses: -3, accept: 1.6, acceptabilities: 1.6, acceptability: 1.1, acceptable: 1.3, acceptableness: 1.3, acceptably: 1.5, acceptance: 2, acceptances: 1.7, acceptant: 1.6, acceptation: 1.3, acceptations: 0.9, accepted: 1.1, accepting: 1.6, accepts: 1.3, accident: -2.1, accidental: -0.3, accidentally: -1.4, accidents: -1.3, accomplish: 1.8, accomplished: 1.9, accomplishes: 1.7, accusation: -1, accusations: -1.3, accuse: -0.8, accused: -1.2, accuses: -1.4, accusing: -0.7, ache: -1.6, ached: -1.6, aches: -1, achievable: 1.3, aching: -2.2, acquit: 0.8, acquits: 0.1, acquitted: 1, acquitting: 1.3, acrimonious: -1.7, active: 1.7, actively: 1.3, activeness: 0.6, activenesses: 0.8, actives: 1.1, adequate: 0.9, admirability: 2.4, admirable: 2.6, admirableness: 2.2, admirably: 2.5, admiral: 1.3, admirals: 1.5, admiralties: 1.6, admiralty: 1.2, admiration: 2.5, admirations: 1.6, admire: 2.1, admired: 2.3, admirer: 1.8, admirers: 1.7, admires: 1.5, admiring: 1.6, admiringly: 2.3, admit: 0.8, admits: 1.2, admitted: 0.4, admonished: -1.9, adopt: 0.7, adopts: 0.7, adorability: 2.2, adorable: 2.2, adorableness: 2.5, adorably: 2.1, adoration: 2.9, adorations: 2.2, adore: 2.6, adored: 1.8, adorer: 1.7, adorers: 2.1, adores: 1.6, adoring: 2.6, adoringly: 2.4, adorn: 0.9, adorned: 0.8, adorner: 1.3, adorners: 0.9, adorning: 1, adornment: 1.3, adornments: 0.8, adorns: 0.5, advanced: 1, advantage: 1, advantaged: 1.4, advantageous: 1.5, advantageously: 1.9, advantageousness: 1.6, advantages: 1.5, advantaging: 1.6, adventure: 1.3, adventured: 1.3, adventurer: 1.2, adventurers: 0.9, adventures: 1.4, adventuresome: 1.7, adventuresomeness: 1.3, adventuress: 0.8, adventuresses: 1.4, adventuring: 2.3, adventurism: 1.5, adventurist: 1.4, adventuristic: 1.7, adventurists: 1.2, adventurous: 1.4, adventurously: 1.3, adventurousness: 1.8, adversarial: -1.5, adversaries: -1, adversary: -0.8, adversative: -1.2, adversatively: -0.1, adversatives: -1, adverse: -1.5, adversely: -0.8, adverseness: -0.6, adversities: -1.5, adversity: -1.8, affected: -0.6, affection: 2.4, affectional: 1.9, affectionally: 1.5, affectionate: 1.9, affectionately: 2.2, affectioned: 1.8, affectionless: -2, affections: 1.5, afflicted: -1.5, affronted: 0.2, aggravate: -2.5, aggravated: -1.9, aggravates: -1.9, aggravating: -1.2, aggress: -1.3, aggressed: -1.4, aggresses: -0.5, aggressing: -0.6, aggression: -1.2, aggressions: -1.3, aggressive: -0.6, aggressively: -1.3, aggressiveness: -1.8, aggressivities: -1.4, aggressivity: -0.6, aggressor: -0.8, aggressors: -0.9, aghast: -1.9, agitate: -1.7, agitated: -2, agitatedly: -1.6, agitates: -1.4, agitating: -1.8, agitation: -1, agitational: -1.2, agitations: -1.3, agitative: -1.3, agitato: -0.1, agitator: -1.4, agitators: -2.1, agog: 1.9, agonise: -2.1, agonised: -2.3, agonises: -2.4, agonising: -1.5, agonize: -2.3, agonized: -2.2, agonizes: -2.3, agonizing: -2.7, agonizingly: -2.3, agony: -1.8, agree: 1.5, agreeability: 1.9, agreeable: 1.8, agreeableness: 1.8, agreeablenesses: 1.3, agreeably: 1.6, agreed: 1.1, agreeing: 1.4, agreement: 2.2, agreements: 1.1, agrees: 0.8, alarm: -1.4, alarmed: -1.4, alarming: -0.5, alarmingly: -2.6, alarmism: -0.3, alarmists: -1.1, alarms: -1.1, alas: -1.1, alert: 1.2, alienation: -1.1, alive: 1.6, allergic: -1.2, allow: 0.9, alone: -1, alright: 1, amaze: 2.5, amazed: 2.2, amazedly: 2.1, amazement: 2.5, amazements: 2.2, amazes: 2.2, amazing: 2.8, amazon: 0.7, amazonite: 0.2, amazons: -0.1, amazonstone: 1, amazonstones: 0.2, ambitious: 2.1, ambivalent: 0.5, amor: 3, amoral: -1.6, amoralism: -0.7, amoralisms: -0.7, amoralities: -1.2, amorality: -1.5, amorally: -1, amoretti: 0.2, amoretto: 0.6, amorettos: 0.3, amorino: 1.2, amorist: 1.6, amoristic: 1, amorists: 0.1, amoroso: 2.3, amorous: 1.8, amorously: 2.3, amorousness: 2, amorphous: -0.2, amorphously: 0.1, amorphousness: 0.3, amort: -2.1, amortise: 0.5, amortised: -0.2, amortises: 0.1, amortizable: 0.5, amortization: 0.6, amortizations: 0.2, amortize: -0.1, amortized: 0.8, amortizes: 0.6, amortizing: 0.8, amusable: 0.7, amuse: 1.7, amused: 1.8, amusedly: 2.2, amusement: 1.5, amusements: 1.5, amuser: 1.1, amusers: 1.3, amuses: 1.7, amusia: 0.3, amusias: -0.4, amusing: 1.6, amusingly: 0.8, amusingness: 1.8, amusive: 1.7, anger: -2.7, angered: -2.3, angering: -2.2, angerly: -1.9, angers: -2.3, angrier: -2.3, angriest: -3.1, angrily: -1.8, angriness: -1.7, angry: -2.3, anguish: -2.9, anguished: -1.8, anguishes: -2.1, anguishing: -2.7, animosity: -1.9, annoy: -1.9, annoyance: -1.3, annoyances: -1.8, annoyed: -1.6, annoyer: -2.2, annoyers: -1.5, annoying: -1.7, annoys: -1.8, antagonism: -1.9, antagonisms: -1.2, antagonist: -1.9, antagonistic: -1.7, antagonistically: -2.2, antagonists: -1.7, antagonize: -2, antagonized: -1.4, antagonizes: -0.5, antagonizing: -2.7, anti: -1.3, anticipation: 0.4, anxieties: -0.6, anxiety: -0.7, anxious: -1, anxiously: -0.9, anxiousness: -1, aok: 2, apathetic: -1.2, apathetically: -0.4, apathies: -0.6, apathy: -1.2, apeshit: -0.9, apocalyptic: -3.4, apologise: 1.6, apologised: 0.4, apologises: 0.8, apologising: 0.2, apologize: 0.4, apologized: 1.3, apologizes: 1.5, apologizing: -0.3, apology: 0.2, appall: -2.4, appalled: -2, appalling: -1.5, appallingly: -2, appalls: -1.9, appease: 1.1, appeased: 0.9, appeases: 0.9, appeasing: 1, applaud: 2, applauded: 1.5, applauding: 2.1, applauds: 1.4, applause: 1.8, appreciate: 1.7, appreciated: 2.3, appreciates: 2.3, appreciating: 1.9, appreciation: 2.3, appreciations: 1.7, appreciative: 2.6, appreciatively: 1.8, appreciativeness: 1.6, appreciator: 2.6, appreciators: 1.5, appreciatory: 1.7, apprehensible: 1.1, apprehensibly: -0.2, apprehension: -2.1, apprehensions: -0.9, apprehensively: -0.3, apprehensiveness: -0.7, approval: 2.1, approved: 1.8, approves: 1.7, ardent: 2.1, arguable: -1, arguably: -1, argue: -1.4, argued: -1.5, arguer: -1.6, arguers: -1.4, argues: -1.6, arguing: -2, argument: -1.5, argumentative: -1.5, argumentatively: -1.8, argumentive: -1.5, arguments: -1.7, arrest: -1.4, arrested: -2.1, arrests: -1.9, arrogance: -2.4, arrogances: -1.9, arrogant: -2.2, arrogantly: -1.8, ashamed: -2.1, ashamedly: -1.7, ass: -2.5, assassination: -2.9, assassinations: -2.7, assault: -2.8, assaulted: -2.4, assaulting: -2.3, assaultive: -2.8, assaults: -2.5, asset: 1.5, assets: 0.7, assfucking: -2.5, assholes: -2.8, assurance: 1.4, assurances: 1.4, assure: 1.4, assured: 1.5, assuredly: 1.6, assuredness: 1.4, assurer: 0.9, assurers: 1.1, assures: 1.3, assurgent: 1.3, assuring: 1.6, assuror: 0.5, assurors: 0.7, astonished: 1.6, astound: 1.7, astounded: 1.8, astounding: 1.8, astoundingly: 2.1, astounds: 2.1, attachment: 1.2, attachments: 1.1, attack: -2.1, attacked: -2, attacker: -2.7, attackers: -2.7, attacking: -2, attacks: -1.9, attract: 1.5, attractancy: 0.9, attractant: 1.3, attractants: 1.4, attracted: 1.8, attracting: 2.1, attraction: 2, attractions: 1.8, attractive: 1.9, attractively: 2.2, attractiveness: 1.8, attractivenesses: 2.1, attractor: 1.2, attractors: 1.2, attracts: 1.7, audacious: 0.9, authority: 0.3, aversion: -1.9, aversions: -1.1, aversive: -1.6, aversively: -0.8, avert: -0.7, averted: -0.3, averts: -0.4, avid: 1.2, avoid: -1.2, avoidance: -1.7, avoidances: -1.1, avoided: -1.4, avoider: -1.8, avoiders: -1.4, avoiding: -1.4, avoids: -0.7, await: 0.4, awaited: -0.1, awaits: 0.3, award: 2.5, awardable: 2.4, awarded: 1.7, awardee: 1.8, awardees: 1.2, awarder: 0.9, awarders: 1.3, awarding: 1.9, awards: 2, awesome: 3.1, awful: -2, awkward: -0.6, awkwardly: -1.3, awkwardness: -0.7, axe: -0.4, axed: -1.3, backed: 0.1, backing: 0.1, backs: -0.2, bad: -2.5, badass: 1.4, badly: -2.1, bailout: -0.4, bamboozle: -1.5, bamboozled: -1.5, bamboozles: -1.5, ban: -2.6, banish: -1.9, bankrupt: -2.6, bankster: -2.1, banned: -2, bargain: 0.8, barrier: -0.5, bashful: -0.1, bashfully: 0.2, bashfulness: -0.8, bastard: -2.5, bastardies: -1.8, bastardise: -2.1, bastardised: -2.3, bastardises: -2.3, bastardising: -2.6, bastardization: -2.4, bastardizations: -2.1, bastardize: -2.4, bastardized: -2, bastardizes: -1.8, bastardizing: -2.3, bastardly: -2.7, bastards: -3, bastardy: -2.7, battle: -1.6, battled: -1.2, battlefield: -1.6, battlefields: -0.9, battlefront: -1.2, battlefronts: -0.8, battleground: -1.7, battlegrounds: -0.6, battlement: -0.4, battlements: -0.4, battler: -0.8, battlers: -0.2, battles: -1.6, battleship: -0.1, battleships: -0.5, battlewagon: -0.3, battlewagons: -0.5, battling: -1.1, beaten: -1.8, beatific: 1.8, beating: -2, beaut: 1.6, beauteous: 2.5, beauteously: 2.6, beauteousness: 2.7, beautician: 1.2, beauticians: 0.4, beauties: 2.4, beautification: 1.9, beautifications: 2.4, beautified: 2.1, beautifier: 1.7, beautifiers: 1.7, beautifies: 1.8, beautiful: 2.9, beautifuler: 2.1, beautifulest: 2.6, beautifully: 2.7, beautifulness: 2.6, beautify: 2.3, beautifying: 2.3, beauts: 1.7, beauty: 2.8, belittle: -1.9, belittled: -2, beloved: 2.3, benefic: 1.4, benefice: 0.4, beneficed: 1.1, beneficence: 2.8, beneficences: 1.5, beneficent: 2.3, beneficently: 2.2, benefices: 1.1, beneficial: 1.9, beneficially: 2.4, beneficialness: 1.7, beneficiaries: 1.8, beneficiary: 2.1, beneficiate: 1, beneficiation: 0.4, benefit: 2, benefits: 1.6, benefitted: 1.7, benefitting: 1.9, benevolence: 1.7, benevolences: 1.9, benevolent: 2.7, benevolently: 1.4, benevolentness: 1.2, benign: 1.3, benignancy: 0.6, benignant: 2.2, benignantly: 1.1, benignities: 0.9, benignity: 1.3, benignly: 0.2, bereave: -2.1, bereaved: -2.1, bereaves: -1.9, bereaving: -1.3, best: 3.2, betray: -3.2, betrayal: -2.8, betrayed: -3, betraying: -2.5, betrays: -2.5, better: 1.9, bias: -0.4, biased: -1.1, bitch: -2.8, bitched: -2.6, bitcheries: -2.3, bitchery: -2.7, bitches: -2.9, bitchier: -2, bitchiest: -3, bitchily: -2.6, bitchiness: -2.6, bitching: -1.1, bitchy: -2.3, bitter: -1.8, bitterbrush: -0.2, bitterbrushes: -0.6, bittered: -1.8, bitterer: -1.9, bitterest: -2.3, bittering: -1.2, bitterish: -1.6, bitterly: -2, bittern: -0.2, bitterness: -1.7, bitterns: -0.4, bitterroots: -0.2, bitters: -0.4, bittersweet: -0.3, bittersweetness: -0.6, bittersweets: -0.2, bitterweeds: -0.5, bizarre: -1.3, blah: -0.4, blam: -0.2, blamable: -1.8, blamably: -1.8, blame: -1.4, blamed: -2.1, blameful: -1.7, blamefully: -1.6, blameless: 0.7, blamelessly: 0.9, blamelessness: 0.6, blamer: -2.1, blamers: -2, blames: -1.7, blameworthiness: -1.6, blameworthy: -2.3, blaming: -2.2, bless: 1.8, blessed: 2.9, blesseder: 2, blessedest: 2.8, blessedly: 1.7, blessedness: 1.6, blesser: 2.6, blessers: 1.9, blesses: 2.6, blessing: 2.2, blessings: 2.5, blind: -1.7, bliss: 2.7, blissful: 2.9, blithe: 1.2, block: -1.9, blockbuster: 2.9, blocked: -1.1, blocking: -1.6, blocks: -0.9, bloody: -1.9, blurry: -0.4, bold: 1.6, bolder: 1.2, boldest: 1.6, boldface: 0.3, boldfaced: -0.1, boldfaces: 0.1, boldfacing: 0.1, boldly: 1.5, boldness: 1.5, boldnesses: 0.9, bolds: 1.3, bomb: -2.2, bonus: 2.5, bonuses: 2.6, boost: 1.7, boosted: 1.5, boosting: 1.4, boosts: 1.3, bore: -1, boreal: -0.3, borecole: -0.2, borecoles: -0.3, bored: -1.1, boredom: -1.3, boredoms: -1.1, boreen: 0.1, boreens: 0.2, boreholes: -0.2, borer: -0.4, borers: -1.2, bores: -1.3, borescopes: -0.1, boresome: -1.3, boring: -1.3, bother: -1.4, botheration: -1.7, botherations: -1.3, bothered: -1.3, bothering: -1.6, bothers: -0.8, bothersome: -1.3, boycott: -1.3, boycotted: -1.7, boycotting: -1.7, boycotts: -1.4, brainwashing: -1.5, brave: 2.4, braved: 1.9, bravely: 2.3, braver: 2.4, braveries: 2, bravery: 2.2, braves: 1.9, bravest: 2.3, breathtaking: 2, bribe: -0.8, bright: 1.9, brighten: 1.9, brightened: 2.1, brightener: 1, brighteners: 1, brightening: 2.5, brightens: 1.5, brighter: 1.6, brightest: 3, brightly: 1.5, brightness: 1.6, brightnesses: 1.4, brights: 0.4, brightwork: 1.1, brilliance: 2.9, brilliances: 2.9, brilliancies: 2.3, brilliancy: 2.6, brilliant: 2.8, brilliantine: 0.8, brilliantines: 2, brilliantly: 3, brilliants: 1.9, brisk: 0.6, broke: -1.8, broken: -2.1, brooding: 0.1, brutal: -3.1, brutalise: -2.7, brutalised: -2.9, brutalises: -3.2, brutalising: -2.8, brutalities: -2.6, brutality: -3, brutalization: -2.1, brutalizations: -2.3, brutalize: -2.9, brutalized: -2.4, brutalizes: -3.2, brutalizing: -3.4, brutally: -3, bullied: -3.1, bullshit: -2.8, bully: -2.2, bullying: -2.9, bummer: -1.6, buoyant: 0.9, burden: -1.9, burdened: -1.7, burdener: -1.3, burdeners: -1.7, burdening: -1.4, burdens: -1.5, burdensome: -1.8, bwahaha: 0.4, bwahahah: 2.5, calm: 1.3, calmative: 1.1, calmatives: 0.5, calmed: 1.6, calmer: 1.5, calmest: 1.6, calming: 1.7, calmly: 1.3, calmness: 1.7, calmnesses: 1.6, calmodulin: 0.2, calms: 1.3, "can't stand": -2, cancel: -1, cancelled: -1, cancelling: -0.8, cancels: -0.9, cancer: -3.4, capable: 1.6, captivated: 1.6, care: 2.2, cared: 1.8, carefree: 1.7, careful: 0.6, carefully: 0.5, carefulness: 2, careless: -1.5, carelessly: -1, carelessness: -1.4, carelessnesses: -1.6, cares: 2, caring: 2.2, casual: 0.8, casually: 0.7, casualty: -2.4, catastrophe: -3.4, catastrophic: -2.2, cautious: -0.4, celebrate: 2.7, celebrated: 2.7, celebrates: 2.7, celebrating: 2.7, censor: -2, censored: -0.6, censors: -1.2, certain: 1.1, certainly: 1.4, certainties: 0.9, certainty: 1, chagrin: -1.9, chagrined: -1.4, challenge: 0.3, challenged: -0.4, challenger: 0.5, challengers: 0.4, challenges: 0.3, challenging: 0.6, challengingly: -0.6, champ: 2.1, champac: -0.2, champagne: 1.2, champagnes: 0.5, champaign: 0.2, champaigns: 0.5, champaks: -0.2, champed: 1, champer: -0.1, champers: 0.5, champerties: -0.1, champertous: 0.3, champerty: -0.2, champignon: 0.4, champignons: 0.2, champing: 0.7, champion: 2.9, championed: 1.2, championing: 1.8, champions: 2.4, championship: 1.9, championships: 2.2, champs: 1.8, champy: 1, chance: 1, chances: 0.8, chaos: -2.7, chaotic: -2.2, charged: -0.8, charges: -1.1, charitable: 1.7, charitableness: 1.9, charitablenesses: 1.6, charitably: 1.4, charities: 2.2, charity: 1.8, charm: 1.7, charmed: 2, charmer: 1.9, charmers: 2.1, charmeuse: 0.3, charmeuses: 0.4, charming: 2.8, charminger: 1.5, charmingest: 2.4, charmingly: 2.2, charmless: -1.8, charms: 1.9, chastise: -2.5, chastised: -2.2, chastises: -1.7, chastising: -1.7, cheat: -2, cheated: -2.3, cheater: -2.5, cheaters: -1.9, cheating: -2.6, cheats: -1.8, cheer: 2.3, cheered: 2.3, cheerer: 1.7, cheerers: 1.8, cheerful: 2.5, cheerfuller: 1.9, cheerfullest: 3.2, cheerfully: 2.1, cheerfulness: 2.1, cheerier: 2.6, cheeriest: 2.2, cheerily: 2.5, cheeriness: 2.5, cheering: 2.3, cheerio: 1.2, cheerlead: 1.7, cheerleader: 0.9, cheerleaders: 1.2, cheerleading: 1.2, cheerleads: 1.2, cheerled: 1.5, cheerless: -1.7, cheerlessly: -0.8, cheerlessness: -1.7, cheerly: 2.4, cheers: 2.1, cheery: 2.6, cherish: 1.6, cherishable: 2, cherished: 2.3, cherisher: 2.2, cherishers: 1.9, cherishes: 2.2, cherishing: 2, chic: 1.1, childish: -1.2, chilling: -0.1, choke: -2.5, choked: -2.1, chokes: -2, choking: -2, chuckle: 1.7, chuckled: 1.2, chucklehead: -1.9, chuckleheaded: -1.3, chuckleheads: -1.1, chuckler: 0.8, chucklers: 1.2, chuckles: 1.1, chucklesome: 1.1, chuckling: 1.4, chucklingly: 1.2, clarifies: 0.9, clarity: 1.7, classy: 1.9, clean: 1.7, cleaner: 0.7, clear: 1.6, cleared: 0.4, clearly: 1.7, clears: 0.3, clever: 2, cleverer: 2, cleverest: 2.6, cleverish: 1, cleverly: 2.3, cleverness: 2.3, clevernesses: 1.4, clouded: -0.2, clueless: -1.5, cock: -0.6, cocksucker: -3.1, cocksuckers: -2.6, cocky: -0.5, coerced: -1.5, collapse: -2.2, collapsed: -1.1, collapses: -1.2, collapsing: -1.2, collide: -0.3, collides: -1.1, colliding: -0.5, collision: -1.5, collisions: -1.1, colluding: -1.2, combat: -1.4, combats: -0.8, comedian: 1.6, comedians: 1.2, comedic: 1.7, comedically: 2.1, comedienne: 0.6, comediennes: 1.6, comedies: 1.7, comedo: 0.3, comedones: -0.8, comedown: -0.8, comedowns: -0.9, comedy: 1.5, comfort: 1.5, comfortable: 2.3, comfortableness: 1.3, comfortably: 1.8, comforted: 1.8, comforter: 1.9, comforters: 1.2, comforting: 1.7, comfortingly: 1.7, comfortless: -1.8, comforts: 2.1, commend: 1.9, commended: 1.9, commit: 1.2, commitment: 1.6, commitments: 0.5, commits: 0.1, committed: 1.1, committing: 0.3, compassion: 2, compassionate: 2.2, compassionated: 1.6, compassionately: 1.7, compassionateness: 0.9, compassionates: 1.6, compassionating: 1.6, compassionless: -2.6, compelled: 0.2, compelling: 0.9, competent: 1.3, competitive: 0.7, complacent: -0.3, complain: -1.5, complainant: -0.7, complainants: -1.1, complained: -1.7, complainer: -1.8, complainers: -1.3, complaining: -0.8, complainingly: -1.7, complains: -1.6, complaint: -1.2, complaints: -1.7, compliment: 2.1, complimentarily: 1.7, complimentary: 1.9, complimented: 1.8, complimenting: 2.3, compliments: 1.7, comprehensive: 1, conciliate: 1, conciliated: 1.1, conciliates: 1.1, conciliating: 1.3, condemn: -1.6, condemnation: -2.8, condemned: -1.9, condemns: -2.3, confidence: 2.3, confident: 2.2, confidently: 2.1, conflict: -1.3, conflicting: -1.7, conflictive: -1.8, conflicts: -1.6, confront: -0.7, confrontation: -1.3, confrontational: -1.6, confrontationist: -1, confrontationists: -1.2, confrontations: -1.5, confronted: -0.8, confronter: -0.3, confronters: -1.3, confronting: -0.6, confronts: -0.9, confuse: -0.9, confused: -1.3, confusedly: -0.6, confusedness: -1.5, confuses: -1.3, confusing: -0.9, confusingly: -1.4, confusion: -1.2, confusional: -1.2, confusions: -0.9, congrats: 2.4, congratulate: 2.2, congratulation: 2.9, congratulations: 2.9, consent: 0.9, consents: 1, considerate: 1.9, consolable: 1.1, conspiracy: -2.4, constrained: -0.4, contagion: -2, contagions: -1.5, contagious: -1.4, contempt: -2.8, contemptibilities: -2, contemptibility: -0.9, contemptible: -1.6, contemptibleness: -1.9, contemptibly: -1.4, contempts: -1, contemptuous: -2.2, contemptuously: -2.4, contemptuousness: -1.1, contend: 0.2, contender: 0.5, contented: 1.4, contentedly: 1.9, contentedness: 1.4, contentious: -1.2, contentment: 1.5, contestable: 0.6, contradict: -1.3, contradictable: -1, contradicted: -1.3, contradicting: -1.3, contradiction: -1, contradictions: -1.3, contradictious: -1.9, contradictor: -1, contradictories: -0.5, contradictorily: -0.9, contradictoriness: -1.4, contradictors: -1.6, contradictory: -1.4, contradicts: -1.4, controversial: -0.8, controversially: -1.1, convince: 1, convinced: 1.7, convincer: 0.6, convincers: 0.3, convinces: 0.7, convincing: 1.7, convincingly: 1.6, convincingness: 0.7, convivial: 1.2, cool: 1.3, cornered: -1.1, corpse: -2.7, costly: -0.4, courage: 2.2, courageous: 2.4, courageously: 2.3, courageousness: 2.1, courteous: 2.3, courtesy: 1.5, "cover-up": -1.2, coward: -2, cowardly: -1.6, coziness: 1.5, cramp: -0.8, crap: -1.6, crappy: -2.6, crash: -1.7, craze: -0.6, crazed: -0.5, crazes: 0.2, crazier: -0.1, craziest: -0.2, crazily: -1.5, craziness: -1.6, crazinesses: -1, crazing: -0.5, crazy: -1.4, crazyweed: 0.8, create: 1.1, created: 1, creates: 1.1, creatin: 0.1, creatine: 0.2, creating: 1.2, creatinine: 0.4, creation: 1.1, creationism: 0.7, creationisms: 1.1, creationist: 0.8, creationists: 0.5, creations: 1.6, creative: 1.9, creatively: 1.5, creativeness: 1.8, creativities: 1.7, creativity: 1.6, credit: 1.6, creditabilities: 1.4, creditability: 1.9, creditable: 1.8, creditableness: 1.2, creditably: 1.7, credited: 1.5, crediting: 0.6, creditor: -0.1, credits: 1.5, creditworthiness: 1.9, creditworthy: 2.4, crestfallen: -2.5, cried: -1.6, cries: -1.7, crime: -2.5, criminal: -2.4, criminals: -2.7, crisis: -3.1, critic: -1.1, critical: -1.3, criticise: -1.9, criticised: -1.8, criticises: -1.3, criticising: -1.7, criticism: -1.9, criticisms: -0.9, criticizable: -1, criticize: -1.6, criticized: -1.5, criticizer: -1.5, criticizers: -1.6, criticizes: -1.4, criticizing: -1.5, critics: -1.2, crude: -2.7, crudely: -1.2, crudeness: -2, crudenesses: -2, cruder: -2, crudes: -1.1, crudest: -2.4, cruel: -2.8, crueler: -2.3, cruelest: -2.6, crueller: -2.4, cruellest: -2.9, cruelly: -2.8, cruelness: -2.9, cruelties: -2.3, cruelty: -2.9, crush: -0.6, crushed: -1.8, crushes: -1.9, crushing: -1.5, cry: -2.1, crying: -2.1, cunt: -2.2, cunts: -2.9, curious: 1.3, curse: -2.5, cut: -1.1, cute: 2, cutely: 1.3, cuteness: 2.3, cutenesses: 1.9, cuter: 2.3, cutes: 1.8, cutesie: 1, cutesier: 1.5, cutesiest: 2.2, cutest: 2.8, cutesy: 2.1, cutey: 2.1, cuteys: 1.5, cutie: 1.5, cutiepie: 2, cuties: 2.2, cuts: -1.2, cutting: -0.5, cynic: -1.4, cynical: -1.6, cynically: -1.3, cynicism: -1.7, cynicisms: -1.7, cynics: -0.3, "d-:": 1.6, damage: -2.2, damaged: -1.9, damager: -1.9, damagers: -2, damages: -1.9, damaging: -2.3, damagingly: -2, damn: -1.7, damnable: -1.7, damnableness: -1.8, damnably: -1.7, damnation: -2.6, damnations: -1.4, damnatory: -2.6, damned: -1.6, damnedest: -0.5, damnified: -2.8, damnifies: -1.8, damnify: -2.2, damnifying: -2.4, damning: -1.4, damningly: -2, damnit: -2.4, damns: -2.2, danger: -2.4, dangered: -2.4, dangering: -2.5, dangerous: -2.1, dangerously: -2, dangerousness: -2, dangers: -2.2, daredevil: 0.5, daring: 1.5, daringly: 2.1, daringness: 1.4, darings: 0.4, darkest: -2.2, darkness: -1, darling: 2.8, darlingly: 1.6, darlingness: 2.3, darlings: 2.2, dauntless: 2.3, daze: -0.7, dazed: -0.7, dazedly: -0.4, dazedness: -0.5, dazes: -0.3, dead: -3.3, deadlock: -1.4, deafening: -1.2, dear: 1.6, dearer: 1.9, dearest: 2.6, dearie: 2.2, dearies: 1, dearly: 1.8, dearness: 2, dears: 1.9, dearth: -2.3, dearths: -0.9, deary: 1.9, death: -2.9, debonair: 0.8, debt: -1.5, decay: -1.7, decayed: -1.6, decayer: -1.6, decayers: -1.6, decaying: -1.7, decays: -1.7, deceit: -2, deceitful: -1.9, deceive: -1.7, deceived: -1.9, deceives: -1.6, deceiving: -1.4, deception: -1.9, decisive: 0.9, dedicated: 2, defeat: -2, defeated: -2.1, defeater: -1.4, defeaters: -0.9, defeating: -1.6, defeatism: -1.3, defeatist: -1.7, defeatists: -2.1, defeats: -1.3, defeature: -1.9, defeatures: -1.5, defect: -1.4, defected: -1.7, defecting: -1.8, defection: -1.4, defections: -1.5, defective: -1.9, defectively: -2.1, defectiveness: -1.8, defectives: -1.8, defector: -1.9, defectors: -1.3, defects: -1.7, defence: 0.4, defenceman: 0.4, defencemen: 0.6, defences: -0.2, defender: 0.4, defenders: 0.3, defense: 0.5, defenseless: -1.4, defenselessly: -1.1, defenselessness: -1.3, defenseman: 0.1, defensemen: -0.4, defenses: 0.7, defensibility: 0.4, defensible: 0.8, defensibly: 0.1, defensive: 0.1, defensively: -0.6, defensiveness: -0.4, defensives: -0.3, defer: -1.2, deferring: -0.7, defiant: -0.9, deficit: -1.7, definite: 1.1, definitely: 1.7, degradable: -1, degradation: -2.4, degradations: -1.5, degradative: -2, degrade: -1.9, degraded: -1.8, degrader: -2, degraders: -2, degrades: -2.1, degrading: -2.8, degradingly: -2.7, dehumanize: -1.8, dehumanized: -1.9, dehumanizes: -1.5, dehumanizing: -2.4, deject: -2.2, dejected: -2.2, dejecting: -2.3, dejects: -2, delay: -1.3, delayed: -0.9, delectable: 2.9, delectables: 1.4, delectably: 2.8, delicate: 0.2, delicately: 1, delicates: 0.6, delicatessen: 0.4, delicatessens: 0.4, delicious: 2.7, deliciously: 1.9, deliciousness: 1.8, delight: 2.9, delighted: 2.3, delightedly: 2.4, delightedness: 2.1, delighter: 2, delighters: 2.6, delightful: 2.8, delightfully: 2.7, delightfulness: 2.1, delighting: 1.6, delights: 2, delightsome: 2.3, demand: -0.5, demanded: -0.9, demanding: -0.9, demonstration: 0.4, demoralized: -1.6, denied: -1.9, denier: -1.5, deniers: -1.1, denies: -1.8, denounce: -1.4, denounces: -1.9, deny: -1.4, denying: -1.4, depress: -2.2, depressant: -1.6, depressants: -1.6, depressed: -2.3, depresses: -2.2, depressible: -1.7, depressing: -1.6, depressingly: -2.3, depression: -2.7, depressions: -2.2, depressive: -1.6, depressively: -2.1, depressives: -1.5, depressor: -1.8, depressors: -1.7, depressurization: -0.3, depressurizations: -0.4, depressurize: -0.5, depressurized: -0.3, depressurizes: -0.3, depressurizing: -0.7, deprival: -2.1, deprivals: -1.2, deprivation: -1.8, deprivations: -1.8, deprive: -2.1, deprived: -2.1, depriver: -1.6, deprivers: -1.4, deprives: -1.7, depriving: -2, derail: -1.2, derailed: -1.4, derails: -1.3, deride: -1.1, derided: -0.8, derides: -1, deriding: -1.5, derision: -1.2, desirable: 1.3, desire: 1.7, desired: 1.1, desirous: 1.3, despair: -1.3, despaired: -2.7, despairer: -1.3, despairers: -1.3, despairing: -2.3, despairingly: -2.2, despairs: -2.7, desperate: -1.3, desperately: -1.6, desperateness: -1.5, desperation: -2, desperations: -2.2, despise: -1.4, despised: -1.7, despisement: -2.4, despisements: -2.5, despiser: -1.8, despisers: -1.6, despises: -2, despising: -2.7, despondent: -2.1, destroy: -2.5, destroyed: -2.2, destroyer: -2, destroyers: -2.3, destroying: -2.6, destroys: -2.6, destruct: -2.4, destructed: -1.9, destructibility: -1.8, destructible: -1.5, destructing: -2.5, destruction: -2.7, destructionist: -2.6, destructionists: -2.1, destructions: -2.3, destructive: -3, destructively: -2.4, destructiveness: -2.4, destructivity: -2.2, destructs: -2.4, detached: -0.5, detain: -1.8, detained: -1.7, detention: -1.5, determinable: 0.9, determinableness: 0.2, determinably: 0.9, determinacy: 1, determinant: 0.2, determinantal: -0.3, determinate: 0.8, determinately: 1.2, determinateness: 1.1, determination: 1.7, determinations: 0.8, determinative: 1.1, determinatives: 0.9, determinator: 1.1, determined: 1.4, devastate: -3.1, devastated: -3, devastates: -2.8, devastating: -3.3, devastatingly: -2.4, devastation: -1.8, devastations: -1.9, devastative: -3.2, devastator: -2.8, devastators: -2.9, devil: -3.4, deviled: -1.6, devilfish: -0.8, devilfishes: -0.6, deviling: -2.2, devilish: -2.1, devilishly: -1.6, devilishness: -2.3, devilkin: -2.4, devilled: -2.3, devilling: -1.8, devilment: -1.9, devilments: -1.1, devilries: -1.6, devilry: -2.8, devils: -2.7, deviltries: -1.5, deviltry: -2.8, devilwood: -0.8, devilwoods: -1, devote: 1.4, devoted: 1.7, devotedly: 1.6, devotedness: 2, devotee: 1.6, devotees: 0.5, devotement: 1.5, devotements: 1.1, devotes: 1.6, devoting: 2.1, devotion: 2, devotional: 1.2, devotionally: 2.2, devotionals: 1.2, devotions: 1.8, diamond: 1.4, dick: -2.3, dickhead: -3.1, die: -2.9, died: -2.6, difficult: -1.5, difficulties: -1.2, difficultly: -1.7, difficulty: -1.4, diffident: -1, dignified: 2.2, dignifies: 2, dignify: 1.8, dignifying: 2.1, dignitaries: 0.6, dignitary: 1.9, dignities: 1.4, dignity: 1.7, dilemma: -0.7, dipshit: -2.1, dire: -2, direful: -3.1, dirt: -1.4, dirtier: -1.4, dirtiest: -2.4, dirty: -1.9, disabling: -2.1, disadvantage: -1.8, disadvantaged: -1.7, disadvantageous: -1.8, disadvantageously: -2.1, disadvantageousness: -1.6, disadvantages: -1.7, disagree: -1.6, disagreeable: -1.7, disagreeableness: -1.7, disagreeablenesses: -1.9, disagreeably: -1.5, disagreed: -1.3, disagreeing: -1.4, disagreement: -1.5, disagreements: -1.8, disagrees: -1.3, disappear: -0.9, disappeared: -0.9, disappears: -1.4, disappoint: -1.7, disappointed: -2.1, disappointedly: -1.7, disappointing: -2.2, disappointingly: -1.9, disappointment: -2.3, disappointments: -2, disappoints: -1.6, disaster: -3.1, disasters: -2.6, disastrous: -2.9, disbelieve: -1.2, discard: -1, discarded: -1.4, discarding: -0.7, discards: -1, discomfort: -1.8, discomfortable: -1.6, discomforted: -1.6, discomforting: -1.6, discomforts: -1.3, disconsolate: -2.3, disconsolation: -1.7, discontented: -1.8, discord: -1.7, discounted: 0.2, discourage: -1.8, discourageable: -1.2, discouraged: -1.7, discouragement: -2, discouragements: -1.8, discourager: -1.7, discouragers: -1.9, discourages: -1.9, discouraging: -1.9, discouragingly: -1.8, discredited: -1.9, disdain: -2.1, disgrace: -2.2, disgraced: -2, disguise: -1, disguised: -1.1, disguises: -1, disguising: -1.3, disgust: -2.9, disgusted: -2.4, disgustedly: -3, disgustful: -2.6, disgusting: -2.4, disgustingly: -2.9, disgusts: -2.1, dishearten: -2, disheartened: -2.2, disheartening: -1.8, dishearteningly: -2, disheartenment: -2.3, disheartenments: -2.2, disheartens: -2.2, dishonest: -2.7, disillusion: -1, disillusioned: -1.9, disillusioning: -1.3, disillusionment: -1.7, disillusionments: -1.5, disillusions: -1.6, disinclined: -1.1, disjointed: -1.3, dislike: -1.6, disliked: -1.7, dislikes: -1.7, disliking: -1.3, dismal: -3, dismay: -1.8, dismayed: -1.9, dismaying: -2.2, dismayingly: -1.9, dismays: -1.8, disorder: -1.7, disorganized: -1.2, disoriented: -1.5, disparage: -2, disparaged: -1.4, disparages: -1.6, disparaging: -2.2, displeased: -1.9, dispute: -1.7, disputed: -1.4, disputes: -1.1, disputing: -1.7, disqualified: -1.8, disquiet: -1.3, disregard: -1.1, disregarded: -1.6, disregarding: -0.9, disregards: -1.4, disrespect: -1.8, disrespected: -2, disruption: -1.5, disruptions: -1.4, disruptive: -1.3, dissatisfaction: -2.2, dissatisfactions: -1.9, dissatisfactory: -2, dissatisfied: -1.6, dissatisfies: -1.8, dissatisfy: -2.2, dissatisfying: -2.4, distort: -1.3, distorted: -1.7, distorting: -1.1, distorts: -1.4, distract: -1.2, distractable: -1.3, distracted: -1.4, distractedly: -0.9, distractibility: -1.3, distractible: -1.5, distracting: -1.2, distractingly: -1.4, distraction: -1.6, distractions: -1, distractive: -1.6, distracts: -1.3, distraught: -2.6, distress: -2.4, distressed: -1.8, distresses: -1.6, distressful: -2.2, distressfully: -1.7, distressfulness: -2.4, distressing: -1.7, distressingly: -2.2, distrust: -1.8, distrusted: -2.4, distrustful: -2.1, distrustfully: -1.8, distrustfulness: -1.6, distrusting: -2.1, distrusts: -1.3, disturb: -1.7, disturbance: -1.6, disturbances: -1.4, disturbed: -1.6, disturber: -1.4, disturbers: -2.1, disturbing: -2.3, disturbingly: -2.3, disturbs: -1.9, dithering: -0.5, divination: 1.7, divinations: 1.1, divinatory: 1.6, divine: 2.6, divined: 0.8, divinely: 2.9, diviner: 0.3, diviners: 1.2, divines: 0.8, divinest: 2.7, diving: 0.3, divining: 0.9, divinise: 0.5, divinities: 1.8, divinity: 2.7, divinize: 2.3, dizzy: -0.9, dodging: -0.4, dodgy: -0.9, dolorous: -2.2, dominance: 0.8, dominances: -0.1, dominantly: 0.2, dominants: 0.2, dominate: -0.5, dominates: 0.2, dominating: -1.2, domination: -0.2, dominations: -0.3, dominative: -0.7, dominators: -0.4, dominatrices: -0.2, dominatrix: -0.5, dominatrixes: 0.6, doom: -1.7, doomed: -3.2, doomful: -2.1, dooming: -2.8, dooms: -1.1, doomsayer: -0.7, doomsayers: -1.7, doomsaying: -1.5, doomsayings: -1.5, doomsday: -2.8, doomsdayer: -2.2, doomsdays: -2.4, doomster: -2.2, doomsters: -1.6, doomy: -1.1, dork: -1.4, dorkier: -1.1, dorkiest: -1.2, dorks: -0.5, dorky: -1.1, doubt: -1.5, doubtable: -1.5, doubted: -1.1, doubter: -1.6, doubters: -1.3, doubtful: -1.4, doubtfully: -1.2, doubtfulness: -1.2, doubting: -1.4, doubtingly: -1.4, doubtless: 0.9, doubtlessly: 1.2, doubtlessness: 0.8, doubts: -1.2, douche: -1.5, douchebag: -3, downcast: -1.8, downhearted: -2.3, downside: -1, drag: -0.9, dragged: -0.2, drags: -0.7, drained: -1.5, dread: -2, dreaded: -2.7, dreadful: -1.9, dreadfully: -2.7, dreadfulness: -3.2, dreadfuls: -2.4, dreading: -2.4, dreadlock: -0.4, dreadlocks: -0.2, dreadnought: -0.6, dreadnoughts: -0.4, dreads: -1.4, dream: 1, dreams: 1.7, dreary: -1.4, droopy: -0.8, drop: -1.1, drown: -2.7, drowned: -2.9, drowns: -2.2, drunk: -1.4, dubious: -1.5, dud: -1, dull: -1.7, dullard: -1.6, dullards: -1.8, dulled: -1.5, duller: -1.7, dullest: -1.7, dulling: -1.1, dullish: -1.1, dullness: -1.4, dullnesses: -1.9, dulls: -1, dullsville: -2.4, dully: -1.1, dumb: -2.3, dumbass: -2.6, dumbbell: -0.8, dumbbells: -0.2, dumbcane: -0.3, dumbcanes: -0.6, dumbed: -1.4, dumber: -1.5, dumbest: -2.3, dumbfound: -0.1, dumbfounded: -1.6, dumbfounder: -1, dumbfounders: -1, dumbfounding: -0.8, dumbfounds: -0.3, dumbhead: -2.6, dumbheads: -1.9, dumbing: -0.5, dumbly: -1.3, dumbness: -1.9, dumbs: -1.5, dumbstruck: -1, dumbwaiter: 0.2, dumbwaiters: -0.1, dump: -1.6, dumpcart: -0.6, dumped: -1.7, dumper: -1.2, dumpers: -0.8, dumpier: -1.4, dumpiest: -1.6, dumpiness: -1.2, dumping: -1.3, dumpings: -1.1, dumpish: -1.8, dumpling: 0.4, dumplings: -0.3, dumps: -1.7, dumpster: -0.6, dumpsters: -1, dumpy: -1.7, dupe: -1.5, duped: -1.8, dwell: 0.5, dwelled: 0.4, dweller: 0.3, dwellers: -0.3, dwelling: 0.1, dwells: -0.1, dynamic: 1.6, dynamical: 1.2, dynamically: 1.5, dynamics: 1.1, dynamism: 1.6, dynamisms: 1.2, dynamist: 1.4, dynamistic: 1.5, dynamists: 0.9, dynamite: 0.7, dynamited: -0.9, dynamiter: -1.2, dynamiters: 0.4, dynamites: -0.3, dynamitic: 0.9, dynamiting: 0.2, dynamometer: 0.3, dynamometers: 0.3, dynamometric: 0.3, dynamometry: 0.6, dynamos: 0.3, dynamotor: 0.6, dysfunction: -1.8, eager: 1.5, eagerly: 1.6, eagerness: 1.7, eagers: 1.6, earnest: 2.3, ease: 1.5, eased: 1.2, easeful: 1.5, easefully: 1.4, easel: 0.3, easement: 1.6, easements: 0.4, eases: 1.3, easier: 1.8, easiest: 1.8, easily: 1.4, easiness: 1.6, easing: 1, easy: 1.9, easygoing: 1.3, easygoingness: 1.5, ecstacy: 3.3, ecstasies: 2.3, ecstasy: 2.9, ecstatic: 2.3, ecstatically: 2.8, ecstatics: 2.9, eerie: -1.5, eery: -0.9, effective: 2.1, effectively: 1.9, efficiencies: 1.6, efficiency: 1.5, efficient: 1.8, efficiently: 1.7, effin: -2.3, egotism: -1.4, egotisms: -1, egotist: -2.3, egotistic: -1.4, egotistical: -0.9, egotistically: -1.8, egotists: -1.7, elated: 3.2, elation: 1.5, elegance: 2.1, elegances: 1.8, elegancies: 1.6, elegancy: 2.1, elegant: 2.1, elegantly: 1.9, embarrass: -1.2, embarrassable: -1.6, embarrassed: -1.5, embarrassedly: -1.1, embarrasses: -1.7, embarrassing: -1.6, embarrassingly: -1.7, embarrassment: -1.9, embarrassments: -1.7, embittered: -0.4, embrace: 1.3, emergency: -1.6, emotional: 0.6, empathetic: 1.7, emptied: -0.7, emptier: -0.7, emptiers: -0.7, empties: -0.7, emptiest: -1.8, emptily: -1, emptiness: -1.9, emptinesses: -1.5, emptins: -0.3, empty: -0.8, emptying: -0.6, enchanted: 1.6, encourage: 2.3, encouraged: 1.5, encouragement: 1.8, encouragements: 2.1, encourager: 1.5, encouragers: 1.5, encourages: 1.9, encouraging: 2.4, encouragingly: 2, endorse: 1.3, endorsed: 1, endorsement: 1.3, endorses: 1.4, enemies: -2.2, enemy: -2.5, energetic: 1.9, energetically: 1.8, energetics: 0.3, energies: 0.9, energise: 2.2, energised: 2.1, energises: 2.2, energising: 1.9, energization: 1.6, energizations: 1.5, energize: 2.1, energized: 2.3, energizer: 2.1, energizers: 1.7, energizes: 2.1, energizing: 2, energy: 1.1, engage: 1.4, engaged: 1.7, engagement: 2, engagements: 0.6, engager: 1.1, engagers: 1, engages: 1, engaging: 1.4, engagingly: 1.5, engrossed: 0.6, enjoy: 2.2, enjoyable: 1.9, enjoyableness: 1.9, enjoyably: 1.8, enjoyed: 2.3, enjoyer: 2.2, enjoyers: 2.2, enjoying: 2.4, enjoyment: 2.6, enjoyments: 2, enjoys: 2.3, enlighten: 2.3, enlightened: 2.2, enlightening: 2.3, enlightens: 1.7, ennui: -1.2, enrage: -2.6, enraged: -1.7, enrages: -1.8, enraging: -2.8, enrapture: 3, enslave: -3.1, enslaved: -1.7, enslaves: -1.6, ensure: 1.6, ensuring: 1.1, enterprising: 2.3, entertain: 1.3, entertained: 1.7, entertainer: 1.6, entertainers: 1, entertaining: 1.9, entertainingly: 1.9, entertainment: 1.8, entertainments: 2.3, entertains: 2.4, enthral: 0.4, enthuse: 1.6, enthused: 2, enthuses: 1.7, enthusiasm: 1.9, enthusiasms: 2, enthusiast: 1.5, enthusiastic: 2.2, enthusiastically: 2.6, enthusiasts: 1.4, enthusing: 1.9, entitled: 1.1, entrusted: 0.8, envied: -1.1, envier: -1, enviers: -1.1, envies: -0.8, envious: -1.1, envy: -1.1, envying: -0.8, envyingly: -1.3, erroneous: -1.8, error: -1.7, errors: -1.4, escape: 0.7, escapes: 0.5, escaping: 0.2, esteemed: 1.9, ethical: 2.3, euphoria: 3.3, euphoric: 3.2, eviction: -2, evil: -3.4, evildoer: -3.1, evildoers: -2.4, evildoing: -3.1, evildoings: -2.5, eviler: -2.1, evilest: -2.5, eviller: -2.9, evillest: -3.3, evilly: -3.4, evilness: -3.1, evils: -2.7, exaggerate: -0.6, exaggerated: -0.4, exaggerates: -0.6, exaggerating: -0.7, exasperated: -1.8, excel: 2, excelled: 2.2, excellence: 3.1, excellences: 2.5, excellencies: 2.4, excellency: 2.5, excellent: 2.7, excellently: 3.1, excelling: 2.5, excels: 2.5, excelsior: 0.7, excitabilities: 1.5, excitability: 1.2, excitable: 1.5, excitableness: 1, excitant: 1.8, excitants: 1.2, excitation: 1.8, excitations: 1.8, excitative: 0.3, excitatory: 1.1, excite: 2.1, excited: 1.4, excitedly: 2.3, excitement: 2.2, excitements: 1.9, exciter: 1.9, exciters: 1.4, excites: 2.1, exciting: 2.2, excitingly: 1.9, exciton: 0.3, excitonic: 0.2, excitons: 0.8, excitor: 0.5, exclude: -0.9, excluded: -1.4, exclusion: -1.2, exclusive: 0.5, excruciate: -2.7, excruciated: -1.3, excruciates: -1, excruciating: -3.3, excruciatingly: -2.9, excruciation: -3.4, excruciations: -1.9, excuse: 0.3, exempt: 0.4, exhaust: -1.2, exhausted: -1.5, exhauster: -1.3, exhausters: -1.3, exhaustibility: -0.8, exhaustible: -1, exhausting: -1.5, exhaustion: -1.5, exhaustions: -1.1, exhaustive: -0.5, exhaustively: -0.7, exhaustiveness: -1.1, exhaustless: 0.2, exhaustlessness: 0.9, exhausts: -1.1, exhilarated: 3, exhilarates: 2.8, exhilarating: 1.7, exonerate: 1.8, exonerated: 1.8, exonerates: 1.6, exonerating: 1, expand: 1.3, expands: 0.4, expel: -1.9, expelled: -1, expelling: -1.6, expels: -1.6, exploit: -0.4, exploited: -2, exploiting: -1.9, exploits: -1.4, exploration: 0.9, explorations: 0.3, expose: -0.6, exposed: -0.3, exposes: -0.5, exposing: -1.1, extend: 0.7, extends: 0.5, exuberant: 2.8, exultant: 3, exultantly: 1.4, fab: 2, fabulous: 2.4, fabulousness: 2.8, fad: 0.9, fag: -2.1, faggot: -3.4, faggots: -3.2, fail: -2.5, failed: -2.3, failing: -2.3, failingly: -1.4, failings: -2.2, faille: 0.1, fails: -1.8, failure: -2.3, failures: -2, fainthearted: -0.3, fair: 1.3, faith: 1.8, faithed: 1.3, faithful: 1.9, faithfully: 1.8, faithfulness: 1.9, faithless: -1, faithlessly: -0.9, faithlessness: -1.8, faiths: 1.8, fake: -2.1, fakes: -1.8, faking: -1.8, fallen: -1.5, falling: -0.6, falsified: -1.6, falsify: -2, fame: 1.9, fan: 1.3, fantastic: 2.6, fantastical: 2, fantasticalities: 2.1, fantasticality: 1.7, fantasticalness: 1.3, fantasticate: 1.5, fantastico: 0.4, farce: -1.7, fascinate: 2.4, fascinated: 2.1, fascinates: 2, fascination: 2.2, fascinating: 2.5, fascist: -2.6, fascists: -0.8, fatal: -2.5, fatalism: -0.6, fatalisms: -1.7, fatalist: -0.5, fatalistic: -1, fatalists: -1.2, fatalities: -2.9, fatality: -3.5, fatally: -3.2, fatigue: -1, fatigued: -1.4, fatigues: -1.3, fatiguing: -1.2, fatiguingly: -1.5, fault: -1.7, faulted: -1.4, faultfinder: -0.8, faultfinders: -1.5, faultfinding: -2.1, faultier: -2.1, faultiest: -2.1, faultily: -2, faultiness: -1.5, faulting: -1.4, faultless: 2, faultlessly: 2, faultlessness: 1.1, faults: -2.1, faulty: -1.3, fave: 1.9, favor: 1.7, favorable: 2.1, favorableness: 2.2, favorably: 1.6, favored: 1.8, favorer: 1.3, favorers: 1.4, favoring: 1.8, favorite: 2, favorited: 1.7, favorites: 1.8, favoritism: 0.7, favoritisms: 0.7, favors: 1, favour: 1.9, favoured: 1.8, favourer: 1.6, favourers: 1.6, favouring: 1.3, favours: 1.8, fear: -2.2, feared: -2.2, fearful: -2.2, fearfuller: -2.2, fearfullest: -2.5, fearfully: -2.2, fearfulness: -1.8, fearing: -2.7, fearless: 1.9, fearlessly: 1.1, fearlessness: 1.1, fears: -1.8, fearsome: -1.7, "fed up": -1.8, feeble: -1.2, feeling: 0.5, felonies: -2.5, felony: -2.5, ferocious: -0.4, ferociously: -1.1, ferociousness: -1, ferocities: -1, ferocity: -0.7, fervent: 1.1, fervid: 0.5, festival: 2.2, festivalgoer: 1.3, festivalgoers: 1.2, festivals: 1.5, festive: 2, festively: 2.2, festiveness: 2.4, festivities: 2.1, festivity: 2.2, feud: -1.4, feudal: -0.8, feudalism: -0.9, feudalisms: -0.2, feudalist: -0.9, feudalistic: -1.1, feudalities: -0.4, feudality: -0.5, feudalization: -0.3, feudalize: -0.5, feudalized: -0.8, feudalizes: -0.1, feudalizing: -0.7, feudally: -0.6, feudaries: -0.3, feudary: -0.8, feudatories: -0.5, feudatory: -0.1, feuded: -2.2, feuding: -1.6, feudist: -1.1, feudists: -0.7, feuds: -1.4, fiasco: -2.3, fidgety: -1.4, fiery: -1.4, fiesta: 2.1, fiestas: 1.5, fight: -1.6, fighter: 0.6, fighters: -0.2, fighting: -1.5, fightings: -1.9, fights: -1.7, fine: 0.8, fire: -1.4, fired: -2.6, firing: -1.4, fit: 1.5, fitness: 1.1, flagship: 0.4, flatter: 0.4, flattered: 1.6, flatterer: -0.3, flatterers: 0.3, flatteries: 1.2, flattering: 1.3, flatteringly: 1, flatters: 0.6, flattery: 0.4, flawed: -2.1, flawless: 2.3, flawlessly: 0.8, flees: -0.7, flexibilities: 1, flexibility: 1.4, flexible: 0.9, flexibly: 1.3, flirtation: 1.7, flirtations: -0.1, flirtatious: 0.5, flirtatiously: -0.1, flirtatiousness: 0.6, flirted: -0.2, flirter: -0.4, flirters: 0.6, flirtier: -0.1, flirtiest: 0.4, flirting: 0.8, flirts: 0.7, flirty: 0.6, flop: -1.4, flops: -1.4, flu: -1.6, flunk: -1.3, flunked: -2.1, flunker: -1.9, flunkers: -1.6, flunkey: -1.8, flunkeys: -0.6, flunkies: -1.4, flunking: -1.5, flunks: -1.8, flunky: -1.8, flustered: -1, focused: 1.6, foe: -1.9, foehns: 0.2, foeman: -1.8, foemen: -0.3, foes: -2, foetal: -0.1, foetid: -2.3, foetor: -3, foetors: -2.1, foetus: 0.2, foetuses: 0.2, fond: 1.9, fondly: 1.9, fondness: 2.5, fool: -1.9, fooled: -1.6, fooleries: -1.8, foolery: -1.8, foolfish: -0.8, foolfishes: -0.4, foolhardier: -1.5, foolhardiest: -1.3, foolhardily: -1, foolhardiness: -1.6, foolhardy: -1.4, fooling: -1.7, foolish: -1.1, foolisher: -1.7, foolishest: -1.4, foolishly: -1.8, foolishness: -1.8, foolishnesses: -2, foolproof: 1.6, fools: -2.2, foolscaps: -0.8, forbid: -1.3, forbiddance: -1.4, forbiddances: -1, forbidden: -1.8, forbidder: -1.6, forbidders: -1.5, forbidding: -1.9, forbiddingly: -1.9, forbids: -1.3, forced: -2, foreclosure: -0.5, foreclosures: -2.4, forgave: 1.4, forget: -0.9, forgetful: -1.1, forgivable: 1.7, forgivably: 1.6, forgive: 1.1, forgiven: 1.6, forgiveness: 1.1, forgiver: 1.7, forgivers: 1.2, forgives: 1.7, forgiving: 1.9, forgivingly: 1.4, forgivingness: 1.8, forgotten: -0.9, fortunate: 1.9, fought: -1.3, foughten: -1.9, frantic: -1.9, frantically: -1.4, franticness: -0.7, fraud: -2.8, frauds: -2.3, fraudster: -2.5, fraudsters: -2.4, fraudulence: -2.3, fraudulent: -2.2, freak: -1.9, freaked: -1.2, freakier: -1.3, freakiest: -1.6, freakiness: -1.4, freaking: -1.8, freakish: -2.1, freakishly: -0.8, freakishness: -1.4, freakout: -1.8, freakouts: -1.5, freaks: -0.4, freaky: -1.5, free: 2.3, freebase: -0.1, freebased: 0.8, freebases: 0.8, freebasing: -0.4, freebee: 1.3, freebees: 1.3, freebie: 1.8, freebies: 1.8, freeboard: 0.3, freeboards: 0.7, freeboot: -0.7, freebooter: -1.7, freebooters: -0.2, freebooting: -0.8, freeborn: 1.2, freed: 1.7, freedman: 1.1, freedmen: 0.7, freedom: 3.2, freedoms: 1.2, freedwoman: 1.6, freedwomen: 1.3, freeform: 0.9, freehand: 0.5, freehanded: 1.4, freehearted: 1.5, freehold: 0.7, freeholder: 0.5, freeholders: 0.1, freeholds: 1, freeing: 2.1, freelance: 1.2, freelanced: 0.7, freelancer: 1.1, freelancers: 0.4, freelances: 0.7, freelancing: 0.4, freeload: -1.9, freeloaded: -1.6, freeloader: -0.7, freeloaders: -0.1, freeloading: -1.3, freeloads: -1.3, freely: 1.9, freeman: 1.7, freemartin: -0.5, freemasonries: 0.7, freemasonry: 0.3, freemen: 1.5, freeness: 1.6, freenesses: 1.7, freer: 1.1, freers: 1, frees: 1.2, freesia: 0.4, freesias: 0.4, freest: 1.6, freestanding: 1.1, freestyle: 0.7, freestyler: 0.4, freestylers: 0.8, freestyles: 0.3, freethinker: 1, freethinkers: 1, freethinking: 1.1, freeware: 0.7, freeway: 0.2, freewheel: 0.5, freewheeled: 0.3, freewheeler: 0.2, freewheelers: -0.3, freewheeling: 0.5, freewheelingly: 0.8, freewheels: 0.6, freewill: 1, freewriting: 0.8, freeze: 0.2, freezers: -0.1, freezes: -0.1, freezing: -0.4, freezingly: -1.6, frenzy: -1.3, fresh: 1.3, friend: 2.2, friended: 1.7, friending: 1.8, friendless: -1.5, friendlessness: -0.3, friendlier: 2, friendlies: 2.2, friendliest: 2.6, friendlily: 1.8, friendliness: 2, friendly: 2.2, friends: 2.1, friendship: 1.9, friendships: 1.6, fright: -1.6, frighted: -1.4, frighten: -1.4, frightened: -1.9, frightening: -2.2, frighteningly: -2.1, frightens: -1.7, frightful: -2.3, frightfully: -1.7, frightfulness: -1.9, frighting: -1.5, frights: -1.1, frisky: 1, frowning: -1.4, frustrate: -2, frustrated: -2.4, frustrates: -1.9, frustrating: -1.9, frustratingly: -2, frustration: -2.1, frustrations: -2, fuck: -2.5, fucked: -3.4, fucker: -3.3, fuckers: -2.9, fuckface: -3.2, fuckhead: -3.1, fucks: -2.1, fucktard: -3.1, fud: -1.1, fuked: -2.5, fuking: -3.2, fulfill: 1.9, fulfilled: 1.8, fulfills: 1, fume: -1.2, fumed: -1.8, fumeless: 0.3, fumelike: -0.7, fumer: 0.7, fumers: -0.8, fumes: -0.1, fumet: 0.4, fumets: -0.4, fumette: -0.6, fuming: -2.7, fun: 2.3, funeral: -1.5, funerals: -1.6, funky: -0.4, funned: 2.3, funnel: 0.1, funneled: 0.1, funnelform: 0.5, funneling: -0.1, funnelled: -0.1, funnelling: 0.1, funnels: 0.4, funner: 2.2, funnest: 2.9, funnier: 1.7, funnies: 1.3, funniest: 2.6, funnily: 1.9, funniness: 1.8, funninesses: 1.6, funning: 1.8, funny: 1.9, funnyman: 1.4, funnymen: 1.3, furious: -2.7, furiously: -1.9, fury: -2.7, futile: -1.9, gag: -1.4, gagged: -1.3, gain: 2.4, gained: 1.6, gaining: 1.8, gains: 1.4, gallant: 1.7, gallantly: 1.9, gallantry: 2.6, geek: -0.8, geekier: 0.2, geekiest: -0.1, geeks: -0.4, geeky: -0.6, generosities: 2.6, generosity: 2.3, generous: 2.3, generously: 1.8, generousness: 2.4, genial: 1.8, gentle: 1.9, gentler: 1.4, gentlest: 1.8, gently: 2, ghost: -1.3, giddy: -0.6, gift: 1.9, giggle: 1.8, giggled: 1.5, giggler: 0.6, gigglers: 1.4, giggles: 0.8, gigglier: 1, giggliest: 1.7, giggling: 1.5, gigglingly: 1.1, giggly: 1, giver: 1.4, givers: 1.7, giving: 1.4, glad: 2, gladly: 1.4, glamor: 2.1, glamorise: 1.3, glamorised: 1.8, glamorises: 2.1, glamorising: 1.2, glamorization: 1.6, glamorize: 1.7, glamorized: 2.1, glamorizer: 2.4, glamorizers: 1.6, glamorizes: 2.4, glamorizing: 1.8, glamorous: 2.3, glamorously: 2.1, glamors: 1.4, glamour: 2.4, glamourize: 0.8, glamourless: -1.6, glamourous: 2, glamours: 1.9, glee: 3.2, gleeful: 2.9, gloom: -2.6, gloomed: -1.9, gloomful: -2.1, gloomier: -1.5, gloomiest: -1.8, gloominess: -1.8, gloominesses: -1, glooming: -1.8, glooms: -0.9, gloomy: -0.6, gloried: 2.4, glories: 2.1, glorification: 2, glorified: 2.3, glorifier: 2.3, glorifiers: 1.6, glorifies: 2.2, glorify: 2.7, glorifying: 2.4, gloriole: 1.5, glorioles: 1.2, glorious: 3.2, gloriously: 2.9, gloriousness: 2.6, glory: 2.5, glum: -2.1, gn8: 0.6, god: 1.1, goddam: -2.5, goddammed: -2.4, goddamn: -2.1, goddamned: -1.8, goddamns: -2.1, goddams: -1.9, godsend: 2.8, good: 1.9, goodness: 2, gorgeous: 3, gorgeously: 2.3, gorgeousness: 2.9, gorgeousnesses: 2.1, gossip: -0.7, gossiped: -1.1, gossiper: -1.1, gossipers: -1.1, gossiping: -1.6, gossipmonger: -1, gossipmongers: -1.4, gossipped: -1.3, gossipping: -1.8, gossipries: -0.8, gossipry: -1.2, gossips: -1.3, gossipy: -1.3, grace: 1.8, graced: 0.9, graceful: 2, gracefuller: 2.2, gracefullest: 2.8, gracefully: 2.4, gracefulness: 2.2, graces: 1.6, gracile: 1.7, graciles: 0.6, gracilis: 0.4, gracility: 1.2, gracing: 1.3, gracioso: 1, gracious: 2.6, graciously: 2.3, graciousness: 2.4, grand: 2, grandee: 1.1, grandees: 1.2, grander: 1.7, grandest: 2.4, grandeur: 2.4, grandeurs: 2.1, grant: 1.5, granted: 1, granting: 1.3, grants: 0.9, grateful: 2, gratefuller: 1.8, gratefully: 2.1, gratefulness: 2.2, graticule: 0.1, graticules: 0.2, gratification: 1.6, gratifications: 1.8, gratified: 1.6, gratifies: 1.5, gratify: 1.3, gratifying: 2.3, gratifyingly: 2, gratin: 0.4, grating: -0.4, gratingly: -0.2, gratings: -0.8, gratins: 0.2, gratis: 0.2, gratitude: 2.3, gratz: 2, grave: -1.6, graved: -0.9, gravel: -0.5, graveled: -0.5, graveless: -1.3, graveling: -0.4, gravelled: -0.9, gravelling: -0.4, gravelly: -0.9, gravels: -0.5, gravely: -1.5, graven: -0.9, graveness: -1.5, graver: -1.1, gravers: -1.2, graves: -1.2, graveside: -0.8, gravesides: -1.6, gravest: -1.3, gravestone: -0.7, gravestones: -0.5, graveyard: -1.2, graveyards: -1.2, great: 3.1, greater: 1.5, greatest: 3.2, greed: -1.7, greedier: -2, greediest: -2.8, greedily: -1.9, greediness: -1.7, greeds: -1, greedy: -1.3, greenwash: -1.8, greenwashing: -0.4, greet: 1.3, greeted: 1.1, greeting: 1.6, greetings: 1.8, greets: 0.6, grey: 0.2, grief: -2.2, grievance: -2.1, grievances: -1.5, grievant: -0.8, grievants: -1.1, grieve: -1.6, grieved: -2, griever: -1.9, grievers: -0.3, grieves: -2.1, grieving: -2.3, grievous: -2, grievously: -1.7, grievousness: -2.7, grim: -2.7, grimace: -1, grimaced: -2, grimaces: -1.8, grimacing: -1.4, grimalkin: -0.9, grimalkins: -0.9, grime: -1.5, grimed: -1.2, grimes: -1, grimier: -1.6, grimiest: -0.7, grimily: -0.7, griminess: -1.6, griming: -0.7, grimly: -1.3, grimmer: -1.5, grimmest: -0.8, grimness: -0.8, grimy: -1.8, grin: 2.1, grinned: 1.1, grinner: 1.1, grinners: 1.6, grinning: 1.5, grins: 0.9, gross: -2.1, grossed: -0.4, grosser: -0.3, grosses: -0.8, grossest: -2.1, grossing: -0.3, grossly: -0.9, grossness: -1.8, grossular: -0.3, grossularite: -0.1, grossularites: -0.7, grossulars: -0.3, grouch: -2.2, grouched: -0.8, grouches: -0.9, grouchier: -2, grouchiest: -2.3, grouchily: -1.4, grouchiness: -2, grouching: -1.7, grouchy: -1.9, growing: 0.7, growth: 1.6, guarantee: 1, guilt: -1.1, guiltier: -2, guiltiest: -1.7, guiltily: -1.1, guiltiness: -1.8, guiltless: 0.8, guiltlessly: 0.7, guiltlessness: 0.6, guilts: -1.4, guilty: -1.8, gullibility: -1.6, gullible: -1.5, gun: -1.4, h8: -2.7, ha: 1.4, hacked: -1.7, haha: 2, hahaha: 2.6, hahas: 1.8, hail: 0.3, hailed: 0.9, hallelujah: 3, handsome: 2.2, handsomely: 1.9, handsomeness: 2.4, handsomer: 2, handsomest: 2.6, hapless: -1.4, haplessness: -1.4, happier: 2.4, happiest: 3.2, happily: 2.6, happiness: 2.6, happing: 1.1, happy: 2.7, harass: -2.2, harassed: -2.5, harasser: -2.4, harassers: -2.8, harasses: -2.5, harassing: -2.5, harassment: -2.5, harassments: -2.6, hard: -0.4, hardier: -0.6, hardship: -1.3, hardy: 1.7, harm: -2.5, harmed: -2.1, harmfully: -2.6, harmfulness: -2.6, harming: -2.6, harmless: 1, harmlessly: 1.4, harmlessness: 0.8, harmonic: 1.8, harmonica: 0.6, harmonically: 2.1, harmonicas: 0.1, harmonicist: 0.5, harmonicists: 0.9, harmonics: 1.5, harmonies: 1.3, harmonious: 2, harmoniously: 1.9, harmoniousness: 1.8, harmonise: 1.8, harmonised: 1.3, harmonising: 1.4, harmonium: 0.9, harmoniums: 0.8, harmonization: 1.9, harmonizations: 0.9, harmonize: 1.7, harmonized: 1.6, harmonizer: 1.6, harmonizers: 1.6, harmonizes: 1.5, harmonizing: 1.4, harmony: 1.7, harms: -2.2, harried: -1.4, harsh: -1.9, harsher: -2.2, harshest: -2.9, hate: -2.7, hated: -3.2, hateful: -2.2, hatefully: -2.3, hatefulness: -3.6, hater: -1.8, haters: -2.2, hates: -1.9, hating: -2.3, hatred: -3.2, haunt: -1.7, haunted: -2.1, haunting: -1.1, haunts: -1, havoc: -2.9, healthy: 1.7, heartbreak: -2.7, heartbreaker: -2.2, heartbreakers: -2.1, heartbreaking: -2, heartbreakingly: -1.8, heartbreaks: -1.8, heartbroken: -3.3, heartfelt: 2.5, heartless: -2.2, heartlessly: -2.8, heartlessness: -2.8, heartwarming: 2.1, heaven: 2.3, heavenlier: 3, heavenliest: 2.7, heavenliness: 2.7, heavenlinesses: 2.3, heavenly: 3, heavens: 1.7, heavenward: 1.4, heavenwards: 1.2, heavyhearted: -2.1, heh: -0.6, hell: -3.6, hellish: -3.2, help: 1.7, helper: 1.4, helpers: 1.1, helpful: 1.8, helpfully: 2.3, helpfulness: 1.9, helping: 1.2, helpless: -2, helplessly: -1.4, helplessness: -2.1, helplessnesses: -1.7, helps: 1.6, hero: 2.6, heroes: 2.3, heroic: 2.6, heroical: 2.9, heroically: 2.4, heroicomic: 1, heroicomical: 1.1, heroics: 2.4, heroin: -2.2, heroine: 2.7, heroines: 1.8, heroinism: -2, heroism: 2.8, heroisms: 2.2, heroize: 2.1, heroized: 2, heroizes: 2.2, heroizing: 1.9, heron: 0.1, heronries: 0.7, heronry: 0.1, herons: 0.5, heros: 1.3, hesitance: -0.9, hesitancies: -1, hesitancy: -0.9, hesitant: -1, hesitantly: -1.2, hesitate: -1.1, hesitated: -1.3, hesitater: -1.4, hesitaters: -1.4, hesitates: -1.4, hesitating: -1.4, hesitatingly: -1.5, hesitation: -1.1, hesitations: -1.1, hid: -0.4, hide: -0.7, hides: -0.7, hiding: -1.2, highlight: 1.4, hilarious: 1.7, hindrance: -1.7, hoax: -1.1, holiday: 1.7, holidays: 1.6, homesick: -0.7, homesickness: -1.8, homesicknesses: -1.8, honest: 2.3, honester: 1.9, honestest: 3, honesties: 1.8, honestly: 2, honesty: 2.2, honor: 2.2, honorability: 2.2, honorable: 2.5, honorableness: 2.2, honorably: 2.4, honoraria: 0.6, honoraries: 1.5, honorarily: 1.9, honorarium: 0.7, honorariums: 1, honorary: 1.4, honored: 2.8, honoree: 2.1, honorees: 2.3, honorer: 1.7, honorers: 1.3, honorific: 1.4, honorifically: 2.2, honorifics: 1.7, honoring: 2.3, honors: 2.3, honour: 2.7, honourable: 2.1, honoured: 2.2, honourer: 1.8, honourers: 1.6, honouring: 2.1, honours: 2.2, hooligan: -1.5, hooliganism: -2.1, hooligans: -1.1, hooray: 2.3, hope: 1.9, hoped: 1.6, hopeful: 2.3, hopefully: 1.7, hopefulness: 1.6, hopeless: -2, hopelessly: -2.2, hopelessness: -3.1, hopes: 1.8, hoping: 1.8, horrendous: -2.8, horrendously: -1.9, horrent: -0.9, horrible: -2.5, horribleness: -2.4, horribles: -2.1, horribly: -2.4, horrid: -2.5, horridly: -1.4, horridness: -2.3, horridnesses: -3, horrific: -3.4, horrifically: -2.9, horrified: -2.5, horrifies: -2.9, horrify: -2.5, horrifying: -2.7, horrifyingly: -3.3, horror: -2.7, horrors: -2.7, hostile: -1.6, hostilely: -2.2, hostiles: -1.3, hostilities: -2.1, hostility: -2.5, huckster: -0.9, hug: 2.1, huge: 1.3, huggable: 1.6, hugged: 1.7, hugger: 1.6, huggers: 1.8, hugging: 1.8, hugs: 2.2, humerous: 1.4, humiliate: -2.5, humiliated: -1.4, humiliates: -1, humiliating: -1.2, humiliatingly: -2.6, humiliation: -2.7, humiliations: -2.4, humor: 1.1, humoral: 0.6, humored: 1.2, humoresque: 1.2, humoresques: 0.9, humoring: 2.1, humorist: 1.2, humoristic: 1.5, humorists: 1.3, humorless: -1.3, humorlessness: -1.4, humorous: 1.6, humorously: 2.3, humorousness: 2.4, humors: 1.6, humour: 2.1, humoured: 1.1, humouring: 1.7, humourous: 2, hunger: -1, hurrah: 2.6, hurrahed: 1.9, hurrahing: 2.4, hurrahs: 2.1, hurray: 2.7, hurrayed: 1.8, hurraying: 1.2, hurrays: 2.4, hurt: -2.4, hurter: -2.3, hurters: -1.9, hurtful: -2.4, hurtfully: -2.6, hurtfulness: -1.9, hurting: -1.7, hurtle: -0.3, hurtled: -0.6, hurtles: -1, hurtless: 0.3, hurtling: -1.4, hurts: -2.1, hypocritical: -2, hysteria: -1.9, hysterical: -0.1, hysterics: -1.8, ideal: 2.4, idealess: -1.9, idealise: 1.4, idealised: 2.1, idealises: 2, idealising: 0.6, idealism: 1.7, idealisms: 0.8, idealist: 1.6, idealistic: 1.8, idealistically: 1.7, idealists: 0.7, idealities: 1.5, ideality: 1.9, idealization: 1.8, idealizations: 1.4, idealize: 1.2, idealized: 1.8, idealizer: 1.3, idealizers: 1.9, idealizes: 2, idealizing: 1.4, idealless: -1.7, ideally: 1.8, idealogues: 0.5, idealogy: 0.8, ideals: 0.8, idiot: -2.3, idiotic: -2.6, ignorable: -1, ignorami: -1.9, ignoramus: -1.9, ignoramuses: -2.3, ignorance: -1.5, ignorances: -1.2, ignorant: -1.1, ignorantly: -1.6, ignorantness: -1.1, ignore: -1.5, ignored: -1.3, ignorer: -1.3, ignorers: -0.7, ignores: -1.1, ignoring: -1.7, ill: -1.8, illegal: -2.6, illiteracy: -1.9, illness: -1.7, illnesses: -2.2, imbecile: -2.2, immobilized: -1.2, immoral: -2, immoralism: -1.6, immoralist: -2.1, immoralists: -1.7, immoralities: -1.1, immorality: -0.6, immorally: -2.1, immortal: 1, immune: 1.2, impatience: -1.8, impatiens: -0.2, impatient: -1.2, impatiently: -1.7, imperfect: -1.3, impersonal: -1.3, impolite: -1.6, impolitely: -1.8, impoliteness: -1.8, impolitenesses: -2.3, importance: 1.5, importancies: 0.4, importancy: 1.4, important: 0.8, importantly: 1.3, impose: -1.2, imposed: -0.3, imposes: -0.4, imposing: -0.4, impotent: -1.1, impress: 1.9, impressed: 2.1, impresses: 2.1, impressibility: 1.2, impressible: 0.8, impressing: 2.5, impression: 0.9, impressionable: 0.2, impressionism: 0.8, impressionisms: 0.5, impressionist: 1, impressionistic: 1.5, impressionistically: 1.6, impressionists: 0.5, impressions: 0.9, impressive: 2.3, impressively: 2, impressiveness: 1.7, impressment: -0.4, impressments: 0.5, impressure: 0.6, imprisoned: -2, improve: 1.9, improved: 2.1, improvement: 2, improvements: 1.3, improver: 1.8, improvers: 1.3, improves: 1.8, improving: 1.8, inability: -1.7, inaction: -1, inadequacies: -1.7, inadequacy: -1.7, inadequate: -1.7, inadequately: -1, inadequateness: -1.7, inadequatenesses: -1.6, incapable: -1.6, incapacitated: -1.9, incensed: -2, incentive: 1.5, incentives: 1.3, incompetence: -2.3, incompetent: -2.1, inconsiderate: -1.9, inconvenience: -1.5, inconvenient: -1.4, increase: 1.3, increased: 1.1, indecision: -0.8, indecisions: -1.1, indecisive: -1, indecisively: -0.7, indecisiveness: -1.3, indecisivenesses: -0.9, indestructible: 0.6, indifference: -0.2, indifferent: -0.8, indignant: -1.8, indignation: -2.4, indoctrinate: -1.4, indoctrinated: -0.4, indoctrinates: -0.6, indoctrinating: -0.7, ineffective: -0.5, ineffectively: -1.3, ineffectiveness: -1.3, ineffectual: -1.2, ineffectuality: -1.6, ineffectually: -1.1, ineffectualness: -1.3, infatuated: 0.2, infatuation: 0.6, infected: -2.2, inferior: -1.7, inferiorities: -1.9, inferiority: -1.1, inferiorly: -2, inferiors: -0.5, inflamed: -1.4, influential: 1.9, infringement: -2.1, infuriate: -2.2, infuriated: -3, infuriates: -2.6, infuriating: -2.4, inhibin: -0.2, inhibit: -1.6, inhibited: -0.4, inhibiting: -0.4, inhibition: -1.5, inhibitions: -0.8, inhibitive: -1.4, inhibitor: -0.3, inhibitors: -1, inhibitory: -1, inhibits: -0.9, injured: -1.7, injury: -1.8, injustice: -2.7, innocence: 1.6, innocency: 1.9, innocent: 1.4, innocenter: 0.9, innocently: 1.4, innocents: 1.1, innovate: 2.2, innovates: 2, innovation: 1.6, innovative: 1.9, inquisition: -1.2, inquisitive: 0.7, insane: -1.7, insanity: -2.7, insecure: -1.8, insecurely: -1.4, insecureness: -1.8, insecurities: -1.8, insecurity: -1.8, insensitive: -0.9, insensitivity: -1.8, insignificant: -1.4, insincere: -1.8, insincerely: -1.9, insincerity: -1.4, insipid: -2, inspiration: 2.4, inspirational: 2.3, inspirationally: 2.3, inspirations: 2.1, inspirator: 1.9, inspirators: 1.2, inspiratory: 1.5, inspire: 2.7, inspired: 2.2, inspirer: 2.2, inspirers: 2, inspires: 1.9, inspiring: 1.8, inspiringly: 2.6, inspirit: 1.9, inspirited: 1.3, inspiriting: 1.8, inspiritingly: 2.1, inspirits: 0.8, insult: -2.3, insulted: -2.3, insulter: -2, insulters: -2, insulting: -2.2, insultingly: -2.3, insults: -1.8, intact: 0.8, integrity: 1.6, intellect: 2, intellection: 0.6, intellections: 0.8, intellective: 1.7, intellectively: 0.8, intellects: 1.8, intellectual: 2.3, intellectualism: 2.2, intellectualist: 2, intellectualistic: 1.3, intellectualists: 0.8, intellectualities: 1.7, intellectuality: 1.7, intellectualization: 1.5, intellectualize: 1.5, intellectualized: 1.2, intellectualizes: 1.8, intellectualizing: 0.8, intellectually: 1.4, intellectualness: 1.5, intellectuals: 1.6, intelligence: 2.1, intelligencer: 1.5, intelligencers: 1.6, intelligences: 1.6, intelligent: 2, intelligential: 1.9, intelligently: 2, intelligentsia: 1.5, intelligibility: 1.5, intelligible: 1.4, intelligibleness: 1.5, intelligibly: 1.2, intense: 0.3, interest: 2, interested: 1.7, interestedly: 1.5, interesting: 1.7, interestingly: 1.7, interestingness: 1.8, interests: 1, interrogated: -1.6, interrupt: -1.4, interrupted: -1.2, interrupter: -1.1, interrupters: -1.3, interruptible: -1.3, interrupting: -1.2, interruption: -1.5, interruptions: -1.7, interruptive: -1.4, interruptor: -1.3, interrupts: -1.3, intimidate: -0.8, intimidated: -1.9, intimidates: -1.3, intimidating: -1.9, intimidatingly: -1.1, intimidation: -1.8, intimidations: -1.4, intimidator: -1.6, intimidators: -1.6, intimidatory: -1.1, intricate: 0.6, intrigues: 0.9, invigorate: 1.9, invigorated: 0.8, invigorates: 2.1, invigorating: 2.1, invigoratingly: 2, invigoration: 1.5, invigorations: 1.2, invigorator: 1.1, invigorators: 1.2, invincible: 2.2, invite: 0.6, inviting: 1.3, invulnerable: 1.3, irate: -2.9, ironic: -0.5, irony: -0.2, irrational: -1.4, irrationalism: -1.5, irrationalist: -2.1, irrationalists: -1.5, irrationalities: -1.5, irrationality: -1.7, irrationally: -1.6, irrationals: -1.1, irresistible: 1.4, irresolute: -1.4, irresponsible: -1.9, irreversible: -0.8, irritabilities: -1.7, irritability: -1.4, irritable: -2.1, irritableness: -1.7, irritably: -1.8, irritant: -2.3, irritants: -2.1, irritate: -1.8, irritated: -2, irritates: -1.7, irritating: -2, irritatingly: -2, irritation: -2.3, irritations: -1.5, irritative: -2, isolatable: 0.2, isolate: -0.8, isolated: -1.3, isolates: -1.3, isolation: -1.7, isolationism: 0.4, isolationist: 0.7, isolations: -0.5, isolator: -0.4, isolators: -0.4, itchy: -1.1, jackass: -1.8, jackasses: -2.8, jaded: -1.6, jailed: -2.2, jaunty: 1.2, jealous: -2, jealousies: -2, jealously: -2, jealousness: -1.7, jealousy: -1.3, jeopardy: -2.1, jerk: -1.4, jerked: -0.8, jerks: -1.1, jewel: 1.5, jewels: 2, jocular: 1.2, join: 1.2, joke: 1.2, joked: 1.3, joker: 0.5, jokes: 1, jokester: 1.5, jokesters: 0.9, jokey: 1.1, joking: 0.9, jollied: 2.4, jollier: 2.4, jollies: 2, jolliest: 2.9, jollification: 2.2, jollifications: 2, jollify: 2.1, jollily: 2.7, jolliness: 2.5, jollities: 1.7, jollity: 1.8, jolly: 2.3, jollying: 2.3, jovial: 1.9, joy: 2.8, joyance: 2.3, joyed: 2.9, joyful: 2.9, joyfuller: 2.4, joyfully: 2.5, joyfulness: 2.7, joying: 2.5, joyless: -2.5, joylessly: -1.7, joylessness: -2.7, joyous: 3.1, joyously: 2.9, joyousness: 2.8, joypop: -0.2, joypoppers: -0.1, joyridden: 0.6, joyride: 1.1, joyrider: 0.7, joyriders: 1.3, joyrides: 0.8, joyriding: 0.9, joyrode: 1, joys: 2.2, joystick: 0.7, joysticks: 0.2, jubilant: 3, jumpy: -1, justice: 2.4, justifiably: 1, justified: 1.7, keen: 1.5, keened: 0.3, keener: 0.5, keeners: 0.6, keenest: 1.9, keening: -0.7, keenly: 1, keenness: 1.4, keens: 0.1, kewl: 1.3, kidding: 0.4, kill: -3.7, killdeer: -1.1, killdeers: -0.1, killdees: -0.6, killed: -3.5, killer: -3.3, killers: -3.3, killick: 0.1, killie: -0.1, killifish: -0.1, killifishes: -0.1, killing: -3.4, killingly: -2.6, killings: -3.5, killjoy: -2.1, killjoys: -1.7, killock: -0.3, killocks: -0.4, kills: -2.5, kind: 2.4, kinder: 2.2, kindly: 2.2, kindness: 2, kindnesses: 2.3, kiss: 1.8, kissable: 2, kissably: 1.9, kissed: 1.6, kisser: 1.7, kissers: 1.5, kisses: 2.3, kissing: 2.7, kissy: 1.8, kudos: 2.3, lack: -1.3, lackadaisical: -1.6, lag: -1.4, lagged: -1.2, lagging: -1.1, lags: -1.5, laidback: 0.5, lame: -1.8, lamebrain: -1.6, lamebrained: -2.5, lamebrains: -1.2, lamedh: 0.1, lamella: -0.1, lamellae: -0.1, lamellas: 0.1, lamellibranch: 0.2, lamellibranchs: -0.1, lamely: -2, lameness: -0.8, lament: -2, lamentable: -1.5, lamentableness: -1.3, lamentably: -1.5, lamentation: -1.4, lamentations: -1.9, lamented: -1.4, lamenter: -1.2, lamenters: -0.5, lamenting: -2, laments: -1.5, lamer: -1.4, lames: -1.2, lamest: -1.5, landmark: 0.3, laugh: 2.6, laughable: 0.2, laughableness: 1.2, laughably: 1.2, laughed: 2, laugher: 1.7, laughers: 1.7, laughing: 2.2, laughingly: 2.3, laughings: 1.9, laughingstocks: -1.3, laughs: 2.2, laughter: 2.2, laughters: 2.2, launched: 0.5, lawl: 1.4, lawsuit: -0.9, lawsuits: -0.6, lazier: -2.3, laziest: -2.7, lazy: -1.5, leak: -1.4, leaked: -1.3, leave: -0.2, leet: 1.3, legal: 0.5, legally: 0.4, lenient: 1.1, lethargic: -1.2, lethargy: -1.4, liabilities: -0.8, liability: -0.8, liar: -2.3, liards: -0.4, liars: -2.4, libelous: -2.1, libertarian: 0.9, libertarianism: 0.4, libertarianisms: 0.1, libertarians: 0.1, liberties: 2.3, libertinage: 0.2, libertine: -0.9, libertines: 0.4, libertinisms: 1.2, liberty: 2.4, lied: -1.6, lies: -1.8, lifesaver: 2.8, lighthearted: 1.8, like: 1.5, likeable: 2, liked: 1.8, likes: 1.8, liking: 1.7, limitation: -1.2, limited: -0.9, litigation: -0.8, litigious: -0.8, livelier: 1.7, liveliest: 2.1, livelihood: 0.8, livelihoods: 0.9, livelily: 1.8, liveliness: 1.6, livelong: 1.7, lively: 1.9, livid: -2.5, loathe: -2.2, loathed: -2.1, loathes: -1.9, loathing: -2.7, lobby: 0.1, lobbying: -0.3, lone: -1.1, lonelier: -1.4, loneliest: -2.4, loneliness: -1.8, lonelinesses: -1.5, lonely: -1.5, loneness: -1.1, loner: -1.3, loners: -0.9, lonesome: -1.5, lonesomely: -1.3, lonesomeness: -1.8, lonesomes: -1.4, longing: -0.1, longingly: 0.7, longings: 0.4, loom: -0.9, loomed: -1.1, looming: -0.5, looms: -0.6, loose: -1.3, looses: -0.6, lose: -1.7, loser: -2.4, losers: -2.4, loses: -1.3, losing: -1.6, loss: -1.3, losses: -1.7, lossy: -1.2, lost: -1.3, louse: -1.6, loused: -1, louses: -1.3, lousewort: 0.1, louseworts: -0.6, lousier: -2.2, lousiest: -2.6, lousily: -1.2, lousiness: -1.7, lousing: -1.1, lousy: -2.5, lovable: 3, love: 3.2, loved: 2.9, lovelies: 2.2, lovely: 2.8, lover: 2.8, loverly: 2.8, lovers: 2.4, loves: 2.7, loving: 2.9, lovingly: 3.2, lovingness: 2.7, low: -1.1, lowball: -0.8, lowballed: -1.5, lowballing: -0.7, lowballs: -1.2, lowborn: -0.7, lowboys: -0.6, lowbred: -2.6, lowbrow: -1.9, lowbrows: -0.6, lowdown: -0.8, lowdowns: -0.2, lowe: 0.5, lowed: -0.8, lower: -1.2, lowercase: 0.3, lowercased: -0.2, lowerclassman: -0.4, lowered: -0.5, lowering: -1, lowermost: -1.4, lowers: -0.5, lowery: -1.8, lowest: -1.6, lowing: -0.5, lowish: -0.9, lowland: -0.1, lowlander: -0.4, lowlanders: -0.3, lowlands: -0.1, lowlier: -1.7, lowliest: -1.8, lowlife: -1.5, lowlifes: -2.2, lowlight: -2, lowlights: -0.3, lowlihead: -0.3, lowliness: -1.1, lowlinesses: -1.2, lowlives: -2.1, lowly: -1, lown: 0.9, lowness: -1.3, lowrider: -0.2, lowriders: 0.1, lows: -0.8, lowse: -0.7, loyal: 2.1, loyalism: 1, loyalisms: 0.9, loyalist: 1.5, loyalists: 1.1, loyally: 2.1, loyalties: 1.9, loyalty: 2.5, luck: 2, lucked: 1.9, luckie: 1.6, luckier: 1.9, luckiest: 2.9, luckily: 2.3, luckiness: 1, lucking: 1.2, luckless: -1.3, lucks: 1.6, lucky: 1.8, ludicrous: -1.5, ludicrously: -0.2, ludicrousness: -1.9, lugubrious: -2.1, lulz: 2, lunatic: -2.2, lunatics: -1.6, lurk: -0.8, lurking: -0.5, lurks: -0.9, lying: -2.4, mad: -2.2, maddening: -2.2, madder: -1.2, maddest: -2.8, madly: -1.7, madness: -1.9, magnific: 2.3, magnifical: 2.4, magnifically: 2.4, magnification: 1, magnifications: 1.2, magnificence: 2.4, magnificences: 2.3, magnificent: 2.9, magnificently: 3.4, magnifico: 1.8, magnificoes: 1.4, mandatory: 0.3, maniac: -2.1, maniacal: -0.3, maniacally: -1.7, maniacs: -1.2, manipulated: -1.6, manipulating: -1.5, manipulation: -1.2, marvel: 1.8, marvelous: 2.9, marvels: 2, masochism: -1.6, masochisms: -1.1, masochist: -1.7, masochistic: -2.2, masochistically: -1.6, masochists: -1.2, masterpiece: 3.1, masterpieces: 2.5, matter: 0.1, matters: 0.1, mature: 1.8, meaningful: 1.3, meaningless: -1.9, medal: 2.1, mediocrity: -0.3, meditative: 1.4, meh: -0.3, melancholia: -0.5, melancholiac: -2, melancholias: -1.6, melancholic: -0.3, melancholics: -1, melancholies: -1.1, melancholy: -1.9, menace: -2.2, menaced: -1.7, mercy: 1.5, merit: 1.8, merited: 1.4, meriting: 1.1, meritocracy: 0.6, meritocrat: 0.4, meritocrats: 1.1, meritorious: 2.1, meritoriously: 1.3, meritoriousness: 1.7, merits: 1.7, merrier: 1.7, merriest: 2.7, merrily: 2.4, merriment: 2.4, merriments: 2, merriness: 2.2, merry: 2.5, merrymaker: 2.2, merrymakers: 1.7, merrymaking: 2.2, merrymakings: 2.4, merrythought: 1.1, merrythoughts: 1.6, mess: -1.5, messed: -1.4, messy: -1.5, methodical: 0.6, mindless: -1.9, miracle: 2.8, mirth: 2.6, mirthful: 2.7, mirthfully: 2, misbehave: -1.9, misbehaved: -1.6, misbehaves: -1.6, misbehaving: -1.7, mischief: -1.5, mischiefs: -0.8, miser: -1.8, miserable: -2.2, miserableness: -2.8, miserably: -2.1, miserere: -0.8, misericorde: 0.1, misericordes: -0.5, miseries: -2.7, miserliness: -2.6, miserly: -1.4, misers: -1.5, misery: -2.7, misgiving: -1.4, misinformation: -1.3, misinformed: -1.6, misinterpreted: -1.3, misleading: -1.7, misread: -1.1, misreporting: -1.5, misrepresentation: -2, miss: -0.6, missed: -1.2, misses: -0.9, missing: -1.2, mistakable: -0.8, mistake: -1.4, mistaken: -1.5, mistakenly: -1.2, mistaker: -1.6, mistakers: -1.6, mistakes: -1.5, mistaking: -1.1, misunderstand: -1.5, misunderstanding: -1.8, misunderstands: -1.3, misunderstood: -1.4, mlm: -1.4, mmk: 0.6, moan: -0.6, moaned: -0.4, moaning: -0.4, moans: -0.6, mock: -1.8, mocked: -1.3, mocker: -0.8, mockeries: -1.6, mockers: -1.3, mockery: -1.3, mocking: -1.7, mocks: -2, molest: -2.1, molestation: -1.9, molestations: -2.9, molested: -1.9, molester: -2.3, molesters: -2.2, molesting: -2.8, molests: -3.1, mongering: -0.8, monopolize: -0.8, monopolized: -0.9, monopolizes: -1.1, monopolizing: -0.5, mooch: -1.7, mooched: -1.4, moocher: -1.5, moochers: -1.9, mooches: -1.4, mooching: -1.7, moodier: -1.1, moodiest: -2.1, moodily: -1.3, moodiness: -1.4, moodinesses: -1.4, moody: -1.5, mope: -1.9, moping: -1, moron: -2.2, moronic: -2.7, moronically: -1.4, moronity: -1.1, morons: -1.3, motherfucker: -3.6, motherfucking: -2.8, motivate: 1.6, motivated: 2, motivating: 2.2, motivation: 1.4, mourn: -1.8, mourned: -1.3, mourner: -1.6, mourners: -1.8, mournful: -1.6, mournfuller: -1.9, mournfully: -1.7, mournfulness: -1.8, mourning: -1.9, mourningly: -2.3, mourns: -2.4, mumpish: -1.4, murder: -3.7, murdered: -3.4, murderee: -3.2, murderees: -3.1, murderer: -3.6, murderers: -3.3, murderess: -2.2, murderesses: -2.6, murdering: -3.3, murderous: -3.2, murderously: -3.1, murderousness: -2.9, murders: -3, n00b: -1.6, nag: -1.5, nagana: -1.7, nagged: -1.7, nagger: -1.8, naggers: -1.5, naggier: -1.4, naggiest: -2.4, nagging: -1.7, naggingly: -0.9, naggy: -1.7, nags: -1.1, nah: -0.4, naive: -1.1, nastic: 0.2, nastier: -2.3, nasties: -2.1, nastiest: -2.4, nastily: -1.9, nastiness: -1.1, nastinesses: -2.6, nasturtium: 0.4, nasturtiums: 0.1, nasty: -2.6, natural: 1.5, neat: 2, neaten: 1.2, neatened: 2, neatening: 1.3, neatens: 1.1, neater: 1, neatest: 1.7, neath: 0.2, neatherd: -0.4, neatly: 1.4, neatness: 1.3, neats: 1.1, needy: -1.4, negative: -2.7, negativity: -2.3, neglect: -2, neglected: -2.4, neglecter: -1.7, neglecters: -1.5, neglectful: -2, neglectfully: -2.1, neglectfulness: -2, neglecting: -1.7, neglects: -2.2, nerd: -1.2, nerdier: -0.2, nerdiest: 0.6, nerdish: -0.1, nerdy: -0.2, nerves: -0.4, nervous: -1.1, nervously: -0.6, nervousness: -1.2, neurotic: -1.4, neurotically: -1.8, neuroticism: -0.9, neurotics: -0.7, nice: 1.8, nicely: 1.9, niceness: 1.6, nicenesses: 2.1, nicer: 1.9, nicest: 2.2, niceties: 1.5, nicety: 1.2, nifty: 1.7, niggas: -1.4, nigger: -3.3, no: -1.2, noble: 2, noisy: -0.7, nonsense: -1.7, noob: -0.2, nosey: -0.8, notorious: -1.9, novel: 1.3, numb: -1.4, numbat: 0.2, numbed: -0.9, number: 0.3, numberable: 0.6, numbest: -1, numbfish: -0.4, numbfishes: -0.7, numbing: -1.1, numbingly: -1.3, numbles: 0.4, numbly: -1.4, numbness: -1.1, numbs: -0.7, numbskull: -2.3, numbskulls: -2.2, nurtural: 1.5, nurturance: 1.6, nurturances: 1.3, nurturant: 1.7, nurture: 1.4, nurtured: 1.9, nurturer: 1.9, nurturers: 0.8, nurtures: 1.9, nurturing: 2, nuts: -1.3, "o/\\o": 2.1, o_0: -0.1, obliterate: -2.9, obliterated: -2.1, obnoxious: -2, obnoxiously: -2.3, obnoxiousness: -2.1, obscene: -2.8, obsess: -1, obsessed: -0.7, obsesses: -1, obsessing: -1.4, obsession: -1.4, obsessional: -1.5, obsessionally: -1.3, obsessions: -0.9, obsessive: -0.9, obsessively: -0.4, obsessiveness: -1.2, obsessives: -0.7, obsolete: -1.2, obstacle: -1.5, obstacles: -1.6, obstinate: -1.2, odd: -1.3, offence: -1.2, offences: -1.4, offend: -1.2, offended: -1, offender: -1.5, offenders: -1.5, offending: -2.3, offends: -2, offense: -1, offenseless: 0.7, offenses: -1.5, offensive: -2, offensively: -2.8, offensiveness: -2.3, offensives: -0.8, offline: -0.5, okay: 0.9, okays: 2.1, ominous: -1.4, "once-in-a-lifetime": 1.8, openness: 1.4, opportune: 1.7, opportunely: 1.5, opportuneness: 1.2, opportunism: 0.4, opportunisms: 0.2, opportunist: 0.2, opportunistic: -0.1, opportunistically: 0.9, opportunists: 0.3, opportunities: 1.6, opportunity: 1.8, oppressed: -2.1, oppressive: -1.7, optimal: 1.5, optimality: 1.9, optimally: 1.3, optimisation: 1.6, optimisations: 1.8, optimise: 1.9, optimised: 1.7, optimises: 1.6, optimising: 1.7, optimism: 2.5, optimisms: 2, optimist: 2.4, optimistic: 1.3, optimistically: 2.1, optimists: 1.6, optimization: 1.6, optimizations: 0.9, optimize: 2.2, optimized: 2, optimizer: 1.5, optimizers: 2.1, optimizes: 1.8, optimizing: 2, optionless: -1.7, original: 1.3, outcry: -2.3, outgoing: 1.2, outmaneuvered: 0.5, outrage: -2.3, outraged: -2.5, outrageous: -2, outrageously: -1.2, outrageousness: -1.2, outrageousnesses: -1.3, outrages: -2.3, outraging: -2, outreach: 1.1, outstanding: 3, overjoyed: 2.7, overload: -1.5, overlooked: -0.1, overreact: -1, overreacted: -1.7, overreaction: -0.7, overreacts: -2.2, oversell: -0.9, overselling: -0.8, oversells: 0.3, oversimplification: 0.2, oversimplifies: 0.1, oversimplify: -0.6, overstatement: -1.1, overstatements: -0.7, overweight: -1.5, overwhelm: -0.7, overwhelmed: 0.2, overwhelmingly: -0.5, overwhelms: -0.8, oxymoron: -0.5, pain: -2.3, pained: -1.8, painful: -1.9, painfuller: -1.7, painfully: -2.4, painfulness: -2.7, paining: -1.7, painless: 1.2, painlessly: 1.1, painlessness: 0.4, pains: -1.8, palatable: 1.6, palatableness: 0.8, palatably: 1.1, panic: -2.3, panicked: -2, panicking: -1.9, panicky: -1.5, panicle: 0.5, panicled: 0.1, panicles: -0.2, panics: -1.9, paniculate: 0.1, panicums: -0.1, paradise: 3.2, paradox: -0.4, paranoia: -1, paranoiac: -1.3, paranoiacs: -0.7, paranoias: -1.5, paranoid: -1, paranoids: -1.6, pardon: 1.3, pardoned: 0.9, pardoning: 1.7, pardons: 1.2, parley: -0.4, partied: 1.4, partier: 1.4, partiers: 0.7, parties: 1.7, party: 1.7, partyer: 1.2, partyers: 1.1, partying: 1.6, passion: 2, passional: 1.6, passionate: 2.4, passionately: 2.4, passionateness: 2.3, passionflower: 0.3, passionflowers: 0.4, passionless: -1.9, passions: 2.2, passive: 0.8, passively: -0.7, pathetic: -2.7, pathetical: -1.2, pathetically: -1.8, pay: -0.4, peace: 2.5, peaceable: 1.7, peaceableness: 1.8, peaceably: 2, peaceful: 2.2, peacefuller: 1.9, peacefullest: 3.1, peacefully: 2.4, peacefulness: 2.1, peacekeeper: 1.6, peacekeepers: 1.6, peacekeeping: 2, peacekeepings: 1.6, peacemaker: 2, peacemakers: 2.4, peacemaking: 1.7, peacenik: 0.8, peaceniks: 0.7, peaces: 2.1, peacetime: 2.2, peacetimes: 2.1, peculiar: 0.6, peculiarities: 0.1, peculiarity: 0.6, peculiarly: -0.4, penalty: -2, pensive: 0.3, perfect: 2.7, perfecta: 1.4, perfectas: 0.6, perfected: 2.7, perfecter: 1.8, perfecters: 1.4, perfectest: 3.1, perfectibilities: 2.1, perfectibility: 1.8, perfectible: 1.5, perfecting: 2.3, perfection: 2.7, perfectionism: 1.3, perfectionist: 1.5, perfectionistic: 0.7, perfectionists: 0.1, perfections: 2.5, perfective: 1.2, perfectively: 2.1, perfectiveness: 0.9, perfectives: 0.9, perfectivity: 2.2, perfectly: 3.2, perfectness: 3, perfecto: 1.3, perfects: 1.6, peril: -1.7, perjury: -1.9, perpetrator: -2.2, perpetrators: -1, perplexed: -1.3, persecute: -2.1, persecuted: -1.3, persecutes: -1.2, persecuting: -1.5, perturbed: -1.4, perverse: -1.8, perversely: -2.2, perverseness: -2.1, perversenesses: -0.5, perversion: -1.3, perversions: -1.2, perversities: -1.1, perversity: -2.6, perversive: -2.1, pervert: -2.3, perverted: -2.5, pervertedly: -1.2, pervertedness: -1.2, perverter: -1.7, perverters: -0.6, perverting: -1, perverts: -2.8, pesky: -1.2, pessimism: -1.5, pessimisms: -2, pessimist: -1.5, pessimistic: -1.5, pessimistically: -2, pessimists: -1, petrifaction: -1.9, petrifactions: -0.3, petrification: -0.1, petrifications: -0.4, petrified: -2.5, petrifies: -2.3, petrify: -1.7, petrifying: -2.6, pettier: -0.3, pettiest: -1.3, petty: -0.8, phobia: -1.6, phobias: -2, phobic: -1.2, phobics: -1.3, picturesque: 1.6, pileup: -1.1, pique: -1.1, piqued: 0.1, piss: -1.7, pissant: -1.5, pissants: -2.5, pissed: -3.2, pisser: -2, pissers: -1.4, pisses: -1.4, pissing: -1.7, pissoir: -0.8, piteous: -1.2, pitiable: -1.1, pitiableness: -1.1, pitiably: -1.1, pitied: -1.3, pitier: -1.2, pitiers: -1.3, pities: -1.2, pitiful: -2.2, pitifuller: -1.8, pitifullest: -1.1, pitifully: -1.2, pitifulness: -1.2, pitiless: -1.8, pitilessly: -2.1, pitilessness: -0.5, pity: -1.2, pitying: -1.4, pityingly: -1, pityriasis: -0.8, play: 1.4, played: 1.4, playful: 1.9, playfully: 1.6, playfulness: 1.2, playing: 0.8, plays: 1, pleasant: 2.3, pleasanter: 1.5, pleasantest: 2.6, pleasantly: 2.1, pleasantness: 2.3, pleasantnesses: 2.3, pleasantries: 1.3, pleasantry: 2, please: 1.3, pleased: 1.9, pleaser: 1.7, pleasers: 1, pleases: 1.7, pleasing: 2.4, pleasurability: 1.9, pleasurable: 2.4, pleasurableness: 2.4, pleasurably: 2.6, pleasure: 2.7, pleasured: 2.3, pleasureless: -1.6, pleasures: 1.9, pleasuring: 2.8, poised: 1, poison: -2.5, poisoned: -2.2, poisoner: -2.7, poisoners: -3.1, poisoning: -2.8, poisonings: -2.4, poisonous: -2.7, poisonously: -2.9, poisons: -2.7, poisonwood: -1, pollute: -2.3, polluted: -2, polluter: -1.8, polluters: -2, pollutes: -2.2, poor: -2.1, poorer: -1.5, poorest: -2.5, popular: 1.8, popularise: 1.6, popularised: 1.1, popularises: 0.5, popularising: 1.2, popularities: 1.6, popularity: 2.1, popularization: 1.3, popularizations: 0.9, popularize: 1.3, popularized: 1.9, popularizer: 1.8, popularizers: 1, popularizes: 1.4, popularizing: 1.5, popularly: 1.8, positive: 2.6, positively: 2.4, positiveness: 2.3, positivenesses: 2.2, positiver: 2.3, positives: 2.4, positivest: 2.9, positivism: 1.6, positivisms: 1.8, positivist: 2, positivistic: 1.9, positivists: 1.7, positivities: 2.6, positivity: 2.3, possessive: -0.9, postpone: -0.9, postponed: -0.8, postpones: -1.1, postponing: -0.5, poverty: -2.3, powerful: 1.8, powerless: -2.2, praise: 2.6, praised: 2.2, praiser: 2, praisers: 2, praises: 2.4, praiseworthily: 1.9, praiseworthiness: 2.4, praiseworthy: 2.6, praising: 2.5, pray: 1.3, praying: 1.5, prays: 1.4, prblm: -1.6, prblms: -2.3, precious: 2.7, preciously: 2.2, preciousness: 1.9, prejudice: -2.3, prejudiced: -1.9, prejudices: -1.8, prejudicial: -2.6, prejudicially: -1.5, prejudicialness: -2.4, prejudicing: -1.8, prepared: 0.9, pressure: -1.2, pressured: -0.9, pressureless: 1, pressures: -1.3, pressuring: -1.4, pressurise: -0.6, pressurised: -0.4, pressurises: -0.8, pressurising: -0.6, pressurizations: -0.3, pressurize: -0.7, pressurized: 0.1, pressurizer: 0.1, pressurizers: -0.7, pressurizes: -0.2, pressurizing: -0.2, pretend: -0.4, pretending: 0.4, pretends: -0.4, prettied: 1.6, prettier: 2.1, pretties: 1.7, prettiest: 2.7, pretty: 2.2, prevent: 0.1, prevented: 0.1, preventing: -0.1, prevents: 0.3, prick: -1.4, pricked: -0.6, pricker: -0.3, prickers: -0.2, pricket: -0.5, prickets: 0.3, pricking: -0.9, prickle: -1, prickled: -0.2, prickles: -0.8, pricklier: -1.6, prickliest: -1.4, prickliness: -0.6, prickling: -0.8, prickly: -0.9, pricks: -0.9, pricky: -0.6, pride: 1.4, prison: -2.3, prisoner: -2.5, prisoners: -2.3, privilege: 1.5, privileged: 1.9, privileges: 1.6, privileging: 0.7, prize: 2.3, prized: 2.4, prizefight: -0.1, prizefighter: 1, prizefighters: -0.1, prizefighting: 0.4, prizefights: 0.3, prizer: 1, prizers: 0.8, prizes: 2, prizewinner: 2.3, prizewinners: 2.4, prizewinning: 3, proactive: 1.8, problem: -1.7, problematic: -1.9, problematical: -1.8, problematically: -2, problematics: -1.3, problems: -1.7, profit: 1.9, profitabilities: 1.1, profitability: 1.1, profitable: 1.9, profitableness: 2.4, profitably: 1.6, profited: 1.3, profiteer: 0.8, profiteered: -0.5, profiteering: -0.6, profiteers: 0.5, profiter: 0.7, profiterole: 0.4, profiteroles: 0.5, profiting: 1.6, profitless: -1.5, profits: 1.9, profitwise: 0.9, progress: 1.8, prominent: 1.3, promiscuities: -0.8, promiscuity: -1.8, promiscuous: -0.3, promiscuously: -1.5, promiscuousness: -0.9, promise: 1.3, promised: 1.5, promisee: 0.8, promisees: 1.1, promiser: 1.3, promisers: 1.6, promises: 1.6, promising: 1.7, promisingly: 1.2, promisor: 1, promisors: 0.4, promissory: 0.9, promote: 1.6, promoted: 1.8, promotes: 1.4, promoting: 1.5, propaganda: -1, prosecute: -1.7, prosecuted: -1.6, prosecutes: -1.8, prosecution: -2.2, prospect: 1.2, prospects: 1.2, prosperous: 2.1, protect: 1.6, protected: 1.9, protects: 1.3, protest: -1, protested: -0.5, protesters: -0.9, protesting: -1.8, protests: -0.9, proud: 2.1, prouder: 2.2, proudest: 2.6, proudful: 1.9, proudhearted: 1.4, proudly: 2.6, provoke: -1.7, provoked: -1.1, provokes: -1.3, provoking: -0.8, pseudoscience: -1.2, puke: -2.4, puked: -1.8, pukes: -1.9, puking: -1.8, pukka: 2.8, punish: -2.4, punishabilities: -1.7, punishability: -1.6, punishable: -1.9, punished: -2, punisher: -1.9, punishers: -2.6, punishes: -2.1, punishing: -2.6, punishment: -2.2, punishments: -1.8, punitive: -2.3, pushy: -1.1, puzzled: -0.7, quaking: -1.5, questionable: -1.2, questioned: -0.4, questioning: -0.4, racism: -3.1, racist: -3, racists: -2.5, radian: 0.4, radiance: 1.4, radiances: 1.1, radiancies: 0.8, radiancy: 1.4, radians: 0.2, radiant: 2.1, radiantly: 1.3, radiants: 1.2, rage: -2.6, raged: -2, ragee: -0.4, rageful: -2.8, rages: -2.1, raging: -2.4, rainy: -0.3, rancid: -2.5, rancidity: -2.6, rancidly: -2.5, rancidness: -2.6, rancidnesses: -1.6, rant: -1.4, ranter: -1.2, ranters: -1.2, rants: -1.3, rape: -3.7, raped: -3.6, raper: -3.4, rapers: -3.6, rapes: -3.5, rapeseeds: -0.5, raping: -3.8, rapist: -3.9, rapists: -3.3, rapture: 0.6, raptured: 0.9, raptures: 0.7, rapturous: 1.7, rash: -1.7, ratified: 0.6, reach: 0.1, reached: 0.4, reaches: 0.2, reaching: 0.8, readiness: 1, ready: 1.5, reassurance: 1.5, reassurances: 1.4, reassure: 1.4, reassured: 1.7, reassures: 1.5, reassuring: 1.7, reassuringly: 1.8, rebel: -0.6, rebeldom: -1.5, rebelled: -1, rebelling: -1.1, rebellion: -0.5, rebellions: -1.1, rebellious: -1.2, rebelliously: -1.8, rebelliousness: -1.2, rebels: -0.8, recession: -1.8, reckless: -1.7, recommend: 1.5, recommended: 0.8, recommends: 0.9, redeemed: 1.3, reek: -2.4, reeked: -2, reeker: -1.7, reekers: -1.5, reeking: -2, refuse: -1.2, refused: -1.2, refusing: -1.7, regret: -1.8, regretful: -1.9, regretfully: -1.9, regretfulness: -1.6, regrets: -1.5, regrettable: -2.3, regrettably: -2, regretted: -1.6, regretter: -1.6, regretters: -2, regretting: -1.7, reinvigorate: 2.3, reinvigorated: 1.9, reinvigorates: 1.8, reinvigorating: 1.7, reinvigoration: 2.2, reject: -1.7, rejected: -2.3, rejectee: -2.3, rejectees: -1.8, rejecter: -1.6, rejecters: -1.8, rejecting: -2, rejectingly: -1.7, rejection: -2.5, rejections: -2.1, rejective: -1.8, rejector: -1.8, rejects: -2.2, rejoice: 1.9, rejoiced: 2, rejoices: 2.1, rejoicing: 2.8, relax: 1.9, relaxant: 1, relaxants: 0.7, relaxation: 2.4, relaxations: 1, relaxed: 2.2, relaxedly: 1.5, relaxedness: 2, relaxer: 1.6, relaxers: 1.4, relaxes: 1.5, relaxin: 1.7, relaxing: 2.2, relaxins: 1.2, relentless: 0.2, reliant: 0.5, relief: 2.1, reliefs: 1.3, relievable: 1.1, relieve: 1.5, relieved: 1.6, relievedly: 1.4, reliever: 1.5, relievers: 1, relieves: 1.5, relieving: 1.5, relievo: 1.3, relishing: 1.6, reluctance: -1.4, reluctancy: -1.6, reluctant: -1, reluctantly: -0.4, remarkable: 2.6, remorse: -1.1, remorseful: -0.9, remorsefully: -0.7, remorsefulness: -0.7, remorseless: -2.3, remorselessly: -2, remorselessness: -2.8, repetitive: -1, repress: -1.4, repressed: -1.3, represses: -1.3, repressible: -1.5, repressing: -1.8, repression: -1.6, repressions: -1.7, repressive: -1.4, repressively: -1.7, repressiveness: -1, repressor: -1.4, repressors: -2.2, repressurize: -0.3, repressurized: 0.1, repressurizes: 0.1, repressurizing: -0.1, repulse: -2.8, repulsed: -2.2, rescue: 2.3, rescued: 1.8, rescues: 1.3, resent: -0.7, resented: -1.6, resentence: -1, resentenced: -0.8, resentences: -0.6, resentencing: 0.2, resentful: -2.1, resentfully: -1.4, resentfulness: -2, resenting: -1.2, resentment: -1.9, resentments: -1.9, resents: -1.2, resign: -1.4, resignation: -1.2, resignations: -1.2, resigned: -1, resignedly: -0.7, resignedness: -0.8, resigner: -1.2, resigners: -1, resigning: -0.9, resigns: -1.3, resolute: 1.1, resolvable: 1, resolve: 1.6, resolved: 0.7, resolvent: 0.7, resolvents: 0.4, resolver: 0.7, resolvers: 1.4, resolves: 0.7, resolving: 1.6, respect: 2.1, respectabilities: 1.8, respectability: 2.4, respectable: 1.9, respectableness: 1.2, respectably: 1.7, respected: 2.1, respecter: 2.1, respecters: 1.6, respectful: 2, respectfully: 1.7, respectfulness: 1.9, respectfulnesses: 1.3, respecting: 2.2, respective: 1.8, respectively: 1.4, respectiveness: 1.1, respects: 1.3, responsible: 1.3, responsive: 1.5, restful: 1.5, restless: -1.1, restlessly: -1.4, restlessness: -1.2, restore: 1.2, restored: 1.4, restores: 1.2, restoring: 1.2, restrict: -1.6, restricted: -1.6, restricting: -1.6, restriction: -1.1, restricts: -1.3, retained: 0.1, retard: -2.4, retarded: -2.7, retreat: 0.8, revenge: -2.4, revenged: -0.9, revengeful: -2.4, revengefully: -1.4, revengefulness: -2.2, revenger: -2.1, revengers: -2, revenges: -1.9, revered: 2.3, revive: 1.4, revives: 1.6, reward: 2.7, rewardable: 2, rewarded: 2.2, rewarder: 1.6, rewarders: 1.9, rewarding: 2.4, rewardingly: 2.4, rewards: 2.1, rich: 2.6, richened: 1.9, richening: 1, richens: 0.8, richer: 2.4, riches: 2.4, richest: 2.4, richly: 1.9, richness: 2.2, richnesses: 2.1, richweed: 0.1, richweeds: -0.1, ridicule: -2, ridiculed: -1.5, ridiculer: -1.6, ridiculers: -1.6, ridicules: -1.8, ridiculing: -1.8, ridiculous: -1.5, ridiculously: -1.4, ridiculousness: -1.1, ridiculousnesses: -1.6, rig: -0.5, rigged: -1.5, rigid: -0.5, rigidification: -1.1, rigidifications: -0.8, rigidified: -0.7, rigidifies: -0.6, rigidify: -0.3, rigidities: -0.7, rigidity: -0.7, rigidly: -0.7, rigidness: -0.3, rigorous: -1.1, rigorously: -0.4, riot: -2.6, riots: -2.3, risk: -1.1, risked: -0.9, risker: -0.8, riskier: -1.4, riskiest: -1.5, riskily: -0.7, riskiness: -1.3, riskinesses: -1.6, risking: -1.3, riskless: 1.3, risks: -1.1, risky: -0.8, rob: -2.6, robber: -2.6, robed: -0.7, robing: -1.5, robs: -2, robust: 1.4, roflcopter: 2.1, romance: 2.6, romanced: 2.2, romancer: 1.3, romancers: 1.7, romances: 1.3, romancing: 2, romantic: 1.7, romantically: 1.8, romanticise: 1.7, romanticised: 1.7, romanticises: 1.3, romanticising: 2.7, romanticism: 2.2, romanticisms: 2.1, romanticist: 1.9, romanticists: 1.3, romanticization: 1.5, romanticizations: 2, romanticize: 1.8, romanticized: 0.9, romanticizes: 1.8, romanticizing: 1.2, romantics: 1.9, rotten: -2.3, rude: -2, rudely: -2.2, rudeness: -1.5, ruder: -2.1, ruderal: -0.8, ruderals: -0.4, rudesby: -2, rudest: -2.5, ruin: -2.8, ruinable: -1.6, ruinate: -2.8, ruinated: -1.5, ruinates: -1.5, ruinating: -1.5, ruination: -2.7, ruinations: -1.6, ruined: -2.1, ruiner: -2, ruing: -1.6, ruining: -1, ruinous: -2.7, ruinously: -2.6, ruinousness: -1, ruins: -1.9, sabotage: -2.4, sad: -2.1, sadden: -2.6, saddened: -2.4, saddening: -2.2, saddens: -1.9, sadder: -2.4, saddest: -3, sadly: -1.8, sadness: -1.9, safe: 1.9, safecracker: -0.7, safecrackers: -0.9, safecracking: -0.9, safecrackings: -0.7, safeguard: 1.6, safeguarded: 1.5, safeguarding: 1.1, safeguards: 1.4, safekeeping: 1.4, safelight: 1.1, safelights: 0.8, safely: 2.2, safeness: 1.5, safer: 1.8, safes: 0.4, safest: 1.7, safeties: 1.5, safety: 1.8, safetyman: 0.3, salient: 1.1, sappy: -1, sarcasm: -0.9, sarcasms: -0.9, sarcastic: -1, sarcastically: -1.1, satisfaction: 1.9, satisfactions: 2.1, satisfactorily: 1.6, satisfactoriness: 1.5, satisfactory: 1.5, satisfiable: 1.9, satisfied: 1.8, satisfies: 1.8, satisfy: 2, satisfying: 2, satisfyingly: 1.9, savage: -2, savaged: -2, savagely: -2.2, savageness: -2.6, savagenesses: -0.9, savageries: -1.9, savagery: -2.5, savages: -2.4, save: 2.2, saved: 1.8, scam: -2.7, scams: -2.8, scandal: -1.9, scandalous: -2.4, scandals: -2.2, scapegoat: -1.7, scapegoats: -1.4, scare: -2.2, scarecrow: -0.8, scarecrows: -0.7, scared: -1.9, scaremonger: -2.1, scaremongers: -2, scarer: -1.7, scarers: -1.3, scares: -1.4, scarey: -1.7, scaring: -1.9, scary: -2.2, sceptic: -1, sceptical: -1.2, scepticism: -0.8, sceptics: -0.7, scold: -1.7, scoop: 0.6, scorn: -1.7, scornful: -1.8, scream: -1.7, screamed: -1.3, screamers: -1.5, screaming: -1.6, screams: -1.2, screw: -0.4, screwball: -0.2, screwballs: -0.3, screwbean: 0.3, screwdriver: 0.3, screwdrivers: 0.1, screwed: -2.2, "screwed up": -1.5, screwer: -1.2, screwers: -0.5, screwier: -0.6, screwiest: -2, screwiness: -0.5, screwing: -0.9, screwlike: 0.1, screws: -1, screwup: -1.7, screwups: -1, screwworm: -0.4, screwworms: -0.1, screwy: -1.4, scrumptious: 2.1, scrumptiously: 1.5, scumbag: -3.2, secure: 1.4, secured: 1.7, securely: 1.4, securement: 1.1, secureness: 1.4, securer: 1.5, securers: 0.6, secures: 1.3, securest: 2.6, securing: 1.3, securities: 1.2, securitization: 0.2, securitizations: 0.1, securitize: 0.3, securitized: 1.4, securitizes: 1.6, securitizing: 0.7, security: 1.4, sedition: -1.8, seditious: -1.7, seduced: -1.5, "self-confident": 2.5, selfish: -2.1, selfishly: -1.4, selfishness: -1.7, selfishnesses: -2, sentence: 0.3, sentenced: -0.1, sentences: 0.2, sentencing: -0.6, sentimental: 1.3, sentimentalise: 1.2, sentimentalised: 0.8, sentimentalising: 0.4, sentimentalism: 1, sentimentalisms: 0.4, sentimentalist: 0.8, sentimentalists: 0.7, sentimentalities: 0.9, sentimentality: 1.2, sentimentalization: 1.2, sentimentalizations: 0.4, sentimentalize: 0.8, sentimentalized: 1.1, sentimentalizes: 1.1, sentimentalizing: 0.8, sentimentally: 1.9, serene: 2, serious: -0.3, seriously: -0.7, seriousness: -0.2, severe: -1.6, severed: -1.5, severely: -2, severeness: -1, severer: -1.6, severest: -1.5, sexy: 2.4, shake: -0.7, shakeable: -0.3, shakedown: -1.2, shakedowns: -1.4, shaken: -0.3, shakeout: -1.3, shakeouts: -0.8, shakers: 0.3, shakeup: -0.6, shakeups: -0.5, shakier: -0.9, shakiest: -1.2, shakily: -0.7, shakiness: -0.7, shaking: -0.7, shaky: -0.9, shame: -2.1, shamed: -2.6, shamefaced: -2.3, shamefacedly: -1.9, shamefacedness: -2, shamefast: -1, shameful: -2.2, shamefully: -1.9, shamefulness: -2.4, shamefulnesses: -2.3, shameless: -1.4, shamelessly: -1.4, shamelessness: -1.4, shamelessnesses: -2, shames: -1.7, share: 1.2, shared: 1.4, shares: 1.2, sharing: 1.8, shattered: -2.1, shit: -2.6, shitake: -0.3, shitakes: -1.1, shithead: -3.1, shitheads: -2.6, shits: -2.1, shittah: 0.1, shitted: -1.7, shittier: -2.1, shittiest: -3.4, shittim: -0.6, shittimwood: -0.3, shitting: -1.8, shitty: -2.6, shock: -1.6, shockable: -1, shocked: -1.3, shocker: -0.6, shockers: -1.1, shocking: -1.7, shockingly: -0.7, shockproof: 1.3, shocks: -1.6, shook: -0.4, shoot: -1.4, "short-sighted": -1.2, "short-sightedness": -1.1, shortage: -1, shortages: -0.6, shrew: -0.9, shy: -1, shyer: -0.8, shying: -0.9, shylock: -2.1, shylocked: -0.7, shylocking: -1.5, shylocks: -1.4, shyly: -0.7, shyness: -1.3, shynesses: -1.2, shyster: -1.6, shysters: -0.9, sick: -2.3, sicken: -1.9, sickened: -2.5, sickener: -2.2, sickeners: -2.2, sickening: -2.4, sickeningly: -2.1, sickens: -2, sigh: 0.1, significance: 1.1, significant: 0.8, silencing: -0.5, sillibub: -0.1, sillier: 1, sillies: 0.8, silliest: 0.8, sillily: -0.1, sillimanite: 0.1, sillimanites: 0.2, silliness: -0.9, sillinesses: -1.2, silly: 0.1, sin: -2.6, sincere: 1.7, sincerely: 2.1, sincereness: 1.8, sincerer: 2, sincerest: 2, sincerities: 1.5, sinful: -2.6, singleminded: 1.2, sinister: -2.9, sins: -2, skeptic: -0.9, skeptical: -1.3, skeptically: -1.2, skepticism: -1, skepticisms: -1.2, skeptics: -0.4, slam: -1.6, slash: -1.1, slashed: -0.9, slashes: -0.8, slashing: -1.1, slavery: -3.8, sleeplessness: -1.6, slicker: 0.4, slickest: 0.3, sluggish: -1.7, slut: -2.8, sluts: -2.7, sluttier: -2.7, sluttiest: -3.1, sluttish: -2.2, sluttishly: -2.1, sluttishness: -2.5, sluttishnesses: -2, slutty: -2.3, smart: 1.7, smartass: -2.1, smartasses: -1.7, smarted: 0.7, smarten: 1.9, smartened: 1.5, smartening: 1.7, smartens: 1.5, smarter: 2, smartest: 3, smartie: 1.3, smarties: 1.7, smarting: -0.7, smartly: 1.5, smartness: 2, smartnesses: 1.5, smarts: 1.6, smartweed: 0.2, smartweeds: 0.1, smarty: 1.1, smear: -1.5, smilax: 0.6, smilaxes: 0.3, smile: 1.5, smiled: 2.5, smileless: -1.4, smiler: 1.7, smiles: 2.1, smiley: 1.7, smileys: 1.5, smiling: 2, smilingly: 2.3, smog: -1.2, smother: -1.8, smothered: -0.9, smothering: -1.4, smothers: -1.9, smothery: -1.1, smug: 0.8, smugger: -1, smuggest: -1.5, smuggle: -1.6, smuggled: -1.5, smuggler: -2.1, smugglers: -1.4, smuggles: -1.7, smuggling: -2.1, smugly: 0.2, smugness: -1.4, smugnesses: -1.7, sneaky: -0.9, snob: -2, snobbery: -2, snobbier: -0.7, snobbiest: -0.5, snobbily: -1.6, snobbish: -0.9, snobbishly: -1.2, snobbishness: -1.1, snobbishnesses: -1.7, snobbism: -1, snobbisms: -0.3, snobby: -1.7, snobs: -1.4, snub: -1.8, snubbed: -2, snubbing: -0.9, snubs: -2.1, sobbed: -1.9, sobbing: -1.6, sobering: -0.8, sobs: -2.5, sociabilities: 1.2, sociability: 1.1, sociable: 1.9, sociableness: 1.5, sociably: 1.6, sok: 1.3, solemn: -0.3, solemnified: -0.5, solemnifies: -0.5, solemnify: 0.3, solemnifying: 0.1, solemnities: 0.3, solemnity: -1.1, solemnization: 0.7, solemnize: 0.3, solemnized: -0.7, solemnizes: 0.6, solemnizing: -0.6, solemnly: 0.8, solid: 0.6, solidarity: 1.2, solution: 1.3, solutions: 0.7, solve: 0.8, solved: 1.1, solves: 1.1, solving: 1.4, somber: -1.8, "son-of-a-bitch": -2.7, soothe: 1.5, soothed: 0.5, soothing: 1.3, sophisticated: 2.6, sore: -1.5, sorrow: -2.4, sorrowed: -2.4, sorrower: -2.3, sorrowful: -2.2, sorrowfully: -2.3, sorrowfulness: -2.5, sorrowing: -1.7, sorrows: -1.6, sorry: -0.3, soulmate: 2.9, spam: -1.5, spammer: -2.2, spammers: -1.6, spamming: -2.1, spark: 0.9, sparkle: 1.8, sparkles: 1.3, sparkling: 1.2, special: 1.7, speculative: 0.4, spirit: 0.7, spirited: 1.3, spiritless: -1.3, spite: -2.4, spited: -2.4, spiteful: -1.9, spitefully: -2.3, spitefulness: -1.5, spitefulnesses: -2.3, spites: -1.4, splendent: 2.7, splendid: 2.8, splendidly: 2.1, splendidness: 2.3, splendiferous: 2.6, splendiferously: 1.9, splendiferousness: 1.7, splendor: 3, splendorous: 2.2, splendors: 2, splendour: 2.2, splendours: 2.2, splendrous: 2.2, sprightly: 2, squelched: -1, stab: -2.8, stabbed: -1.9, stable: 1.2, stabs: -1.9, stall: -0.8, stalled: -0.8, stalling: -0.8, stamina: 1.2, stammer: -0.9, stammered: -0.9, stammerer: -1.1, stammerers: -0.8, stammering: -1, stammers: -0.8, stampede: -1.8, stank: -1.9, startle: -1.3, startled: -0.7, startlement: -0.5, startlements: 0.2, startler: -0.8, startlers: -0.5, startles: -0.5, startling: 0.3, startlingly: -0.3, starve: -1.9, starved: -2.6, starves: -2.3, starving: -1.8, steadfast: 1, steal: -2.2, stealable: -1.7, stealer: -1.7, stealers: -2.2, stealing: -2.7, stealings: -1.9, steals: -2.3, stealth: -0.3, stealthier: -0.3, stealthiest: 0.4, stealthily: 0.1, stealthiness: 0.2, stealths: -0.3, stealthy: -0.1, stench: -2.3, stenches: -1.5, stenchful: -2.4, stenchy: -2.3, stereotype: -1.3, stereotyped: -1.2, stifled: -1.4, stimulate: 0.9, stimulated: 0.9, stimulates: 1, stimulating: 1.9, stingy: -1.6, stink: -1.7, stinkard: -2.3, stinkards: -1, stinkbug: -0.2, stinkbugs: -1, stinker: -1.5, stinkers: -1.2, stinkhorn: -0.2, stinkhorns: -0.8, stinkier: -1.5, stinkiest: -2.1, stinking: -2.4, stinkingly: -1.3, stinko: -1.5, stinkpot: -2.5, stinkpots: -0.7, stinks: -1, stinkweed: -0.4, stinkwood: -0.1, stinky: -1.5, stolen: -2.2, stop: -1.2, stopped: -0.9, stopping: -0.6, stops: -0.6, stout: 0.7, straight: 0.9, strain: -0.2, strained: -1.7, strainer: -0.8, strainers: -0.3, straining: -1.3, strains: -1.2, strange: -0.8, strangely: -1.2, strangled: -2.5, strength: 2.2, strengthen: 1.3, strengthened: 1.8, strengthener: 1.8, strengtheners: 1.4, strengthening: 2.2, strengthens: 2, strengths: 1.7, stress: -1.8, stressed: -1.4, stresses: -2, stressful: -2.3, stressfully: -2.6, stressing: -1.5, stressless: 1.6, stresslessness: 1.6, stressor: -1.8, stressors: -2.1, stricken: -2.3, strike: -0.5, strikers: -0.6, strikes: -1.5, strong: 2.3, strongbox: 0.7, strongboxes: 0.3, stronger: 1.6, strongest: 1.9, stronghold: 0.5, strongholds: 1, strongish: 1.7, strongly: 1.1, strongman: 0.7, strongmen: 0.5, strongyl: 0.6, strongyles: 0.2, strongyloidosis: -0.8, strongyls: 0.1, struck: -1, struggle: -1.3, struggled: -1.4, struggler: -1.1, strugglers: -1.4, struggles: -1.5, struggling: -1.8, stubborn: -1.7, stubborner: -1.5, stubbornest: -0.6, stubbornly: -1.4, stubbornness: -1.1, stubbornnesses: -1.5, stuck: -1, stunk: -1.6, stunned: -0.4, stunning: 1.6, stuns: 0.1, stupid: -2.4, stupider: -2.5, stupidest: -2.4, stupidities: -2, stupidity: -1.9, stupidly: -2, stupidness: -1.7, stupidnesses: -2.6, stupids: -2.3, stutter: -1, stuttered: -0.9, stutterer: -1, stutterers: -1.1, stuttering: -1.3, stutters: -1, suave: 2, submissive: -1.3, submissively: -1, submissiveness: -0.7, substantial: 0.8, subversive: -0.9, succeed: 2.2, succeeded: 1.8, succeeder: 1.2, succeeders: 1.3, succeeding: 2.2, succeeds: 2.2, success: 2.7, successes: 2.6, successful: 2.8, successfully: 2.2, successfulness: 2.7, succession: 0.8, successional: 0.9, successionally: 1.1, successions: 0.1, successive: 1.1, successively: 0.9, successiveness: 1, successor: 0.9, successors: 1.1, suck: -1.9, sucked: -2, sucker: -2.4, suckered: -2, suckering: -2.1, suckers: -2.3, sucks: -1.5, sucky: -1.9, suffer: -2.5, suffered: -2.2, sufferer: -2, sufferers: -2.4, suffering: -2.1, suffers: -2.1, suicidal: -3.5, suicide: -3.5, suing: -1.1, sulking: -1.5, sulky: -0.8, sullen: -1.7, sunnier: 2.3, sunniest: 2.4, sunny: 1.8, sunshine: 2.2, sunshiny: 1.9, super: 2.9, superb: 3.1, superior: 2.5, superiorities: 0.8, superiority: 1.4, superiorly: 2.2, superiors: 1, support: 1.7, supported: 1.3, supporter: 1.1, supporters: 1.9, supporting: 1.9, supportive: 1.2, supportiveness: 1.5, supports: 1.5, supremacies: 0.8, supremacist: 0.5, supremacists: -1, supremacy: 0.2, suprematists: 0.4, supreme: 2.6, supremely: 2.7, supremeness: 2.3, supremer: 2.3, supremest: 2.2, supremo: 1.9, supremos: 1.3, sure: 1.3, surefire: 1, surefooted: 1.9, surefootedly: 1.6, surefootedness: 1.5, surely: 1.9, sureness: 2, surer: 1.2, surest: 1.3, sureties: 1.3, surety: 1, suretyship: -0.1, suretyships: 0.4, surprisal: 1.5, surprisals: 0.7, surprise: 1.1, surprised: 0.9, surpriser: 0.6, surprisers: 0.3, surprises: 0.9, surprising: 1.1, surprisingly: 1.2, survived: 2.3, surviving: 1.2, survivor: 1.5, suspect: -1.2, suspected: -0.9, suspecting: -0.7, suspects: -1.4, suspend: -1.3, suspended: -2.1, suspicion: -1.6, suspicions: -1.5, suspicious: -1.5, suspiciously: -1.7, suspiciousness: -1.2, sux: -1.5, swear: -0.2, swearing: -1, swears: 0.2, sweet: 2, "sweet<3": 3, sweetheart: 3.3, sweethearts: 2.8, sweetie: 2.2, sweeties: 2.1, sweetly: 2.1, sweetness: 2.2, sweets: 2.2, swift: 0.8, swiftly: 1.2, swindle: -2.4, swindles: -1.5, swindling: -2, sympathetic: 2.3, sympathy: 1.5, talent: 1.8, talented: 2.3, talentless: -1.6, talents: 2, tantrum: -1.8, tantrums: -1.5, tard: -2.5, tears: -0.9, teas: 0.3, tease: -1.3, teased: -1.2, teasel: -0.1, teaseled: -0.8, teaseler: -0.8, teaselers: -1.2, teaseling: -0.4, teaselled: -0.4, teaselling: -0.2, teasels: -0.1, teaser: -1, teasers: -0.7, teases: -1.2, teashops: 0.2, teasing: -0.3, teasingly: -0.4, teaspoon: 0.2, teaspoonful: 0.2, teaspoonfuls: 0.4, teaspoons: 0.5, teaspoonsful: 0.3, temper: -1.8, tempers: -1.3, tendered: 0.5, tenderer: 0.6, tenderers: 1.2, tenderest: 1.4, tenderfeet: -0.4, tenderfoot: -0.1, tenderfoots: -0.5, tenderhearted: 1.5, tenderheartedly: 2.7, tenderheartedness: 0.7, tenderheartednesses: 2.8, tendering: 0.6, tenderization: 0.2, tenderize: 0.1, tenderized: 0.1, tenderizer: 0.4, tenderizes: 0.3, tenderizing: 0.3, tenderloin: -0.2, tenderloins: 0.4, tenderly: 1.8, tenderness: 1.8, tendernesses: 0.9, tenderometer: 0.2, tenderometers: 0.2, tenders: 0.6, tense: -1.4, tensed: -1, tensely: -1.2, tenseness: -1.5, tenser: -1.5, tenses: -0.9, tensest: -1.2, tensing: -1, tension: -1.3, tensional: -0.8, tensioned: -0.4, tensioner: -1.6, tensioners: -0.9, tensioning: -1.4, tensionless: 0.6, tensions: -1.7, terrible: -2.1, terribleness: -1.9, terriblenesses: -2.6, terribly: -2.6, terrific: 2.1, terrifically: 1.7, terrified: -3, terrifies: -2.6, terrify: -2.3, terrifying: -2.7, terror: -2.4, terrorise: -3.1, terrorised: -3.3, terrorises: -3.3, terrorising: -3, terrorism: -3.6, terrorisms: -3.2, terrorist: -3.7, terroristic: -3.3, terrorists: -3.1, terrorization: -2.7, terrorize: -3.3, terrorized: -3.1, terrorizes: -3.1, terrorizing: -3, terrorless: 0.9, terrors: -2.6, thank: 1.5, thanked: 1.9, thankful: 2.7, thankfuller: 1.9, thankfullest: 2, thankfully: 1.8, thankfulness: 2.1, thanks: 1.9, thief: -2.4, thieve: -2.2, thieved: -1.4, thieveries: -2.1, thievery: -2, thieves: -2.3, thorny: -1.1, thoughtful: 1.6, thoughtfully: 1.7, thoughtfulness: 1.9, thoughtless: -2, threat: -2.4, threaten: -1.6, threatened: -2, threatener: -1.4, threateners: -1.8, threatening: -2.4, threateningly: -2.2, threatens: -1.6, threating: -2, threats: -1.8, thrill: 1.5, thrilled: 1.9, thriller: 0.4, thrillers: 0.1, thrilling: 2.1, thrillingly: 2, thrills: 1.5, thwarted: -0.1, thwarting: -0.7, thwarts: -0.4, ticked: -1.8, timid: -1, timider: -1, timidest: -0.9, timidities: -0.7, timidity: -1.3, timidly: -0.7, timidness: -1, timorous: -0.8, tired: -1.9, tits: -0.9, tolerance: 1.2, tolerances: 0.3, tolerant: 1.1, tolerantly: 0.4, toothless: -1.4, top: 0.8, tops: 2.3, torn: -1, torture: -2.9, tortured: -2.6, torturer: -2.3, torturers: -3.5, tortures: -2.5, torturing: -3, torturous: -2.7, torturously: -2.2, totalitarian: -2.1, totalitarianism: -2.7, tough: -0.5, toughed: 0.7, toughen: 0.1, toughened: 0.1, toughening: 0.9, toughens: -0.2, tougher: 0.7, toughest: -0.3, toughie: -0.7, toughies: -0.6, toughing: -0.5, toughish: -1, toughly: -1.1, toughness: -0.2, toughnesses: 0.3, toughs: -0.8, toughy: -0.5, tout: -0.5, touted: -0.2, touting: -0.7, touts: -0.1, tragedian: -0.5, tragedians: -1, tragedienne: -0.4, tragediennes: -1.4, tragedies: -1.9, tragedy: -3.4, tragic: -2, tragical: -2.4, tragically: -2.7, tragicomedy: 0.2, tragicomic: -0.2, tragics: -2.2, tranquil: 0.2, tranquiler: 1.9, tranquilest: 1.6, tranquilities: 1.5, tranquility: 1.8, tranquilize: 0.3, tranquilized: -0.2, tranquilizer: -0.1, tranquilizers: -0.4, tranquilizes: -0.1, tranquilizing: -0.5, tranquillest: 0.8, tranquillities: 0.5, tranquillity: 1.8, tranquillized: -0.2, tranquillizer: -0.1, tranquillizers: -0.2, tranquillizes: 0.1, tranquillizing: 0.8, tranquilly: 1.2, tranquilness: 1.5, trap: -1.3, trapped: -2.4, trauma: -1.8, traumas: -2.2, traumata: -1.7, traumatic: -2.7, traumatically: -2.8, traumatise: -2.8, traumatised: -2.4, traumatises: -2.2, traumatising: -1.9, traumatism: -2.4, traumatization: -3, traumatizations: -2.2, traumatize: -2.4, traumatized: -1.7, traumatizes: -1.4, traumatizing: -2.3, travesty: -2.7, treason: -1.9, treasonous: -2.7, treasurable: 2.5, treasure: 1.2, treasured: 2.6, treasurer: 0.5, treasurers: 0.4, treasurership: 0.4, treasurerships: 1.2, treasures: 1.8, treasuries: 0.9, treasuring: 2.1, treasury: 0.8, treat: 1.7, tremble: -1.1, trembled: -1.1, trembler: -0.6, tremblers: -1, trembles: -0.1, trembling: -1.5, trembly: -1.2, tremulous: -1, trick: -0.2, tricked: -0.6, tricker: -0.9, trickeries: -1.2, trickers: -1.4, trickery: -1.1, trickie: -0.4, trickier: -0.7, trickiest: -1.2, trickily: -0.8, trickiness: -1.2, trickinesses: -0.4, tricking: 0.1, trickish: -1, trickishly: -0.7, trickishness: -0.4, trickled: 0.1, trickledown: -0.7, trickles: 0.2, trickling: -0.2, trickly: -0.3, tricks: -0.5, tricksier: -0.5, tricksiness: -1, trickster: -0.9, tricksters: -1.3, tricksy: -0.8, tricky: -0.6, trite: -0.8, triumph: 2.1, triumphal: 2, triumphalisms: 1.9, triumphalist: 0.5, triumphalists: 0.9, triumphant: 2.4, triumphantly: 2.3, triumphed: 2.2, triumphing: 2.3, triumphs: 2, trivial: -0.1, trivialise: -0.8, trivialised: -0.8, trivialises: -1.1, trivialising: -1.4, trivialities: -1, triviality: -0.5, trivialization: -0.9, trivializations: -0.7, trivialize: -1.1, trivialized: -0.6, trivializes: -1, trivializing: -0.6, trivially: 0.4, trivium: -0.3, trouble: -1.7, troubled: -2, troublemaker: -2, troublemakers: -2.2, troublemaking: -1.8, troubler: -1.4, troublers: -1.9, troubles: -2, troubleshoot: 0.8, troubleshooter: 1, troubleshooters: 0.8, troubleshooting: 0.7, troubleshoots: 0.5, troublesome: -2.3, troublesomely: -1.8, troublesomeness: -1.9, troubling: -2.5, troublous: -2.1, troublously: -2.1, trueness: 2.1, truer: 1.5, truest: 1.9, truly: 1.9, trust: 2.3, trustability: 2.1, trustable: 2.3, trustbuster: -0.5, trusted: 2.1, trustee: 1, trustees: 0.3, trusteeship: 0.5, trusteeships: 0.6, truster: 1.9, trustful: 2.1, trustfully: 1.5, trustfulness: 2.1, trustier: 1.3, trusties: 1, trustiest: 2.2, trustily: 1.6, trustiness: 1.6, trusting: 1.7, trustingly: 1.6, trustingness: 1.6, trustless: -2.3, trustor: 0.4, trustors: 1.2, trusts: 2.1, trustworthily: 2.3, trustworthiness: 1.8, trustworthy: 2.6, trusty: 2.2, truth: 1.3, truthful: 2, truthfully: 1.9, truthfulness: 1.7, truths: 1.8, tumor: -1.6, turmoil: -1.5, twat: -3.4, ugh: -1.8, uglier: -2.2, uglies: -2, ugliest: -2.8, uglification: -2.2, uglified: -1.5, uglifies: -1.8, uglify: -2.1, uglifying: -2.2, uglily: -2.1, ugliness: -2.7, uglinesses: -2.5, ugly: -2.3, unacceptable: -2, unappreciated: -1.7, unapproved: -1.4, unattractive: -1.9, unaware: -0.8, unbelievable: 0.8, unbelieving: -0.8, unbiased: -0.1, uncertain: -1.2, uncertainly: -1.4, uncertainness: -1.3, uncertainties: -1.4, uncertainty: -1.4, unclear: -1, uncomfortable: -1.6, uncomfortably: -1.7, uncompelling: -0.9, unconcerned: -0.9, unconfirmed: -0.5, uncontrollability: -1.7, uncontrollable: -1.5, uncontrollably: -1.5, uncontrolled: -1, unconvinced: -1.6, uncredited: -1, undecided: -0.9, underestimate: -1.2, underestimated: -1.1, underestimates: -1.1, undermine: -1.2, undermined: -1.5, undermines: -1.4, undermining: -1.5, undeserving: -1.9, undesirable: -1.9, unease: -1.7, uneasier: -1.4, uneasiest: -2.1, uneasily: -1.4, uneasiness: -1.6, uneasinesses: -1.8, uneasy: -1.6, unemployment: -1.9, unequal: -1.4, unequaled: 0.5, unethical: -2.3, unfair: -2.1, unfocused: -1.7, unfortunate: -2, unfortunately: -1.4, unfortunates: -1.9, unfriendly: -1.5, unfulfilled: -1.8, ungrateful: -2, ungratefully: -1.8, ungratefulness: -1.6, unhappier: -2.4, unhappiest: -2.5, unhappily: -1.9, unhappiness: -2.4, unhappinesses: -2.2, unhappy: -1.8, unhealthy: -2.4, unified: 1.6, unimportant: -1.3, unimpressed: -1.4, unimpressive: -1.4, unintelligent: -2, uninvolved: -2.2, uninvolving: -2, united: 1.8, unjust: -2.3, unkind: -1.6, unlovable: -2.7, unloved: -1.9, unlovelier: -1.9, unloveliest: -1.9, unloveliness: -2, unlovely: -2.1, unloving: -2.3, unmatched: -0.3, unmotivated: -1.4, unpleasant: -2.1, unprofessional: -2.3, unprotected: -1.5, unresearched: -1.1, unsatisfied: -1.7, unsavory: -1.9, unsecured: -1.6, unsettled: -1.3, unsophisticated: -1.2, unstable: -1.5, unstoppable: -0.8, unsuccessful: -1.5, unsuccessfully: -1.7, unsupported: -1.7, unsure: -1, unsurely: -1.3, untarnished: 1.6, unwanted: -0.9, unwelcome: -1.7, unworthy: -2, upset: -1.6, upsets: -1.5, upsetter: -1.9, upsetters: -2, upsetting: -2.1, uptight: -1.6, uptightness: -1.2, urgent: 0.8, useful: 1.9, usefully: 1.8, usefulness: 1.2, useless: -1.8, uselessly: -1.5, uselessness: -1.6, "v.v": -2.9, vague: -0.4, vain: -1.8, validate: 1.5, validated: 0.9, validates: 1.4, validating: 1.4, valuable: 2.1, valuableness: 1.7, valuables: 2.1, valuably: 2.3, value: 1.4, valued: 1.9, values: 1.7, valuing: 1.4, vanity: -0.9, verdict: 0.6, verdicts: 0.3, vested: 0.6, vexation: -1.9, vexing: -2, vibrant: 2.4, vicious: -1.5, viciously: -1.3, viciousness: -2.4, viciousnesses: -0.6, victim: -1.1, victimhood: -2, victimhoods: -0.9, victimise: -1.1, victimised: -1.5, victimises: -1.2, victimising: -2.5, victimization: -2.3, victimizations: -1.5, victimize: -2.5, victimized: -1.8, victimizer: -1.8, victimizers: -1.6, victimizes: -1.5, victimizing: -2.6, victimless: 0.6, victimologies: -0.6, victimologist: -0.5, victimologists: -0.4, victimology: 0.3, victims: -1.3, vigilant: 0.7, vigor: 1.1, vigorish: -0.4, vigorishes: 0.4, vigoroso: 1.5, vigorously: 0.5, vigorousness: 0.4, vigors: 1, vigour: 0.9, vigours: 0.4, vile: -3.1, villain: -2.6, villainess: -2.9, villainesses: -2, villainies: -2.3, villainous: -2, villainously: -2.9, villainousness: -2.7, villains: -3.4, villainy: -2.6, vindicate: 0.3, vindicated: 1.8, vindicates: 1.6, vindicating: -1.1, violate: -2.2, violated: -2.4, violater: -2.6, violaters: -2.4, violates: -2.3, violating: -2.5, violation: -2.2, violations: -2.4, violative: -2.4, violator: -2.4, violators: -1.9, violence: -3.1, violent: -2.9, violently: -2.8, virtue: 1.8, virtueless: -1.4, virtues: 1.5, virtuosa: 1.7, virtuosas: 1.8, virtuose: 1, virtuosi: 0.9, virtuosic: 2.2, virtuosity: 2.1, virtuoso: 2, virtuosos: 1.8, virtuous: 2.4, virtuously: 1.8, virtuousness: 2, virulent: -2.7, vision: 1, visionary: 2.4, visioning: 1.1, visions: 0.9, vital: 1.2, vitalise: 1.1, vitalised: 0.6, vitalises: 1.1, vitalising: 2.1, vitalism: 0.2, vitalist: 0.3, vitalists: 0.3, vitalities: 1.2, vitality: 1.3, vitalization: 1.6, vitalizations: 0.8, vitalize: 1.6, vitalized: 1.5, vitalizes: 1.4, vitalizing: 1.3, vitally: 1.1, vitals: 1.1, vitamin: 1.2, vitriolic: -2.1, vivacious: 1.8, vociferous: -0.8, vulnerabilities: -0.6, vulnerability: -0.9, vulnerable: -0.9, vulnerableness: -1.1, vulnerably: -1.2, vulture: -2, vultures: -1.3, w00t: 2.2, walkout: -1.3, walkouts: -0.7, wanker: -2.5, want: 0.3, war: -2.9, warfare: -1.2, warfares: -1.8, warm: 0.9, warmblooded: 0.2, warmed: 1.1, warmer: 1.2, warmers: 1, warmest: 1.7, warmhearted: 1.8, warmheartedness: 2.7, warming: 0.6, warmish: 1.4, warmly: 1.7, warmness: 1.5, warmonger: -2.9, warmongering: -2.5, warmongers: -2.8, warmouth: 0.4, warmouths: -0.8, warms: 1.1, warmth: 2, warmup: 0.4, warmups: 0.8, warn: -0.4, warned: -1.1, warning: -1.4, warnings: -1.2, warns: -0.4, warred: -2.4, warring: -1.9, wars: -2.6, warsaw: -0.1, warsaws: -0.2, warship: -0.7, warships: -0.5, warstle: 0.1, waste: -1.8, wasted: -2.2, wasting: -1.7, wavering: -0.6, weak: -1.9, weaken: -1.8, weakened: -1.3, weakener: -1.6, weakeners: -1.3, weakening: -1.3, weakens: -1.3, weaker: -1.9, weakest: -2.3, weakfish: -0.2, weakfishes: -0.6, weakhearted: -1.6, weakish: -1.2, weaklier: -1.5, weakliest: -2.1, weakling: -1.3, weaklings: -1.4, weakly: -1.8, weakness: -1.8, weaknesses: -1.5, weakside: -1.1, wealth: 2.2, wealthier: 2.2, wealthiest: 2.2, wealthily: 2, wealthiness: 2.4, wealthy: 1.5, weapon: -1.2, weaponed: -1.4, weaponless: 0.1, weaponry: -0.9, weapons: -1.9, weary: -1.1, weep: -2.7, weeper: -1.9, weepers: -1.1, weepie: -0.4, weepier: -1.8, weepies: -1.6, weepiest: -2.4, weeping: -1.9, weepings: -1.9, weeps: -1.4, weepy: -1.3, weird: -0.7, weirder: -0.5, weirdest: -0.9, weirdie: -1.3, weirdies: -1, weirdly: -1.2, weirdness: -0.9, weirdnesses: -0.7, weirdo: -1.8, weirdoes: -1.3, weirdos: -1.1, weirds: -0.6, weirdy: -0.9, welcome: 2, welcomed: 1.4, welcomely: 1.9, welcomeness: 2, welcomer: 1.4, welcomers: 1.9, welcomes: 1.7, welcoming: 1.9, well: 1.1, welladay: 0.3, wellaway: -0.8, wellborn: 1.8, welldoer: 2.5, welldoers: 1.6, welled: 0.4, wellhead: 0.1, wellheads: 0.5, wellhole: -0.1, wellies: 0.4, welling: 1.6, wellness: 1.9, wells: 1, wellsite: 0.5, wellspring: 1.5, wellsprings: 1.4, welly: 0.2, wept: -2, whimsical: 0.3, whine: -1.5, whined: -0.9, whiner: -1.2, whiners: -0.6, whines: -1.8, whiney: -1.3, whining: -0.9, whitewash: 0.1, whore: -3.3, whored: -2.8, whoredom: -2.1, whoredoms: -2.4, whorehouse: -1.1, whorehouses: -1.9, whoremaster: -1.9, whoremasters: -1.5, whoremonger: -2.6, whoremongers: -2, whores: -3, whoreson: -2.2, whoresons: -2.5, wicked: -2.4, wickeder: -2.2, wickedest: -2.9, wickedly: -2.1, wickedness: -2.1, wickednesses: -2.2, widowed: -2.1, willingness: 1.1, wimp: -1.4, wimpier: -1, wimpiest: -0.9, wimpiness: -1.2, wimpish: -1.6, wimpishness: -0.2, wimple: -0.2, wimples: -0.3, wimps: -1, wimpy: -0.9, win: 2.8, winnable: 1.8, winned: 1.8, winner: 2.8, winners: 2.1, winning: 2.4, winningly: 2.3, winnings: 2.5, winnow: -0.3, winnower: -0.1, winnowers: -0.2, winnowing: -0.1, winnows: -0.2, wins: 2.7, wisdom: 2.4, wise: 2.1, wiseacre: -1.2, wiseacres: -0.1, wiseass: -1.8, wiseasses: -1.5, wisecrack: -0.1, wisecracked: -0.5, wisecracker: -0.1, wisecrackers: 0.1, wisecracking: -0.6, wisecracks: -0.3, wised: 1.5, wiseguys: 0.3, wiselier: 0.9, wiseliest: 1.6, wisely: 1.8, wiseness: 1.9, wisenheimer: -1, wisenheimers: -1.4, wisents: 0.4, wiser: 1.2, wises: 1.3, wisest: 2.1, wisewomen: 1.3, wish: 1.7, wishes: 0.6, wishing: 0.9, witch: -1.5, withdrawal: 0.1, woe: -1.8, woebegone: -2.6, woebegoneness: -1.1, woeful: -1.9, woefully: -1.7, woefulness: -2.1, woes: -1.9, woesome: -1.2, won: 2.7, wonderful: 2.7, wonderfully: 2.9, wonderfulness: 2.9, woo: 2.1, woohoo: 2.3, woot: 1.8, worn: -1.2, worried: -1.2, worriedly: -2, worrier: -1.8, worriers: -1.7, worries: -1.8, worriment: -1.5, worriments: -1.9, worrisome: -1.7, worrisomely: -2, worrisomeness: -1.9, worrit: -2.1, worrits: -1.2, worry: -1.9, worrying: -1.4, worrywart: -1.8, worrywarts: -1.5, worse: -2.1, worsen: -2.3, worsened: -1.9, worsening: -2, worsens: -2.1, worser: -2, worship: 1.2, worshiped: 2.4, worshiper: 1, worshipers: 0.9, worshipful: 0.7, worshipfully: 1.1, worshipfulness: 1.6, worshiping: 1, worshipless: -0.6, worshipped: 2.7, worshipper: 0.6, worshippers: 0.8, worshipping: 1.6, worships: 1.4, worst: -3.1, worth: 0.9, worthless: -1.9, worthwhile: 1.4, worthy: 1.9, wow: 2.8, wowed: 2.6, wowing: 2.5, wows: 2, wowser: -1.1, wowsers: 1, wrathful: -2.7, wreck: -1.9, wrong: -2.1, wronged: -1.9, yay: 2.4, yeah: 1.2, yearning: 0.5, yeees: 1.7, yep: 1.2, yes: 1.7, youthful: 1.3, yucky: -1.8, yummy: 2.4, zealot: -1.9, zealots: -0.8, zealous: 0.5, "{:": 1.8, "|-0": -1.2, "|-:": -0.8, "|-:>": -1.6, "|-o": -1.2, "|:": -0.5, "|;-)": 2.2, "|=": -0.4, "|^:": -1.1, "|o:": -0.9, "||-:": -2.3, "}:": -2.1, "}:(": -2, "}:)": 0.4, "}:-(": -2.1, "}:-)": 0.3 }, emojis: { "\u{1F600}": "grinning face", "\u{1F601}": "beaming face with smiling eyes", "\u{1F602}": "face with tears of joy", "\u{1F923}": "rolling on the floor laughing", "\u{1F603}": "grinning face with big eyes", "\u{1F604}": "grinning face with smiling eyes", "\u{1F605}": "grinning face with sweat", "\u{1F606}": "grinning squinting face", "\u{1F609}": "winking face", "\u{1F60A}": "smiling face with smiling eyes", "\u{1F60B}": "face savoring food", "\u{1F60E}": "smiling face with sunglasses", "\u{1F60D}": "smiling face with heart-eyes", "\u{1F618}": "face blowing a kiss", "\u{1F970}": "smiling face with 3 hearts", "\u{1F617}": "kissing face", "\u{1F619}": "kissing face with smiling eyes", "\u{1F61A}": "kissing face with closed eyes", "\u263A\uFE0F": "smiling face", "\u263A": "smiling face", "\u{1F642}": "slightly smiling face", "\u{1F917}": "hugging face", "\u{1F929}": "star-struck", "\u{1F914}": "thinking face", "\u{1F928}": "face with raised eyebrow", "\u{1F610}": "neutral face", "\u{1F611}": "expressionless face", "\u{1F636}": "face without mouth", "\u{1F644}": "face with rolling eyes", "\u{1F60F}": "smirking face", "\u{1F623}": "persevering face", "\u{1F625}": "sad but relieved face", "\u{1F62E}": "face with open mouth", "\u{1F910}": "zipper-mouth face", "\u{1F62F}": "hushed face", "\u{1F62A}": "sleepy face", "\u{1F62B}": "tired face", "\u{1F634}": "sleeping face", "\u{1F60C}": "relieved face", "\u{1F61B}": "face with tongue", "\u{1F61C}": "winking face with tongue", "\u{1F61D}": "squinting face with tongue", "\u{1F924}": "drooling face", "\u{1F612}": "unamused face", "\u{1F613}": "downcast face with sweat", "\u{1F614}": "pensive face", "\u{1F615}": "confused face", "\u{1F643}": "upside-down face", "\u{1F911}": "money-mouth face", "\u{1F632}": "astonished face", "\u2639\uFE0F": "frowning face", "\u2639": "frowning face", "\u{1F641}": "slightly frowning face", "\u{1F616}": "confounded face", "\u{1F61E}": "disappointed face", "\u{1F61F}": "worried face", "\u{1F624}": "face with steam from nose", "\u{1F622}": "crying face", "\u{1F62D}": "loudly crying face", "\u{1F626}": "frowning face with open mouth", "\u{1F627}": "anguished face", "\u{1F628}": "fearful face", "\u{1F629}": "weary face", "\u{1F92F}": "exploding head", "\u{1F62C}": "grimacing face", "\u{1F630}": "anxious face with sweat", "\u{1F631}": "face screaming in fear", "\u{1F975}": "hot face", "\u{1F976}": "cold face", "\u{1F633}": "flushed face", "\u{1F92A}": "zany face", "\u{1F635}": "dizzy face", "\u{1F621}": "pouting face", "\u{1F620}": "angry face", "\u{1F92C}": "face with symbols on mouth", "\u{1F637}": "face with medical mask", "\u{1F912}": "face with thermometer", "\u{1F915}": "face with head-bandage", "\u{1F922}": "nauseated face", "\u{1F92E}": "face vomiting", "\u{1F927}": "sneezing face", "\u{1F607}": "smiling face with halo", "\u{1F920}": "cowboy hat face", "\u{1F973}": "partying face", "\u{1F974}": "woozy face", "\u{1F97A}": "pleading face", "\u{1F925}": "lying face", "\u{1F92B}": "shushing face", "\u{1F92D}": "face with hand over mouth", "\u{1F9D0}": "face with monocle", "\u{1F913}": "nerd face", "\u{1F608}": "smiling face with horns", "\u{1F47F}": "angry face with horns", "\u{1F921}": "clown face", "\u{1F479}": "ogre", "\u{1F47A}": "goblin", "\u{1F480}": "skull", "\u2620\uFE0F": "skull and crossbones", "\u2620": "skull and crossbones", "\u{1F47B}": "ghost", "\u{1F47D}": "alien", "\u{1F47E}": "alien monster", "\u{1F916}": "robot face", "\u{1F4A9}": "pile of poo", "\u{1F63A}": "grinning cat face", "\u{1F638}": "grinning cat face with smiling eyes", "\u{1F639}": "cat face with tears of joy", "\u{1F63B}": "smiling cat face with heart-eyes", "\u{1F63C}": "cat face with wry smile", "\u{1F63D}": "kissing cat face", "\u{1F640}": "weary cat face", "\u{1F63F}": "crying cat face", "\u{1F63E}": "pouting cat face", "\u{1F648}": "see-no-evil monkey", "\u{1F649}": "hear-no-evil monkey", "\u{1F64A}": "speak-no-evil monkey", "\u{1F3FB}": "light skin tone", "\u{1F3FC}": "medium-light skin tone", "\u{1F3FD}": "medium skin tone", "\u{1F3FE}": "medium-dark skin tone", "\u{1F3FF}": "dark skin tone", "\u{1F476}": "baby", "\u{1F476}\u{1F3FB}": "baby: light skin tone", "\u{1F476}\u{1F3FC}": "baby: medium-light skin tone", "\u{1F476}\u{1F3FD}": "baby: medium skin tone", "\u{1F476}\u{1F3FE}": "baby: medium-dark skin tone", "\u{1F476}\u{1F3FF}": "baby: dark skin tone", "\u{1F9D2}": "child", "\u{1F9D2}\u{1F3FB}": "child: light skin tone", "\u{1F9D2}\u{1F3FC}": "child: medium-light skin tone", "\u{1F9D2}\u{1F3FD}": "child: medium skin tone", "\u{1F9D2}\u{1F3FE}": "child: medium-dark skin tone", "\u{1F9D2}\u{1F3FF}": "child: dark skin tone", "\u{1F466}": "boy", "\u{1F466}\u{1F3FB}": "boy: light skin tone", "\u{1F466}\u{1F3FC}": "boy: medium-light skin tone", "\u{1F466}\u{1F3FD}": "boy: medium skin tone", "\u{1F466}\u{1F3FE}": "boy: medium-dark skin tone", "\u{1F466}\u{1F3FF}": "boy: dark skin tone", "\u{1F467}": "girl", "\u{1F467}\u{1F3FB}": "girl: light skin tone", "\u{1F467}\u{1F3FC}": "girl: medium-light skin tone", "\u{1F467}\u{1F3FD}": "girl: medium skin tone", "\u{1F467}\u{1F3FE}": "girl: medium-dark skin tone", "\u{1F467}\u{1F3FF}": "girl: dark skin tone", "\u{1F9D1}": "adult", "\u{1F9D1}\u{1F3FB}": "adult: light skin tone", "\u{1F9D1}\u{1F3FC}": "adult: medium-light skin tone", "\u{1F9D1}\u{1F3FD}": "adult: medium skin tone", "\u{1F9D1}\u{1F3FE}": "adult: medium-dark skin tone", "\u{1F9D1}\u{1F3FF}": "adult: dark skin tone", "\u{1F468}": "man", "\u{1F468}\u{1F3FB}": "man: light skin tone", "\u{1F468}\u{1F3FC}": "man: medium-light skin tone", "\u{1F468}\u{1F3FD}": "man: medium skin tone", "\u{1F468}\u{1F3FE}": "man: medium-dark skin tone", "\u{1F468}\u{1F3FF}": "man: dark skin tone", "\u{1F469}": "woman", "\u{1F469}\u{1F3FB}": "woman: light skin tone", "\u{1F469}\u{1F3FC}": "woman: medium-light skin tone", "\u{1F469}\u{1F3FD}": "woman: medium skin tone", "\u{1F469}\u{1F3FE}": "woman: medium-dark skin tone", "\u{1F469}\u{1F3FF}": "woman: dark skin tone", "\u{1F9D3}": "older adult", "\u{1F9D3}\u{1F3FB}": "older adult: light skin tone", "\u{1F9D3}\u{1F3FC}": "older adult: medium-light skin tone", "\u{1F9D3}\u{1F3FD}": "older adult: medium skin tone", "\u{1F9D3}\u{1F3FE}": "older adult: medium-dark skin tone", "\u{1F9D3}\u{1F3FF}": "older adult: dark skin tone", "\u{1F474}": "old man", "\u{1F474}\u{1F3FB}": "old man: light skin tone", "\u{1F474}\u{1F3FC}": "old man: medium-light skin tone", "\u{1F474}\u{1F3FD}": "old man: medium skin tone", "\u{1F474}\u{1F3FE}": "old man: medium-dark skin tone", "\u{1F474}\u{1F3FF}": "old man: dark skin tone", "\u{1F475}": "old woman", "\u{1F475}\u{1F3FB}": "old woman: light skin tone", "\u{1F475}\u{1F3FC}": "old woman: medium-light skin tone", "\u{1F475}\u{1F3FD}": "old woman: medium skin tone", "\u{1F475}\u{1F3FE}": "old woman: medium-dark skin tone", "\u{1F475}\u{1F3FF}": "old woman: dark skin tone", "\u{1F468}\u200D\u2695\uFE0F": "man health worker", "\u{1F468}\u200D\u2695": "man health worker", "\u{1F468}\u{1F3FB}\u200D\u2695\uFE0F": "man health worker: light skin tone", "\u{1F468}\u{1F3FB}\u200D\u2695": "man health worker: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u2695\uFE0F": "man health worker: medium-light skin tone", "\u{1F468}\u{1F3FC}\u200D\u2695": "man health worker: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u2695\uFE0F": "man health worker: medium skin tone", "\u{1F468}\u{1F3FD}\u200D\u2695": "man health worker: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u2695\uFE0F": "man health worker: medium-dark skin tone", "\u{1F468}\u{1F3FE}\u200D\u2695": "man health worker: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u2695\uFE0F": "man health worker: dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u2695": "man health worker: dark skin tone", "\u{1F469}\u200D\u2695\uFE0F": "woman health worker", "\u{1F469}\u200D\u2695": "woman health worker", "\u{1F469}\u{1F3FB}\u200D\u2695\uFE0F": "woman health worker: light skin tone", "\u{1F469}\u{1F3FB}\u200D\u2695": "woman health worker: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u2695\uFE0F": "woman health worker: medium-light skin tone", "\u{1F469}\u{1F3FC}\u200D\u2695": "woman health worker: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u2695\uFE0F": "woman health worker: medium skin tone", "\u{1F469}\u{1F3FD}\u200D\u2695": "woman health worker: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u2695\uFE0F": "woman health worker: medium-dark skin tone", "\u{1F469}\u{1F3FE}\u200D\u2695": "woman health worker: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u2695\uFE0F": "woman health worker: dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u2695": "woman health worker: dark skin tone", "\u{1F468}\u200D\u{1F393}": "man student", "\u{1F468}\u{1F3FB}\u200D\u{1F393}": "man student: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F393}": "man student: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F393}": "man student: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F393}": "man student: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F393}": "man student: dark skin tone", "\u{1F469}\u200D\u{1F393}": "woman student", "\u{1F469}\u{1F3FB}\u200D\u{1F393}": "woman student: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F393}": "woman student: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F393}": "woman student: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F393}": "woman student: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F393}": "woman student: dark skin tone", "\u{1F468}\u200D\u{1F3EB}": "man teacher", "\u{1F468}\u{1F3FB}\u200D\u{1F3EB}": "man teacher: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F3EB}": "man teacher: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F3EB}": "man teacher: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F3EB}": "man teacher: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F3EB}": "man teacher: dark skin tone", "\u{1F469}\u200D\u{1F3EB}": "woman teacher", "\u{1F469}\u{1F3FB}\u200D\u{1F3EB}": "woman teacher: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F3EB}": "woman teacher: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F3EB}": "woman teacher: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F3EB}": "woman teacher: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F3EB}": "woman teacher: dark skin tone", "\u{1F468}\u200D\u2696\uFE0F": "man judge", "\u{1F468}\u200D\u2696": "man judge", "\u{1F468}\u{1F3FB}\u200D\u2696\uFE0F": "man judge: light skin tone", "\u{1F468}\u{1F3FB}\u200D\u2696": "man judge: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u2696\uFE0F": "man judge: medium-light skin tone", "\u{1F468}\u{1F3FC}\u200D\u2696": "man judge: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u2696\uFE0F": "man judge: medium skin tone", "\u{1F468}\u{1F3FD}\u200D\u2696": "man judge: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u2696\uFE0F": "man judge: medium-dark skin tone", "\u{1F468}\u{1F3FE}\u200D\u2696": "man judge: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u2696\uFE0F": "man judge: dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u2696": "man judge: dark skin tone", "\u{1F469}\u200D\u2696\uFE0F": "woman judge", "\u{1F469}\u200D\u2696": "woman judge", "\u{1F469}\u{1F3FB}\u200D\u2696\uFE0F": "woman judge: light skin tone", "\u{1F469}\u{1F3FB}\u200D\u2696": "woman judge: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u2696\uFE0F": "woman judge: medium-light skin tone", "\u{1F469}\u{1F3FC}\u200D\u2696": "woman judge: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u2696\uFE0F": "woman judge: medium skin tone", "\u{1F469}\u{1F3FD}\u200D\u2696": "woman judge: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u2696\uFE0F": "woman judge: medium-dark skin tone", "\u{1F469}\u{1F3FE}\u200D\u2696": "woman judge: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u2696\uFE0F": "woman judge: dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u2696": "woman judge: dark skin tone", "\u{1F468}\u200D\u{1F33E}": "man farmer", "\u{1F468}\u{1F3FB}\u200D\u{1F33E}": "man farmer: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F33E}": "man farmer: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F33E}": "man farmer: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F33E}": "man farmer: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F33E}": "man farmer: dark skin tone", "\u{1F469}\u200D\u{1F33E}": "woman farmer", "\u{1F469}\u{1F3FB}\u200D\u{1F33E}": "woman farmer: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F33E}": "woman farmer: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F33E}": "woman farmer: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F33E}": "woman farmer: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F33E}": "woman farmer: dark skin tone", "\u{1F468}\u200D\u{1F373}": "man cook", "\u{1F468}\u{1F3FB}\u200D\u{1F373}": "man cook: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F373}": "man cook: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F373}": "man cook: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F373}": "man cook: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F373}": "man cook: dark skin tone", "\u{1F469}\u200D\u{1F373}": "woman cook", "\u{1F469}\u{1F3FB}\u200D\u{1F373}": "woman cook: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F373}": "woman cook: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F373}": "woman cook: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F373}": "woman cook: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F373}": "woman cook: dark skin tone", "\u{1F468}\u200D\u{1F527}": "man mechanic", "\u{1F468}\u{1F3FB}\u200D\u{1F527}": "man mechanic: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F527}": "man mechanic: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F527}": "man mechanic: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F527}": "man mechanic: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F527}": "man mechanic: dark skin tone", "\u{1F469}\u200D\u{1F527}": "woman mechanic", "\u{1F469}\u{1F3FB}\u200D\u{1F527}": "woman mechanic: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F527}": "woman mechanic: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F527}": "woman mechanic: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F527}": "woman mechanic: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F527}": "woman mechanic: dark skin tone", "\u{1F468}\u200D\u{1F3ED}": "man factory worker", "\u{1F468}\u{1F3FB}\u200D\u{1F3ED}": "man factory worker: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F3ED}": "man factory worker: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F3ED}": "man factory worker: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F3ED}": "man factory worker: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F3ED}": "man factory worker: dark skin tone", "\u{1F469}\u200D\u{1F3ED}": "woman factory worker", "\u{1F469}\u{1F3FB}\u200D\u{1F3ED}": "woman factory worker: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F3ED}": "woman factory worker: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F3ED}": "woman factory worker: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F3ED}": "woman factory worker: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F3ED}": "woman factory worker: dark skin tone", "\u{1F468}\u200D\u{1F4BC}": "man office worker", "\u{1F468}\u{1F3FB}\u200D\u{1F4BC}": "man office worker: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F4BC}": "man office worker: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F4BC}": "man office worker: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F4BC}": "man office worker: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F4BC}": "man office worker: dark skin tone", "\u{1F469}\u200D\u{1F4BC}": "woman office worker", "\u{1F469}\u{1F3FB}\u200D\u{1F4BC}": "woman office worker: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F4BC}": "woman office worker: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F4BC}": "woman office worker: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F4BC}": "woman office worker: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F4BC}": "woman office worker: dark skin tone", "\u{1F468}\u200D\u{1F52C}": "man scientist", "\u{1F468}\u{1F3FB}\u200D\u{1F52C}": "man scientist: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F52C}": "man scientist: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F52C}": "man scientist: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F52C}": "man scientist: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F52C}": "man scientist: dark skin tone", "\u{1F469}\u200D\u{1F52C}": "woman scientist", "\u{1F469}\u{1F3FB}\u200D\u{1F52C}": "woman scientist: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F52C}": "woman scientist: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F52C}": "woman scientist: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F52C}": "woman scientist: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F52C}": "woman scientist: dark skin tone", "\u{1F468}\u200D\u{1F4BB}": "man technologist", "\u{1F468}\u{1F3FB}\u200D\u{1F4BB}": "man technologist: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F4BB}": "man technologist: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F4BB}": "man technologist: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F4BB}": "man technologist: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F4BB}": "man technologist: dark skin tone", "\u{1F469}\u200D\u{1F4BB}": "woman technologist", "\u{1F469}\u{1F3FB}\u200D\u{1F4BB}": "woman technologist: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F4BB}": "woman technologist: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F4BB}": "woman technologist: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F4BB}": "woman technologist: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F4BB}": "woman technologist: dark skin tone", "\u{1F468}\u200D\u{1F3A4}": "man singer", "\u{1F468}\u{1F3FB}\u200D\u{1F3A4}": "man singer: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F3A4}": "man singer: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F3A4}": "man singer: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F3A4}": "man singer: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F3A4}": "man singer: dark skin tone", "\u{1F469}\u200D\u{1F3A4}": "woman singer", "\u{1F469}\u{1F3FB}\u200D\u{1F3A4}": "woman singer: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F3A4}": "woman singer: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F3A4}": "woman singer: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F3A4}": "woman singer: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F3A4}": "woman singer: dark skin tone", "\u{1F468}\u200D\u{1F3A8}": "man artist", "\u{1F468}\u{1F3FB}\u200D\u{1F3A8}": "man artist: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F3A8}": "man artist: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F3A8}": "man artist: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F3A8}": "man artist: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F3A8}": "man artist: dark skin tone", "\u{1F469}\u200D\u{1F3A8}": "woman artist", "\u{1F469}\u{1F3FB}\u200D\u{1F3A8}": "woman artist: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F3A8}": "woman artist: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F3A8}": "woman artist: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F3A8}": "woman artist: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F3A8}": "woman artist: dark skin tone", "\u{1F468}\u200D\u2708\uFE0F": "man pilot", "\u{1F468}\u200D\u2708": "man pilot", "\u{1F468}\u{1F3FB}\u200D\u2708\uFE0F": "man pilot: light skin tone", "\u{1F468}\u{1F3FB}\u200D\u2708": "man pilot: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u2708\uFE0F": "man pilot: medium-light skin tone", "\u{1F468}\u{1F3FC}\u200D\u2708": "man pilot: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u2708\uFE0F": "man pilot: medium skin tone", "\u{1F468}\u{1F3FD}\u200D\u2708": "man pilot: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u2708\uFE0F": "man pilot: medium-dark skin tone", "\u{1F468}\u{1F3FE}\u200D\u2708": "man pilot: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u2708\uFE0F": "man pilot: dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u2708": "man pilot: dark skin tone", "\u{1F469}\u200D\u2708\uFE0F": "woman pilot", "\u{1F469}\u200D\u2708": "woman pilot", "\u{1F469}\u{1F3FB}\u200D\u2708\uFE0F": "woman pilot: light skin tone", "\u{1F469}\u{1F3FB}\u200D\u2708": "woman pilot: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u2708\uFE0F": "woman pilot: medium-light skin tone", "\u{1F469}\u{1F3FC}\u200D\u2708": "woman pilot: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u2708\uFE0F": "woman pilot: medium skin tone", "\u{1F469}\u{1F3FD}\u200D\u2708": "woman pilot: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u2708\uFE0F": "woman pilot: medium-dark skin tone", "\u{1F469}\u{1F3FE}\u200D\u2708": "woman pilot: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u2708\uFE0F": "woman pilot: dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u2708": "woman pilot: dark skin tone", "\u{1F468}\u200D\u{1F680}": "man astronaut", "\u{1F468}\u{1F3FB}\u200D\u{1F680}": "man astronaut: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F680}": "man astronaut: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F680}": "man astronaut: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F680}": "man astronaut: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F680}": "man astronaut: dark skin tone", "\u{1F469}\u200D\u{1F680}": "woman astronaut", "\u{1F469}\u{1F3FB}\u200D\u{1F680}": "woman astronaut: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F680}": "woman astronaut: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F680}": "woman astronaut: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F680}": "woman astronaut: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F680}": "woman astronaut: dark skin tone", "\u{1F468}\u200D\u{1F692}": "man firefighter", "\u{1F468}\u{1F3FB}\u200D\u{1F692}": "man firefighter: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F692}": "man firefighter: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F692}": "man firefighter: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F692}": "man firefighter: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F692}": "man firefighter: dark skin tone", "\u{1F469}\u200D\u{1F692}": "woman firefighter", "\u{1F469}\u{1F3FB}\u200D\u{1F692}": "woman firefighter: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F692}": "woman firefighter: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F692}": "woman firefighter: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F692}": "woman firefighter: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F692}": "woman firefighter: dark skin tone", "\u{1F46E}": "police officer", "\u{1F46E}\u{1F3FB}": "police officer: light skin tone", "\u{1F46E}\u{1F3FC}": "police officer: medium-light skin tone", "\u{1F46E}\u{1F3FD}": "police officer: medium skin tone", "\u{1F46E}\u{1F3FE}": "police officer: medium-dark skin tone", "\u{1F46E}\u{1F3FF}": "police officer: dark skin tone", "\u{1F46E}\u200D\u2642\uFE0F": "man police officer", "\u{1F46E}\u200D\u2642": "man police officer", "\u{1F46E}\u{1F3FB}\u200D\u2642\uFE0F": "man police officer: light skin tone", "\u{1F46E}\u{1F3FB}\u200D\u2642": "man police officer: light skin tone", "\u{1F46E}\u{1F3FC}\u200D\u2642\uFE0F": "man police officer: medium-light skin tone", "\u{1F46E}\u{1F3FC}\u200D\u2642": "man police officer: medium-light skin tone", "\u{1F46E}\u{1F3FD}\u200D\u2642\uFE0F": "man police officer: medium skin tone", "\u{1F46E}\u{1F3FD}\u200D\u2642": "man police officer: medium skin tone", "\u{1F46E}\u{1F3FE}\u200D\u2642\uFE0F": "man police officer: medium-dark skin tone", "\u{1F46E}\u{1F3FE}\u200D\u2642": "man police officer: medium-dark skin tone", "\u{1F46E}\u{1F3FF}\u200D\u2642\uFE0F": "man police officer: dark skin tone", "\u{1F46E}\u{1F3FF}\u200D\u2642": "man police officer: dark skin tone", "\u{1F46E}\u200D\u2640\uFE0F": "woman police officer", "\u{1F46E}\u200D\u2640": "woman police officer", "\u{1F46E}\u{1F3FB}\u200D\u2640\uFE0F": "woman police officer: light skin tone", "\u{1F46E}\u{1F3FB}\u200D\u2640": "woman police officer: light skin tone", "\u{1F46E}\u{1F3FC}\u200D\u2640\uFE0F": "woman police officer: medium-light skin tone", "\u{1F46E}\u{1F3FC}\u200D\u2640": "woman police officer: medium-light skin tone", "\u{1F46E}\u{1F3FD}\u200D\u2640\uFE0F": "woman police officer: medium skin tone", "\u{1F46E}\u{1F3FD}\u200D\u2640": "woman police officer: medium skin tone", "\u{1F46E}\u{1F3FE}\u200D\u2640\uFE0F": "woman police officer: medium-dark skin tone", "\u{1F46E}\u{1F3FE}\u200D\u2640": "woman police officer: medium-dark skin tone", "\u{1F46E}\u{1F3FF}\u200D\u2640\uFE0F": "woman police officer: dark skin tone", "\u{1F46E}\u{1F3FF}\u200D\u2640": "woman police officer: dark skin tone", "\u{1F575}\uFE0F": "detective", "\u{1F575}": "detective", "\u{1F575}\u{1F3FB}": "detective: light skin tone", "\u{1F575}\u{1F3FC}": "detective: medium-light skin tone", "\u{1F575}\u{1F3FD}": "detective: medium skin tone", "\u{1F575}\u{1F3FE}": "detective: medium-dark skin tone", "\u{1F575}\u{1F3FF}": "detective: dark skin tone", "\u{1F575}\uFE0F\u200D\u2642\uFE0F": "man detective", "\u{1F575}\u200D\u2642\uFE0F": "man detective", "\u{1F575}\uFE0F\u200D\u2642": "man detective", "\u{1F575}\u200D\u2642": "man detective", "\u{1F575}\u{1F3FB}\u200D\u2642\uFE0F": "man detective: light skin tone", "\u{1F575}\u{1F3FB}\u200D\u2642": "man detective: light skin tone", "\u{1F575}\u{1F3FC}\u200D\u2642\uFE0F": "man detective: medium-light skin tone", "\u{1F575}\u{1F3FC}\u200D\u2642": "man detective: medium-light skin tone", "\u{1F575}\u{1F3FD}\u200D\u2642\uFE0F": "man detective: medium skin tone", "\u{1F575}\u{1F3FD}\u200D\u2642": "man detective: medium skin tone", "\u{1F575}\u{1F3FE}\u200D\u2642\uFE0F": "man detective: medium-dark skin tone", "\u{1F575}\u{1F3FE}\u200D\u2642": "man detective: medium-dark skin tone", "\u{1F575}\u{1F3FF}\u200D\u2642\uFE0F": "man detective: dark skin tone", "\u{1F575}\u{1F3FF}\u200D\u2642": "man detective: dark skin tone", "\u{1F575}\uFE0F\u200D\u2640\uFE0F": "woman detective", "\u{1F575}\u200D\u2640\uFE0F": "woman detective", "\u{1F575}\uFE0F\u200D\u2640": "woman detective", "\u{1F575}\u200D\u2640": "woman detective", "\u{1F575}\u{1F3FB}\u200D\u2640\uFE0F": "woman detective: light skin tone", "\u{1F575}\u{1F3FB}\u200D\u2640": "woman detective: light skin tone", "\u{1F575}\u{1F3FC}\u200D\u2640\uFE0F": "woman detective: medium-light skin tone", "\u{1F575}\u{1F3FC}\u200D\u2640": "woman detective: medium-light skin tone", "\u{1F575}\u{1F3FD}\u200D\u2640\uFE0F": "woman detective: medium skin tone", "\u{1F575}\u{1F3FD}\u200D\u2640": "woman detective: medium skin tone", "\u{1F575}\u{1F3FE}\u200D\u2640\uFE0F": "woman detective: medium-dark skin tone", "\u{1F575}\u{1F3FE}\u200D\u2640": "woman detective: medium-dark skin tone", "\u{1F575}\u{1F3FF}\u200D\u2640\uFE0F": "woman detective: dark skin tone", "\u{1F575}\u{1F3FF}\u200D\u2640": "woman detective: dark skin tone", "\u{1F482}": "guard", "\u{1F482}\u{1F3FB}": "guard: light skin tone", "\u{1F482}\u{1F3FC}": "guard: medium-light skin tone", "\u{1F482}\u{1F3FD}": "guard: medium skin tone", "\u{1F482}\u{1F3FE}": "guard: medium-dark skin tone", "\u{1F482}\u{1F3FF}": "guard: dark skin tone", "\u{1F482}\u200D\u2642\uFE0F": "man guard", "\u{1F482}\u200D\u2642": "man guard", "\u{1F482}\u{1F3FB}\u200D\u2642\uFE0F": "man guard: light skin tone", "\u{1F482}\u{1F3FB}\u200D\u2642": "man guard: light skin tone", "\u{1F482}\u{1F3FC}\u200D\u2642\uFE0F": "man guard: medium-light skin tone", "\u{1F482}\u{1F3FC}\u200D\u2642": "man guard: medium-light skin tone", "\u{1F482}\u{1F3FD}\u200D\u2642\uFE0F": "man guard: medium skin tone", "\u{1F482}\u{1F3FD}\u200D\u2642": "man guard: medium skin tone", "\u{1F482}\u{1F3FE}\u200D\u2642\uFE0F": "man guard: medium-dark skin tone", "\u{1F482}\u{1F3FE}\u200D\u2642": "man guard: medium-dark skin tone", "\u{1F482}\u{1F3FF}\u200D\u2642\uFE0F": "man guard: dark skin tone", "\u{1F482}\u{1F3FF}\u200D\u2642": "man guard: dark skin tone", "\u{1F482}\u200D\u2640\uFE0F": "woman guard", "\u{1F482}\u200D\u2640": "woman guard", "\u{1F482}\u{1F3FB}\u200D\u2640\uFE0F": "woman guard: light skin tone", "\u{1F482}\u{1F3FB}\u200D\u2640": "woman guard: light skin tone", "\u{1F482}\u{1F3FC}\u200D\u2640\uFE0F": "woman guard: medium-light skin tone", "\u{1F482}\u{1F3FC}\u200D\u2640": "woman guard: medium-light skin tone", "\u{1F482}\u{1F3FD}\u200D\u2640\uFE0F": "woman guard: medium skin tone", "\u{1F482}\u{1F3FD}\u200D\u2640": "woman guard: medium skin tone", "\u{1F482}\u{1F3FE}\u200D\u2640\uFE0F": "woman guard: medium-dark skin tone", "\u{1F482}\u{1F3FE}\u200D\u2640": "woman guard: medium-dark skin tone", "\u{1F482}\u{1F3FF}\u200D\u2640\uFE0F": "woman guard: dark skin tone", "\u{1F482}\u{1F3FF}\u200D\u2640": "woman guard: dark skin tone", "\u{1F477}": "construction worker", "\u{1F477}\u{1F3FB}": "construction worker: light skin tone", "\u{1F477}\u{1F3FC}": "construction worker: medium-light skin tone", "\u{1F477}\u{1F3FD}": "construction worker: medium skin tone", "\u{1F477}\u{1F3FE}": "construction worker: medium-dark skin tone", "\u{1F477}\u{1F3FF}": "construction worker: dark skin tone", "\u{1F477}\u200D\u2642\uFE0F": "man construction worker", "\u{1F477}\u200D\u2642": "man construction worker", "\u{1F477}\u{1F3FB}\u200D\u2642\uFE0F": "man construction worker: light skin tone", "\u{1F477}\u{1F3FB}\u200D\u2642": "man construction worker: light skin tone", "\u{1F477}\u{1F3FC}\u200D\u2642\uFE0F": "man construction worker: medium-light skin tone", "\u{1F477}\u{1F3FC}\u200D\u2642": "man construction worker: medium-light skin tone", "\u{1F477}\u{1F3FD}\u200D\u2642\uFE0F": "man construction worker: medium skin tone", "\u{1F477}\u{1F3FD}\u200D\u2642": "man construction worker: medium skin tone", "\u{1F477}\u{1F3FE}\u200D\u2642\uFE0F": "man construction worker: medium-dark skin tone", "\u{1F477}\u{1F3FE}\u200D\u2642": "man construction worker: medium-dark skin tone", "\u{1F477}\u{1F3FF}\u200D\u2642\uFE0F": "man construction worker: dark skin tone", "\u{1F477}\u{1F3FF}\u200D\u2642": "man construction worker: dark skin tone", "\u{1F477}\u200D\u2640\uFE0F": "woman construction worker", "\u{1F477}\u200D\u2640": "woman construction worker", "\u{1F477}\u{1F3FB}\u200D\u2640\uFE0F": "woman construction worker: light skin tone", "\u{1F477}\u{1F3FB}\u200D\u2640": "woman construction worker: light skin tone", "\u{1F477}\u{1F3FC}\u200D\u2640\uFE0F": "woman construction worker: medium-light skin tone", "\u{1F477}\u{1F3FC}\u200D\u2640": "woman construction worker: medium-light skin tone", "\u{1F477}\u{1F3FD}\u200D\u2640\uFE0F": "woman construction worker: medium skin tone", "\u{1F477}\u{1F3FD}\u200D\u2640": "woman construction worker: medium skin tone", "\u{1F477}\u{1F3FE}\u200D\u2640\uFE0F": "woman construction worker: medium-dark skin tone", "\u{1F477}\u{1F3FE}\u200D\u2640": "woman construction worker: medium-dark skin tone", "\u{1F477}\u{1F3FF}\u200D\u2640\uFE0F": "woman construction worker: dark skin tone", "\u{1F477}\u{1F3FF}\u200D\u2640": "woman construction worker: dark skin tone", "\u{1F934}": "prince", "\u{1F934}\u{1F3FB}": "prince: light skin tone", "\u{1F934}\u{1F3FC}": "prince: medium-light skin tone", "\u{1F934}\u{1F3FD}": "prince: medium skin tone", "\u{1F934}\u{1F3FE}": "prince: medium-dark skin tone", "\u{1F934}\u{1F3FF}": "prince: dark skin tone", "\u{1F478}": "princess", "\u{1F478}\u{1F3FB}": "princess: light skin tone", "\u{1F478}\u{1F3FC}": "princess: medium-light skin tone", "\u{1F478}\u{1F3FD}": "princess: medium skin tone", "\u{1F478}\u{1F3FE}": "princess: medium-dark skin tone", "\u{1F478}\u{1F3FF}": "princess: dark skin tone", "\u{1F473}": "person wearing turban", "\u{1F473}\u{1F3FB}": "person wearing turban: light skin tone", "\u{1F473}\u{1F3FC}": "person wearing turban: medium-light skin tone", "\u{1F473}\u{1F3FD}": "person wearing turban: medium skin tone", "\u{1F473}\u{1F3FE}": "person wearing turban: medium-dark skin tone", "\u{1F473}\u{1F3FF}": "person wearing turban: dark skin tone", "\u{1F473}\u200D\u2642\uFE0F": "man wearing turban", "\u{1F473}\u200D\u2642": "man wearing turban", "\u{1F473}\u{1F3FB}\u200D\u2642\uFE0F": "man wearing turban: light skin tone", "\u{1F473}\u{1F3FB}\u200D\u2642": "man wearing turban: light skin tone", "\u{1F473}\u{1F3FC}\u200D\u2642\uFE0F": "man wearing turban: medium-light skin tone", "\u{1F473}\u{1F3FC}\u200D\u2642": "man wearing turban: medium-light skin tone", "\u{1F473}\u{1F3FD}\u200D\u2642\uFE0F": "man wearing turban: medium skin tone", "\u{1F473}\u{1F3FD}\u200D\u2642": "man wearing turban: medium skin tone", "\u{1F473}\u{1F3FE}\u200D\u2642\uFE0F": "man wearing turban: medium-dark skin tone", "\u{1F473}\u{1F3FE}\u200D\u2642": "man wearing turban: medium-dark skin tone", "\u{1F473}\u{1F3FF}\u200D\u2642\uFE0F": "man wearing turban: dark skin tone", "\u{1F473}\u{1F3FF}\u200D\u2642": "man wearing turban: dark skin tone", "\u{1F473}\u200D\u2640\uFE0F": "woman wearing turban", "\u{1F473}\u200D\u2640": "woman wearing turban", "\u{1F473}\u{1F3FB}\u200D\u2640\uFE0F": "woman wearing turban: light skin tone", "\u{1F473}\u{1F3FB}\u200D\u2640": "woman wearing turban: light skin tone", "\u{1F473}\u{1F3FC}\u200D\u2640\uFE0F": "woman wearing turban: medium-light skin tone", "\u{1F473}\u{1F3FC}\u200D\u2640": "woman wearing turban: medium-light skin tone", "\u{1F473}\u{1F3FD}\u200D\u2640\uFE0F": "woman wearing turban: medium skin tone", "\u{1F473}\u{1F3FD}\u200D\u2640": "woman wearing turban: medium skin tone", "\u{1F473}\u{1F3FE}\u200D\u2640\uFE0F": "woman wearing turban: medium-dark skin tone", "\u{1F473}\u{1F3FE}\u200D\u2640": "woman wearing turban: medium-dark skin tone", "\u{1F473}\u{1F3FF}\u200D\u2640\uFE0F": "woman wearing turban: dark skin tone", "\u{1F473}\u{1F3FF}\u200D\u2640": "woman wearing turban: dark skin tone", "\u{1F472}": "man with Chinese cap", "\u{1F472}\u{1F3FB}": "man with Chinese cap: light skin tone", "\u{1F472}\u{1F3FC}": "man with Chinese cap: medium-light skin tone", "\u{1F472}\u{1F3FD}": "man with Chinese cap: medium skin tone", "\u{1F472}\u{1F3FE}": "man with Chinese cap: medium-dark skin tone", "\u{1F472}\u{1F3FF}": "man with Chinese cap: dark skin tone", "\u{1F9D5}": "woman with headscarf", "\u{1F9D5}\u{1F3FB}": "woman with headscarf: light skin tone", "\u{1F9D5}\u{1F3FC}": "woman with headscarf: medium-light skin tone", "\u{1F9D5}\u{1F3FD}": "woman with headscarf: medium skin tone", "\u{1F9D5}\u{1F3FE}": "woman with headscarf: medium-dark skin tone", "\u{1F9D5}\u{1F3FF}": "woman with headscarf: dark skin tone", "\u{1F9D4}": "bearded person", "\u{1F9D4}\u{1F3FB}": "bearded person: light skin tone", "\u{1F9D4}\u{1F3FC}": "bearded person: medium-light skin tone", "\u{1F9D4}\u{1F3FD}": "bearded person: medium skin tone", "\u{1F9D4}\u{1F3FE}": "bearded person: medium-dark skin tone", "\u{1F9D4}\u{1F3FF}": "bearded person: dark skin tone", "\u{1F471}": "blond-haired person", "\u{1F471}\u{1F3FB}": "blond-haired person: light skin tone", "\u{1F471}\u{1F3FC}": "blond-haired person: medium-light skin tone", "\u{1F471}\u{1F3FD}": "blond-haired person: medium skin tone", "\u{1F471}\u{1F3FE}": "blond-haired person: medium-dark skin tone", "\u{1F471}\u{1F3FF}": "blond-haired person: dark skin tone", "\u{1F471}\u200D\u2642\uFE0F": "blond-haired man", "\u{1F471}\u200D\u2642": "blond-haired man", "\u{1F471}\u{1F3FB}\u200D\u2642\uFE0F": "blond-haired man: light skin tone", "\u{1F471}\u{1F3FB}\u200D\u2642": "blond-haired man: light skin tone", "\u{1F471}\u{1F3FC}\u200D\u2642\uFE0F": "blond-haired man: medium-light skin tone", "\u{1F471}\u{1F3FC}\u200D\u2642": "blond-haired man: medium-light skin tone", "\u{1F471}\u{1F3FD}\u200D\u2642\uFE0F": "blond-haired man: medium skin tone", "\u{1F471}\u{1F3FD}\u200D\u2642": "blond-haired man: medium skin tone", "\u{1F471}\u{1F3FE}\u200D\u2642\uFE0F": "blond-haired man: medium-dark skin tone", "\u{1F471}\u{1F3FE}\u200D\u2642": "blond-haired man: medium-dark skin tone", "\u{1F471}\u{1F3FF}\u200D\u2642\uFE0F": "blond-haired man: dark skin tone", "\u{1F471}\u{1F3FF}\u200D\u2642": "blond-haired man: dark skin tone", "\u{1F471}\u200D\u2640\uFE0F": "blond-haired woman", "\u{1F471}\u200D\u2640": "blond-haired woman", "\u{1F471}\u{1F3FB}\u200D\u2640\uFE0F": "blond-haired woman: light skin tone", "\u{1F471}\u{1F3FB}\u200D\u2640": "blond-haired woman: light skin tone", "\u{1F471}\u{1F3FC}\u200D\u2640\uFE0F": "blond-haired woman: medium-light skin tone", "\u{1F471}\u{1F3FC}\u200D\u2640": "blond-haired woman: medium-light skin tone", "\u{1F471}\u{1F3FD}\u200D\u2640\uFE0F": "blond-haired woman: medium skin tone", "\u{1F471}\u{1F3FD}\u200D\u2640": "blond-haired woman: medium skin tone", "\u{1F471}\u{1F3FE}\u200D\u2640\uFE0F": "blond-haired woman: medium-dark skin tone", "\u{1F471}\u{1F3FE}\u200D\u2640": "blond-haired woman: medium-dark skin tone", "\u{1F471}\u{1F3FF}\u200D\u2640\uFE0F": "blond-haired woman: dark skin tone", "\u{1F471}\u{1F3FF}\u200D\u2640": "blond-haired woman: dark skin tone", "\u{1F468}\u200D\u{1F9B0}": "man, red haired", "\u{1F468}\u{1F3FB}\u200D\u{1F9B0}": "man, red haired: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F9B0}": "man, red haired: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F9B0}": "man, red haired: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F9B0}": "man, red haired: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F9B0}": "man, red haired: dark skin tone", "\u{1F469}\u200D\u{1F9B0}": "woman, red haired", "\u{1F469}\u{1F3FB}\u200D\u{1F9B0}": "woman, red haired: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F9B0}": "woman, red haired: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F9B0}": "woman, red haired: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F9B0}": "woman, red haired: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F9B0}": "woman, red haired: dark skin tone", "\u{1F468}\u200D\u{1F9B1}": "man, curly haired", "\u{1F468}\u{1F3FB}\u200D\u{1F9B1}": "man, curly haired: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F9B1}": "man, curly haired: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F9B1}": "man, curly haired: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F9B1}": "man, curly haired: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F9B1}": "man, curly haired: dark skin tone", "\u{1F469}\u200D\u{1F9B1}": "woman, curly haired", "\u{1F469}\u{1F3FB}\u200D\u{1F9B1}": "woman, curly haired: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F9B1}": "woman, curly haired: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F9B1}": "woman, curly haired: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F9B1}": "woman, curly haired: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F9B1}": "woman, curly haired: dark skin tone", "\u{1F468}\u200D\u{1F9B2}": "man, bald", "\u{1F468}\u{1F3FB}\u200D\u{1F9B2}": "man, bald: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F9B2}": "man, bald: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F9B2}": "man, bald: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F9B2}": "man, bald: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F9B2}": "man, bald: dark skin tone", "\u{1F469}\u200D\u{1F9B2}": "woman, bald", "\u{1F469}\u{1F3FB}\u200D\u{1F9B2}": "woman, bald: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F9B2}": "woman, bald: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F9B2}": "woman, bald: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F9B2}": "woman, bald: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F9B2}": "woman, bald: dark skin tone", "\u{1F468}\u200D\u{1F9B3}": "man, white haired", "\u{1F468}\u{1F3FB}\u200D\u{1F9B3}": "man, white haired: light skin tone", "\u{1F468}\u{1F3FC}\u200D\u{1F9B3}": "man, white haired: medium-light skin tone", "\u{1F468}\u{1F3FD}\u200D\u{1F9B3}": "man, white haired: medium skin tone", "\u{1F468}\u{1F3FE}\u200D\u{1F9B3}": "man, white haired: medium-dark skin tone", "\u{1F468}\u{1F3FF}\u200D\u{1F9B3}": "man, white haired: dark skin tone", "\u{1F469}\u200D\u{1F9B3}": "woman, white haired", "\u{1F469}\u{1F3FB}\u200D\u{1F9B3}": "woman, white haired: light skin tone", "\u{1F469}\u{1F3FC}\u200D\u{1F9B3}": "woman, white haired: medium-light skin tone", "\u{1F469}\u{1F3FD}\u200D\u{1F9B3}": "woman, white haired: medium skin tone", "\u{1F469}\u{1F3FE}\u200D\u{1F9B3}": "woman, white haired: medium-dark skin tone", "\u{1F469}\u{1F3FF}\u200D\u{1F9B3}": "woman, white haired: dark skin tone", "\u{1F935}": "man in tuxedo", "\u{1F935}\u{1F3FB}": "man in tuxedo: light skin tone", "\u{1F935}\u{1F3FC}": "man in tuxedo: medium-light skin tone", "\u{1F935}\u{1F3FD}": "man in tuxedo: medium skin tone", "\u{1F935}\u{1F3FE}": "man in tuxedo: medium-dark skin tone", "\u{1F935}\u{1F3FF}": "man in tuxedo: dark skin tone", "\u{1F470}": "bride with veil", "\u{1F470}\u{1F3FB}": "bride with veil: light skin tone", "\u{1F470}\u{1F3FC}": "bride with veil: medium-light skin tone", "\u{1F470}\u{1F3FD}": "bride with veil: medium skin tone", "\u{1F470}\u{1F3FE}": "bride with veil: medium-dark skin tone", "\u{1F470}\u{1F3FF}": "bride with veil: dark skin tone", "\u{1F930}": "pregnant woman", "\u{1F930}\u{1F3FB}": "pregnant woman: light skin tone", "\u{1F930}\u{1F3FC}": "pregnant woman: medium-light skin tone", "\u{1F930}\u{1F3FD}": "pregnant woman: medium skin tone", "\u{1F930}\u{1F3FE}": "pregnant woman: medium-dark skin tone", "\u{1F930}\u{1F3FF}": "pregnant woman: dark skin tone", "\u{1F931}": "breast-feeding", "\u{1F931}\u{1F3FB}": "breast-feeding: light skin tone", "\u{1F931}\u{1F3FC}": "breast-feeding: medium-light skin tone", "\u{1F931}\u{1F3FD}": "breast-feeding: medium skin tone", "\u{1F931}\u{1F3FE}": "breast-feeding: medium-dark skin tone", "\u{1F931}\u{1F3FF}": "breast-feeding: dark skin tone", "\u{1F47C}": "baby angel", "\u{1F47C}\u{1F3FB}": "baby angel: light skin tone", "\u{1F47C}\u{1F3FC}": "baby angel: medium-light skin tone", "\u{1F47C}\u{1F3FD}": "baby angel: medium skin tone", "\u{1F47C}\u{1F3FE}": "baby angel: medium-dark skin tone", "\u{1F47C}\u{1F3FF}": "baby angel: dark skin tone", "\u{1F385}": "Santa Claus", "\u{1F385}\u{1F3FB}": "Santa Claus: light skin tone", "\u{1F385}\u{1F3FC}": "Santa Claus: medium-light skin tone", "\u{1F385}\u{1F3FD}": "Santa Claus: medium skin tone", "\u{1F385}\u{1F3FE}": "Santa Claus: medium-dark skin tone", "\u{1F385}\u{1F3FF}": "Santa Claus: dark skin tone", "\u{1F936}": "Mrs. Claus", "\u{1F936}\u{1F3FB}": "Mrs. Claus: light skin tone", "\u{1F936}\u{1F3FC}": "Mrs. Claus: medium-light skin tone", "\u{1F936}\u{1F3FD}": "Mrs. Claus: medium skin tone", "\u{1F936}\u{1F3FE}": "Mrs. Claus: medium-dark skin tone", "\u{1F936}\u{1F3FF}": "Mrs. Claus: dark skin tone", "\u{1F9B8}": "superhero", "\u{1F9B8}\u{1F3FB}": "superhero: light skin tone", "\u{1F9B8}\u{1F3FC}": "superhero: medium-light skin tone", "\u{1F9B8}\u{1F3FD}": "superhero: medium skin tone", "\u{1F9B8}\u{1F3FE}": "superhero: medium-dark skin tone", "\u{1F9B8}\u{1F3FF}": "superhero: dark skin tone", "\u{1F9B8}\u200D\u2640\uFE0F": "woman superhero", "\u{1F9B8}\u200D\u2640": "woman superhero", "\u{1F9B8}\u{1F3FB}\u200D\u2640\uFE0F": "woman superhero: light skin tone", "\u{1F9B8}\u{1F3FB}\u200D\u2640": "woman superhero: light skin tone", "\u{1F9B8}\u{1F3FC}\u200D\u2640\uFE0F": "woman superhero: medium-light skin tone", "\u{1F9B8}\u{1F3FC}\u200D\u2640": "woman superhero: medium-light skin tone", "\u{1F9B8}\u{1F3FD}\u200D\u2640\uFE0F": "woman superhero: medium skin tone", "\u{1F9B8}\u{1F3FD}\u200D\u2640": "woman superhero: medium skin tone", "\u{1F9B8}\u{1F3FE}\u200D\u2640\uFE0F": "woman superhero: medium-dark skin tone", "\u{1F9B8}\u{1F3FE}\u200D\u2640": "woman superhero: medium-dark skin tone", "\u{1F9B8}\u{1F3FF}\u200D\u2640\uFE0F": "woman superhero: dark skin tone", "\u{1F9B8}\u{1F3FF}\u200D\u2640": "woman superhero: dark skin tone", "\u{1F9B8}\u200D\u2642\uFE0F": "man superhero", "\u{1F9B8}\u200D\u2642": "man superhero", "\u{1F9B8}\u{1F3FB}\u200D\u2642\uFE0F": "man superhero: light skin tone", "\u{1F9B8}\u{1F3FB}\u200D\u2642": "man superhero: light skin tone", "\u{1F9B8}\u{1F3FC}\u200D\u2642\uFE0F": "man superhero: medium-light skin tone", "\u{1F9B8}\u{1F3FC}\u200D\u2642": "man superhero: medium-light skin tone", "\u{1F9B8}\u{1F3FD}\u200D\u2642\uFE0F": "man superhero: medium skin tone", "\u{1F9B8}\u{1F3FD}\u200D\u2642": "man superhero: medium skin tone", "\u{1F9B8}\u{1F3FE}\u200D\u2642\uFE0F": "man superhero: medium-dark skin tone", "\u{1F9B8}\u{1F3FE}\u200D\u2642": "man superhero: medium-dark skin tone", "\u{1F9B8}\u{1F3FF}\u200D\u2642\uFE0F": "man superhero: dark skin tone", "\u{1F9B8}\u{1F3FF}\u200D\u2642": "man superhero: dark skin tone", "\u{1F9B9}": "supervillain", "\u{1F9B9}\u{1F3FB}": "supervillain: light skin tone", "\u{1F9B9}\u{1F3FC}": "supervillain: medium-light skin tone", "\u{1F9B9}\u{1F3FD}": "supervillain: medium skin tone", "\u{1F9B9}\u{1F3FE}": "supervillain: medium-dark skin tone", "\u{1F9B9}\u{1F3FF}": "supervillain: dark skin tone", "\u{1F9B9}\u200D\u2640\uFE0F": "woman supervillain", "\u{1F9B9}\u200D\u2640": "woman supervillain", "\u{1F9B9}\u{1F3FB}\u200D\u2640\uFE0F": "woman supervillain: light skin tone", "\u{1F9B9}\u{1F3FB}\u200D\u2640": "woman supervillain: light skin tone", "\u{1F9B9}\u{1F3FC}\u200D\u2640\uFE0F": "woman supervillain: medium-light skin tone", "\u{1F9B9}\u{1F3FC}\u200D\u2640": "woman supervillain: medium-light skin tone", "\u{1F9B9}\u{1F3FD}\u200D\u2640\uFE0F": "woman supervillain: medium skin tone", "\u{1F9B9}\u{1F3FD}\u200D\u2640": "woman supervillain: medium skin tone", "\u{1F9B9}\u{1F3FE}\u200D\u2640\uFE0F": "woman supervillain: medium-dark skin tone", "\u{1F9B9}\u{1F3FE}\u200D\u2640": "woman supervillain: medium-dark skin tone", "\u{1F9B9}\u{1F3FF}\u200D\u2640\uFE0F": "woman supervillain: dark skin tone", "\u{1F9B9}\u{1F3FF}\u200D\u2640": "woman supervillain: dark skin tone", "\u{1F9B9}\u200D\u2642\uFE0F": "man supervillain", "\u{1F9B9}\u200D\u2642": "man supervillain", "\u{1F9B9}\u{1F3FB}\u200D\u2642\uFE0F": "man supervillain: light skin tone", "\u{1F9B9}\u{1F3FB}\u200D\u2642": "man supervillain: light skin tone", "\u{1F9B9}\u{1F3FC}\u200D\u2642\uFE0F": "man supervillain: medium-light skin tone", "\u{1F9B9}\u{1F3FC}\u200D\u2642": "man supervillain: medium-light skin tone", "\u{1F9B9}\u{1F3FD}\u200D\u2642\uFE0F": "man supervillain: medium skin tone", "\u{1F9B9}\u{1F3FD}\u200D\u2642": "man supervillain: medium skin tone", "\u{1F9B9}\u{1F3FE}\u200D\u2642\uFE0F": "man supervillain: medium-dark skin tone", "\u{1F9B9}\u{1F3FE}\u200D\u2642": "man supervillain: medium-dark skin tone", "\u{1F9B9}\u{1F3FF}\u200D\u2642\uFE0F": "man supervillain: dark skin tone", "\u{1F9B9}\u{1F3FF}\u200D\u2642": "man supervillain: dark skin tone", "\u{1F9D9}": "mage", "\u{1F9D9}\u{1F3FB}": "mage: light skin tone", "\u{1F9D9}\u{1F3FC}": "mage: medium-light skin tone", "\u{1F9D9}\u{1F3FD}": "mage: medium skin tone", "\u{1F9D9}\u{1F3FE}": "mage: medium-dark skin tone", "\u{1F9D9}\u{1F3FF}": "mage: dark skin tone", "\u{1F9D9}\u200D\u2640\uFE0F": "woman mage", "\u{1F9D9}\u200D\u2640": "woman mage", "\u{1F9D9}\u{1F3FB}\u200D\u2640\uFE0F": "woman mage: light skin tone", "\u{1F9D9}\u{1F3FB}\u200D\u2640": "woman mage: light skin tone", "\u{1F9D9}\u{1F3FC}\u200D\u2640\uFE0F": "woman mage: medium-light skin tone", "\u{1F9D9}\u{1F3FC}\u200D\u2640": "woman mage: medium-light skin tone", "\u{1F9D9}\u{1F3FD}\u200D\u2640\uFE0F": "woman mage: medium skin tone", "\u{1F9D9}\u{1F3FD}\u200D\u2640": "woman mage: medium skin tone", "\u{1F9D9}\u{1F3FE}\u200D\u2640\uFE0F": "woman mage: medium-dark skin tone", "\u{1F9D9}\u{1F3FE}\u200D\u2640": "woman mage: medium-dark skin tone", "\u{1F9D9}\u{1F3FF}\u200D\u2640\uFE0F": "woman mage: dark skin tone", "\u{1F9D9}\u{1F3FF}\u200D\u2640": "woman mage: dark skin tone", "\u{1F9D9}\u200D\u2642\uFE0F": "man mage", "\u{1F9D9}\u200D\u2642": "man mage", "\u{1F9D9}\u{1F3FB}\u200D\u2642\uFE0F": "man mage: light skin tone", "\u{1F9D9}\u{1F3FB}\u200D\u2642": "man mage: light skin tone", "\u{1F9D9}\u{1F3FC}\u200D\u2642\uFE0F": "man mage: medium-light skin tone", "\u{1F9D9}\u{1F3FC}\u200D\u2642": "man mage: medium-light skin tone", "\u{1F9D9}\u{1F3FD}\u200D\u2642\uFE0F": "man mage: medium skin tone", "\u{1F9D9}\u{1F3FD}\u200D\u2642": "man mage: medium skin tone", "\u{1F9D9}\u{1F3FE}\u200D\u2642\uFE0F": "man mage: medium-dark skin tone", "\u{1F9D9}\u{1F3FE}\u200D\u2642": "man mage: medium-dark skin tone", "\u{1F9D9}\u{1F3FF}\u200D\u2642\uFE0F": "man mage: dark skin tone", "\u{1F9D9}\u{1F3FF}\u200D\u2642": "man mage: dark skin tone", "\u{1F9DA}": "fairy", "\u{1F9DA}\u{1F3FB}": "fairy: light skin tone", "\u{1F9DA}\u{1F3FC}": "fairy: medium-light skin tone", "\u{1F9DA}\u{1F3FD}": "fairy: medium skin tone", "\u{1F9DA}\u{1F3FE}": "fairy: medium-dark skin tone", "\u{1F9DA}\u{1F3FF}": "fairy: dark skin tone", "\u{1F9DA}\u200D\u2640\uFE0F": "woman fairy", "\u{1F9DA}\u200D\u2640": "woman fairy", "\u{1F9DA}\u{1F3FB}\u200D\u2640\uFE0F": "woman fairy: light skin tone", "\u{1F9DA}\u{1F3FB}\u200D\u2640": "woman fairy: light skin tone", "\u{1F9DA}\u{1F3FC}\u200D\u2640\uFE0F": "woman fairy: medium-light skin tone", "\u{1F9DA}\u{1F3FC}\u200D\u2640": "woman fairy: medium-light skin tone", "\u{1F9DA}\u{1F3FD}\u200D\u2640\uFE0F": "woman fairy: medium skin tone", "\u{1F9DA}\u{1F3FD}\u200D\u2640": "woman fairy: medium skin tone", "\u{1F9DA}\u{1F3FE}\u200D\u2640\uFE0F": "woman fairy: medium-dark skin tone", "\u{1F9DA}\u{1F3FE}\u200D\u2640": "woman fairy: medium-dark skin tone", "\u{1F9DA}\u{1F3FF}\u200D\u2640\uFE0F": "woman fairy: dark skin tone", "\u{1F9DA}\u{1F3FF}\u200D\u2640": "woman fairy: dark skin tone", "\u{1F9DA}\u200D\u2642\uFE0F": "man fairy", "\u{1F9DA}\u200D\u2642": "man fairy", "\u{1F9DA}\u{1F3FB}\u200D\u2642\uFE0F": "man fairy: light skin tone", "\u{1F9DA}\u{1F3FB}\u200D\u2642": "man fairy: light skin tone", "\u{1F9DA}\u{1F3FC}\u200D\u2642\uFE0F": "man fairy: medium-light skin tone", "\u{1F9DA}\u{1F3FC}\u200D\u2642": "man fairy: medium-light skin tone", "\u{1F9DA}\u{1F3FD}\u200D\u2642\uFE0F": "man fairy: medium skin tone", "\u{1F9DA}\u{1F3FD}\u200D\u2642": "man fairy: medium skin tone", "\u{1F9DA}\u{1F3FE}\u200D\u2642\uFE0F": "man fairy: medium-dark skin tone", "\u{1F9DA}\u{1F3FE}\u200D\u2642": "man fairy: medium-dark skin tone", "\u{1F9DA}\u{1F3FF}\u200D\u2642\uFE0F": "man fairy: dark skin tone", "\u{1F9DA}\u{1F3FF}\u200D\u2642": "man fairy: dark skin tone", "\u{1F9DB}": "vampire", "\u{1F9DB}\u{1F3FB}": "vampire: light skin tone", "\u{1F9DB}\u{1F3FC}": "vampire: medium-light skin tone", "\u{1F9DB}\u{1F3FD}": "vampire: medium skin tone", "\u{1F9DB}\u{1F3FE}": "vampire: medium-dark skin tone", "\u{1F9DB}\u{1F3FF}": "vampire: dark skin tone", "\u{1F9DB}\u200D\u2640\uFE0F": "woman vampire", "\u{1F9DB}\u200D\u2640": "woman vampire", "\u{1F9DB}\u{1F3FB}\u200D\u2640\uFE0F": "woman vampire: light skin tone", "\u{1F9DB}\u{1F3FB}\u200D\u2640": "woman vampire: light skin tone", "\u{1F9DB}\u{1F3FC}\u200D\u2640\uFE0F": "woman vampire: medium-light skin tone", "\u{1F9DB}\u{1F3FC}\u200D\u2640": "woman vampire: medium-light skin tone", "\u{1F9DB}\u{1F3FD}\u200D\u2640\uFE0F": "woman vampire: medium skin tone", "\u{1F9DB}\u{1F3FD}\u200D\u2640": "woman vampire: medium skin tone", "\u{1F9DB}\u{1F3FE}\u200D\u2640\uFE0F": "woman vampire: medium-dark skin tone", "\u{1F9DB}\u{1F3FE}\u200D\u2640": "woman vampire: medium-dark skin tone", "\u{1F9DB}\u{1F3FF}\u200D\u2640\uFE0F": "woman vampire: dark skin tone", "\u{1F9DB}\u{1F3FF}\u200D\u2640": "woman vampire: dark skin tone", "\u{1F9DB}\u200D\u2642\uFE0F": "man vampire", "\u{1F9DB}\u200D\u2642": "man vampire", "\u{1F9DB}\u{1F3FB}\u200D\u2642\uFE0F": "man vampire: light skin tone", "\u{1F9DB}\u{1F3FB}\u200D\u2642": "man vampire: light skin tone", "\u{1F9DB}\u{1F3FC}\u200D\u2642\uFE0F": "man vampire: medium-light skin tone", "\u{1F9DB}\u{1F3FC}\u200D\u2642": "man vampire: medium-light skin tone", "\u{1F9DB}\u{1F3FD}\u200D\u2642\uFE0F": "man vampire: medium skin tone", "\u{1F9DB}\u{1F3FD}\u200D\u2642": "man vampire: medium skin tone", "\u{1F9DB}\u{1F3FE}\u200D\u2642\uFE0F": "man vampire: medium-dark skin tone", "\u{1F9DB}\u{1F3FE}\u200D\u2642": "man vampire: medium-dark skin tone", "\u{1F9DB}\u{1F3FF}\u200D\u2642\uFE0F": "man vampire: dark skin tone", "\u{1F9DB}\u{1F3FF}\u200D\u2642": "man vampire: dark skin tone", "\u{1F9DC}": "merperson", "\u{1F9DC}\u{1F3FB}": "merperson: light skin tone", "\u{1F9DC}\u{1F3FC}": "merperson: medium-light skin tone", "\u{1F9DC}\u{1F3FD}": "merperson: medium skin tone", "\u{1F9DC}\u{1F3FE}": "merperson: medium-dark skin tone", "\u{1F9DC}\u{1F3FF}": "merperson: dark skin tone", "\u{1F9DC}\u200D\u2640\uFE0F": "mermaid", "\u{1F9DC}\u200D\u2640": "mermaid", "\u{1F9DC}\u{1F3FB}\u200D\u2640\uFE0F": "mermaid: light skin tone", "\u{1F9DC}\u{1F3FB}\u200D\u2640": "mermaid: light skin tone", "\u{1F9DC}\u{1F3FC}\u200D\u2640\uFE0F": "mermaid: medium-light skin tone", "\u{1F9DC}\u{1F3FC}\u200D\u2640": "mermaid: medium-light skin tone", "\u{1F9DC}\u{1F3FD}\u200D\u2640\uFE0F": "mermaid: medium skin tone", "\u{1F9DC}\u{1F3FD}\u200D\u2640": "mermaid: medium skin tone", "\u{1F9DC}\u{1F3FE}\u200D\u2640\uFE0F": "mermaid: medium-dark skin tone", "\u{1F9DC}\u{1F3FE}\u200D\u2640": "mermaid: medium-dark skin tone", "\u{1F9DC}\u{1F3FF}\u200D\u2640\uFE0F": "mermaid: dark skin tone", "\u{1F9DC}\u{1F3FF}\u200D\u2640": "mermaid: dark skin tone", "\u{1F9DC}\u200D\u2642\uFE0F": "merman", "\u{1F9DC}\u200D\u2642": "merman", "\u{1F9DC}\u{1F3FB}\u200D\u2642\uFE0F": "merman: light skin tone", "\u{1F9DC}\u{1F3FB}\u200D\u2642": "merman: light skin tone", "\u{1F9DC}\u{1F3FC}\u200D\u2642\uFE0F": "merman: medium-light skin tone", "\u{1F9DC}\u{1F3FC}\u200D\u2642": "merman: medium-light skin tone", "\u{1F9DC}\u{1F3FD}\u200D\u2642\uFE0F": "merman: medium skin tone", "\u{1F9DC}\u{1F3FD}\u200D\u2642": "merman: medium skin tone", "\u{1F9DC}\u{1F3FE}\u200D\u2642\uFE0F": "merman: medium-dark skin tone", "\u{1F9DC}\u{1F3FE}\u200D\u2642": "merman: medium-dark skin tone", "\u{1F9DC}\u{1F3FF}\u200D\u2642\uFE0F": "merman: dark skin tone", "\u{1F9DC}\u{1F3FF}\u200D\u2642": "merman: dark skin tone", "\u{1F9DD}": "elf", "\u{1F9DD}\u{1F3FB}": "elf: light skin tone", "\u{1F9DD}\u{1F3FC}": "elf: medium-light skin tone", "\u{1F9DD}\u{1F3FD}": "elf: medium skin tone", "\u{1F9DD}\u{1F3FE}": "elf: medium-dark skin tone", "\u{1F9DD}\u{1F3FF}": "elf: dark skin tone", "\u{1F9DD}\u200D\u2640\uFE0F": "woman elf", "\u{1F9DD}\u200D\u2640": "woman elf", "\u{1F9DD}\u{1F3FB}\u200D\u2640\uFE0F": "woman elf: light skin tone", "\u{1F9DD}\u{1F3FB}\u200D\u2640": "woman elf: light skin tone", "\u{1F9DD}\u{1F3FC}\u200D\u2640\uFE0F": "woman elf: medium-light skin tone", "\u{1F9DD}\u{1F3FC}\u200D\u2640": "woman elf: medium-light skin tone", "\u{1F9DD}\u{1F3FD}\u200D\u2640\uFE0F": "woman elf: medium skin tone", "\u{1F9DD}\u{1F3FD}\u200D\u2640": "woman elf: medium skin tone", "\u{1F9DD}\u{1F3FE}\u200D\u2640\uFE0F": "woman elf: medium-dark skin tone", "\u{1F9DD}\u{1F3FE}\u200D\u2640": "woman elf: medium-dark skin tone", "\u{1F9DD}\u{1F3FF}\u200D\u2640\uFE0F": "woman elf: dark skin tone", "\u{1F9DD}\u{1F3FF}\u200D\u2640": "woman elf: dark skin tone", "\u{1F9DD}\u200D\u2642\uFE0F": "man elf", "\u{1F9DD}\u200D\u2642": "man elf", "\u{1F9DD}\u{1F3FB}\u200D\u2642\uFE0F": "man elf: light skin tone", "\u{1F9DD}\u{1F3FB}\u200D\u2642": "man elf: light skin tone", "\u{1F9DD}\u{1F3FC}\u200D\u2642\uFE0F": "man elf: medium-light skin tone", "\u{1F9DD}\u{1F3FC}\u200D\u2642": "man elf: medium-light skin tone", "\u{1F9DD}\u{1F3FD}\u200D\u2642\uFE0F": "man elf: medium skin tone", "\u{1F9DD}\u{1F3FD}\u200D\u2642": "man elf: medium skin tone", "\u{1F9DD}\u{1F3FE}\u200D\u2642\uFE0F": "man elf: medium-dark skin tone", "\u{1F9DD}\u{1F3FE}\u200D\u2642": "man elf: medium-dark skin tone", "\u{1F9DD}\u{1F3FF}\u200D\u2642\uFE0F": "man elf: dark skin tone", "\u{1F9DD}\u{1F3FF}\u200D\u2642": "man elf: dark skin tone", "\u{1F9DE}": "genie", "\u{1F9DE}\u200D\u2640\uFE0F": "woman genie", "\u{1F9DE}\u200D\u2640": "woman genie", "\u{1F9DE}\u200D\u2642\uFE0F": "man genie", "\u{1F9DE}\u200D\u2642": "man genie", "\u{1F9DF}": "zombie", "\u{1F9DF}\u200D\u2640\uFE0F": "woman zombie", "\u{1F9DF}\u200D\u2640": "woman zombie", "\u{1F9DF}\u200D\u2642\uFE0F": "man zombie", "\u{1F9DF}\u200D\u2642": "man zombie", "\u{1F64D}": "person frowning", "\u{1F64D}\u{1F3FB}": "person frowning: light skin tone", "\u{1F64D}\u{1F3FC}": "person frowning: medium-light skin tone", "\u{1F64D}\u{1F3FD}": "person frowning: medium skin tone", "\u{1F64D}\u{1F3FE}": "person frowning: medium-dark skin tone", "\u{1F64D}\u{1F3FF}": "person frowning: dark skin tone", "\u{1F64D}\u200D\u2642\uFE0F": "man frowning", "\u{1F64D}\u200D\u2642": "man frowning", "\u{1F64D}\u{1F3FB}\u200D\u2642\uFE0F": "man frowning: light skin tone", "\u{1F64D}\u{1F3FB}\u200D\u2642": "man frowning: light skin tone", "\u{1F64D}\u{1F3FC}\u200D\u2642\uFE0F": "man frowning: medium-light skin tone", "\u{1F64D}\u{1F3FC}\u200D\u2642": "man frowning: medium-light skin tone", "\u{1F64D}\u{1F3FD}\u200D\u2642\uFE0F": "man frowning: medium skin tone", "\u{1F64D}\u{1F3FD}\u200D\u2642": "man frowning: medium skin tone", "\u{1F64D}\u{1F3FE}\u200D\u2642\uFE0F": "man frowning: medium-dark skin tone", "\u{1F64D}\u{1F3FE}\u200D\u2642": "man frowning: medium-dark skin tone", "\u{1F64D}\u{1F3FF}\u200D\u2642\uFE0F": "man frowning: dark skin tone", "\u{1F64D}\u{1F3FF}\u200D\u2642": "man frowning: dark skin tone", "\u{1F64D}\u200D\u2640\uFE0F": "woman frowning", "\u{1F64D}\u200D\u2640": "woman frowning", "\u{1F64D}\u{1F3FB}\u200D\u2640\uFE0F": "woman frowning: light skin tone", "\u{1F64D}\u{1F3FB}\u200D\u2640": "woman frowning: light skin tone", "\u{1F64D}\u{1F3FC}\u200D\u2640\uFE0F": "woman frowning: medium-light skin tone", "\u{1F64D}\u{1F3FC}\u200D\u2640": "woman frowning: medium-light skin tone", "\u{1F64D}\u{1F3FD}\u200D\u2640\uFE0F": "woman frowning: medium skin tone", "\u{1F64D}\u{1F3FD}\u200D\u2640": "woman frowning: medium skin tone", "\u{1F64D}\u{1F3FE}\u200D\u2640\uFE0F": "woman frowning: medium-dark skin tone", "\u{1F64D}\u{1F3FE}\u200D\u2640": "woman frowning: medium-dark skin tone", "\u{1F64D}\u{1F3FF}\u200D\u2640\uFE0F": "woman frowning: dark skin tone", "\u{1F64D}\u{1F3FF}\u200D\u2640": "woman frowning: dark skin tone", "\u{1F64E}": "person pouting", "\u{1F64E}\u{1F3FB}": "person pouting: light skin tone", "\u{1F64E}\u{1F3FC}": "person pouting: medium-light skin tone", "\u{1F64E}\u{1F3FD}": "person pouting: medium skin tone", "\u{1F64E}\u{1F3FE}": "person pouting: medium-dark skin tone", "\u{1F64E}\u{1F3FF}": "person pouting: dark skin tone", "\u{1F64E}\u200D\u2642\uFE0F": "man pouting", "\u{1F64E}\u200D\u2642": "man pouting", "\u{1F64E}\u{1F3FB}\u200D\u2642\uFE0F": "man pouting: light skin tone", "\u{1F64E}\u{1F3FB}\u200D\u2642": "man pouting: light skin tone", "\u{1F64E}\u{1F3FC}\u200D\u2642\uFE0F": "man pouting: medium-light skin tone", "\u{1F64E}\u{1F3FC}\u200D\u2642": "man pouting: medium-light skin tone", "\u{1F64E}\u{1F3FD}\u200D\u2642\uFE0F": "man pouting: medium skin tone", "\u{1F64E}\u{1F3FD}\u200D\u2642": "man pouting: medium skin tone", "\u{1F64E}\u{1F3FE}\u200D\u2642\uFE0F": "man pouting: medium-dark skin tone", "\u{1F64E}\u{1F3FE}\u200D\u2642": "man pouting: medium-dark skin tone", "\u{1F64E}\u{1F3FF}\u200D\u2642\uFE0F": "man pouting: dark skin tone", "\u{1F64E}\u{1F3FF}\u200D\u2642": "man pouting: dark skin tone", "\u{1F64E}\u200D\u2640\uFE0F": "woman pouting", "\u{1F64E}\u200D\u2640": "woman pouting", "\u{1F64E}\u{1F3FB}\u200D\u2640\uFE0F": "woman pouting: light skin tone", "\u{1F64E}\u{1F3FB}\u200D\u2640": "woman pouting: light skin tone", "\u{1F64E}\u{1F3FC}\u200D\u2640\uFE0F": "woman pouting: medium-light skin tone", "\u{1F64E}\u{1F3FC}\u200D\u2640": "woman pouting: medium-light skin tone", "\u{1F64E}\u{1F3FD}\u200D\u2640\uFE0F": "woman pouting: medium skin tone", "\u{1F64E}\u{1F3FD}\u200D\u2640": "woman pouting: medium skin tone", "\u{1F64E}\u{1F3FE}\u200D\u2640\uFE0F": "woman pouting: medium-dark skin tone", "\u{1F64E}\u{1F3FE}\u200D\u2640": "woman pouting: medium-dark skin tone", "\u{1F64E}\u{1F3FF}\u200D\u2640\uFE0F": "woman pouting: dark skin tone", "\u{1F64E}\u{1F3FF}\u200D\u2640": "woman pouting: dark skin tone", "\u{1F645}": "person gesturing NO", "\u{1F645}\u{1F3FB}": "person gesturing NO: light skin tone", "\u{1F645}\u{1F3FC}": "person gesturing NO: medium-light skin tone", "\u{1F645}\u{1F3FD}": "person gesturing NO: medium skin tone", "\u{1F645}\u{1F3FE}": "person gesturing NO: medium-dark skin tone", "\u{1F645}\u{1F3FF}": "person gesturing NO: dark skin tone", "\u{1F645}\u200D\u2642\uFE0F": "man gesturing NO", "\u{1F645}\u200D\u2642": "man gesturing NO", "\u{1F645}\u{1F3FB}\u200D\u2642\uFE0F": "man gesturing NO: light skin tone", "\u{1F645}\u{1F3FB}\u200D\u2642": "man gesturing NO: light skin tone", "\u{1F645}\u{1F3FC}\u200D\u2642\uFE0F": "man gesturing NO: medium-light skin tone", "\u{1F645}\u{1F3FC}\u200D\u2642": "man gesturing NO: medium-light skin tone", "\u{1F645}\u{1F3FD}\u200D\u2642\uFE0F": "man gesturing NO: medium skin tone", "\u{1F645}\u{1F3FD}\u200D\u2642": "man gesturing NO: medium skin tone", "\u{1F645}\u{1F3FE}\u200D\u2642\uFE0F": "man gesturing NO: medium-dark skin tone", "\u{1F645}\u{1F3FE}\u200D\u2642": "man gesturing NO: medium-dark skin tone", "\u{1F645}\u{1F3FF}\u200D\u2642\uFE0F": "man gesturing NO: dark skin tone", "\u{1F645}\u{1F3FF}\u200D\u2642": "man gesturing NO: dark skin tone", "\u{1F645}\u200D\u2640\uFE0F": "woman gesturing NO", "\u{1F645}\u200D\u2640": "woman gesturing NO", "\u{1F645}\u{1F3FB}\u200D\u2640\uFE0F": "woman gesturing NO: light skin tone", "\u{1F645}\u{1F3FB}\u200D\u2640": "woman gesturing NO: light skin tone", "\u{1F645}\u{1F3FC}\u200D\u2640\uFE0F": "woman gesturing NO: medium-light skin tone", "\u{1F645}\u{1F3FC}\u200D\u2640": "woman gesturing NO: medium-light skin tone", "\u{1F645}\u{1F3FD}\u200D\u2640\uFE0F": "woman gesturing NO: medium skin tone", "\u{1F645}\u{1F3FD}\u200D\u2640": "woman gesturing NO: medium skin tone", "\u{1F645}\u{1F3FE}\u200D\u2640\uFE0F": "woman gesturing NO: medium-dark skin tone", "\u{1F645}\u{1F3FE}\u200D\u2640": "woman gesturing NO: medium-dark skin tone", "\u{1F645}\u{1F3FF}\u200D\u2640\uFE0F": "woman gesturing NO: dark skin tone", "\u{1F645}\u{1F3FF}\u200D\u2640": "woman gesturing NO: dark skin tone", "\u{1F646}": "person gesturing OK", "\u{1F646}\u{1F3FB}": "person gesturing OK: light skin tone", "\u{1F646}\u{1F3FC}": "person gesturing OK: medium-light skin tone", "\u{1F646}\u{1F3FD}": "person gesturing OK: medium skin tone", "\u{1F646}\u{1F3FE}": "person gesturing OK: medium-dark skin tone", "\u{1F646}\u{1F3FF}": "person gesturing OK: dark skin tone", "\u{1F646}\u200D\u2642\uFE0F": "man gesturing OK", "\u{1F646}\u200D\u2642": "man gesturing OK", "\u{1F646}\u{1F3FB}\u200D\u2642\uFE0F": "man gesturing OK: light skin tone", "\u{1F646}\u{1F3FB}\u200D\u2642": "man gesturing OK: light skin tone", "\u{1F646}\u{1F3FC}\u200D\u2642\uFE0F": "man gesturing OK: medium-light skin tone", "\u{1F646}\u{1F3FC}\u200D\u2642": "man gesturing OK: medium-light skin tone", "\u{1F646}\u{1F3FD}\u200D\u2642\uFE0F": "man gesturing OK: medium skin tone", "\u{1F646}\u{1F3FD}\u200D\u2642": "man gesturing OK: medium skin tone", "\u{1F646}\u{1F3FE}\u200D\u2642\uFE0F": "man gesturing OK: medium-dark skin tone", "\u{1F646}\u{1F3FE}\u200D\u2642": "man gesturing OK: medium-dark skin tone", "\u{1F646}\u{1F3FF}\u200D\u2642\uFE0F": "man gesturing OK: dark skin tone", "\u{1F646}\u{1F3FF}\u200D\u2642": "man gesturing OK: dark skin tone", "\u{1F646}\u200D\u2640\uFE0F": "woman gesturing OK", "\u{1F646}\u200D\u2640": "woman gesturing OK", "\u{1F646}\u{1F3FB}\u200D\u2640\uFE0F": "woman gesturing OK: light skin tone", "\u{1F646}\u{1F3FB}\u200D\u2640": "woman gesturing OK: light skin tone", "\u{1F646}\u{1F3FC}\u200D\u2640\uFE0F": "woman gesturing OK: medium-light skin tone", "\u{1F646}\u{1F3FC}\u200D\u2640": "woman gesturing OK: medium-light skin tone", "\u{1F646}\u{1F3FD}\u200D\u2640\uFE0F": "woman gesturing OK: medium skin tone", "\u{1F646}\u{1F3FD}\u200D\u2640": "woman gesturing OK: medium skin tone", "\u{1F646}\u{1F3FE}\u200D\u2640\uFE0F": "woman gesturing OK: medium-dark skin tone", "\u{1F646}\u{1F3FE}\u200D\u2640": "woman gesturing OK: medium-dark skin tone", "\u{1F646}\u{1F3FF}\u200D\u2640\uFE0F": "woman gesturing OK: dark skin tone", "\u{1F646}\u{1F3FF}\u200D\u2640": "woman gesturing OK: dark skin tone", "\u{1F481}": "person tipping hand", "\u{1F481}\u{1F3FB}": "person tipping hand: light skin tone", "\u{1F481}\u{1F3FC}": "person tipping hand: medium-light skin tone", "\u{1F481}\u{1F3FD}": "person tipping hand: medium skin tone", "\u{1F481}\u{1F3FE}": "person tipping hand: medium-dark skin tone", "\u{1F481}\u{1F3FF}": "person tipping hand: dark skin tone", "\u{1F481}\u200D\u2642\uFE0F": "man tipping hand", "\u{1F481}\u200D\u2642": "man tipping hand", "\u{1F481}\u{1F3FB}\u200D\u2642\uFE0F": "man tipping hand: light skin tone", "\u{1F481}\u{1F3FB}\u200D\u2642": "man tipping hand: light skin tone", "\u{1F481}\u{1F3FC}\u200D\u2642\uFE0F": "man tipping hand: medium-light skin tone", "\u{1F481}\u{1F3FC}\u200D\u2642": "man tipping hand: medium-light skin tone", "\u{1F481}\u{1F3FD}\u200D\u2642\uFE0F": "man tipping hand: medium skin tone", "\u{1F481}\u{1F3FD}\u200D\u2642": "man tipping hand: medium skin tone", "\u{1F481}\u{1F3FE}\u200D\u2642\uFE0F": "man tipping hand: medium-dark skin tone", "\u{1F481}\u{1F3FE}\u200D\u2642": "man tipping hand: medium-dark skin tone", "\u{1F481}\u{1F3FF}\u200D\u2642\uFE0F": "man tipping hand: dark skin tone", "\u{1F481}\u{1F3FF}\u200D\u2642": "man tipping hand: dark skin tone", "\u{1F481}\u200D\u2640\uFE0F": "woman tipping hand", "\u{1F481}\u200D\u2640": "woman tipping hand", "\u{1F481}\u{1F3FB}\u200D\u2640\uFE0F": "woman tipping hand: light skin tone", "\u{1F481}\u{1F3FB}\u200D\u2640": "woman tipping hand: light skin tone", "\u{1F481}\u{1F3FC}\u200D\u2640\uFE0F": "woman tipping hand: medium-light skin tone", "\u{1F481}\u{1F3FC}\u200D\u2640": "woman tipping hand: medium-light skin tone", "\u{1F481}\u{1F3FD}\u200D\u2640\uFE0F": "woman tipping hand: medium skin tone", "\u{1F481}\u{1F3FD}\u200D\u2640": "woman tipping hand: medium skin tone", "\u{1F481}\u{1F3FE}\u200D\u2640\uFE0F": "woman tipping hand: medium-dark skin tone", "\u{1F481}\u{1F3FE}\u200D\u2640": "woman tipping hand: medium-dark skin tone", "\u{1F481}\u{1F3FF}\u200D\u2640\uFE0F": "woman tipping hand: dark skin tone", "\u{1F481}\u{1F3FF}\u200D\u2640": "woman tipping hand: dark skin tone", "\u{1F64B}": "person raising hand", "\u{1F64B}\u{1F3FB}": "person raising hand: light skin tone", "\u{1F64B}\u{1F3FC}": "person raising hand: medium-light skin tone", "\u{1F64B}\u{1F3FD}": "person raising hand: medium skin tone", "\u{1F64B}\u{1F3FE}": "person raising hand: medium-dark skin tone", "\u{1F64B}\u{1F3FF}": "person raising hand: dark skin tone", "\u{1F64B}\u200D\u2642\uFE0F": "man raising hand", "\u{1F64B}\u200D\u2642": "man raising hand", "\u{1F64B}\u{1F3FB}\u200D\u2642\uFE0F": "man raising hand: light skin tone", "\u{1F64B}\u{1F3FB}\u200D\u2642": "man raising hand: light skin tone", "\u{1F64B}\u{1F3FC}\u200D\u2642\uFE0F": "man raising hand: medium-light skin tone", "\u{1F64B}\u{1F3FC}\u200D\u2642": "man raising hand: medium-light skin tone", "\u{1F64B}\u{1F3FD}\u200D\u2642\uFE0F": "man raising hand: medium skin tone", "\u{1F64B}\u{1F3FD}\u200D\u2642": "man raising hand: medium skin tone", "\u{1F64B}\u{1F3FE}\u200D\u2642\uFE0F": "man raising hand: medium-dark skin tone", "\u{1F64B}\u{1F3FE}\u200D\u2642": "man raising hand: medium-dark skin tone", "\u{1F64B}\u{1F3FF}\u200D\u2642\uFE0F": "man raising hand: dark skin tone", "\u{1F64B}\u{1F3FF}\u200D\u2642": "man raising hand: dark skin tone", "\u{1F64B}\u200D\u2640\uFE0F": "woman raising hand", "\u{1F64B}\u200D\u2640": "woman raising hand", "\u{1F64B}\u{1F3FB}\u200D\u2640\uFE0F": "woman raising hand: light skin tone", "\u{1F64B}\u{1F3FB}\u200D\u2640": "woman raising hand: light skin tone", "\u{1F64B}\u{1F3FC}\u200D\u2640\uFE0F": "woman raising hand: medium-light skin tone", "\u{1F64B}\u{1F3FC}\u200D\u2640": "woman raising hand: medium-light skin tone", "\u{1F64B}\u{1F3FD}\u200D\u2640\uFE0F": "woman raising hand: medium skin tone", "\u{1F64B}\u{1F3FD}\u200D\u2640": "woman raising hand: medium skin tone", "\u{1F64B}\u{1F3FE}\u200D\u2640\uFE0F": "woman raising hand: medium-dark skin tone", "\u{1F64B}\u{1F3FE}\u200D\u2640": "woman raising hand: medium-dark skin tone", "\u{1F64B}\u{1F3FF}\u200D\u2640\uFE0F": "woman raising hand: dark skin tone", "\u{1F64B}\u{1F3FF}\u200D\u2640": "woman raising hand: dark skin tone", "\u{1F647}": "person bowing", "\u{1F647}\u{1F3FB}": "person bowing: light skin tone", "\u{1F647}\u{1F3FC}": "person bowing: medium-light skin tone", "\u{1F647}\u{1F3FD}": "person bowing: medium skin tone", "\u{1F647}\u{1F3FE}": "person bowing: medium-dark skin tone", "\u{1F647}\u{1F3FF}": "person bowing: dark skin tone", "\u{1F647}\u200D\u2642\uFE0F": "man bowing", "\u{1F647}\u200D\u2642": "man bowing", "\u{1F647}\u{1F3FB}\u200D\u2642\uFE0F": "man bowing: light skin tone", "\u{1F647}\u{1F3FB}\u200D\u2642": "man bowing: light skin tone", "\u{1F647}\u{1F3FC}\u200D\u2642\uFE0F": "man bowing: medium-light skin tone", "\u{1F647}\u{1F3FC}\u200D\u2642": "man bowing: medium-light skin tone", "\u{1F647}\u{1F3FD}\u200D\u2642\uFE0F": "man bowing: medium skin tone", "\u{1F647}\u{1F3FD}\u200D\u2642": "man bowing: medium skin tone", "\u{1F647}\u{1F3FE}\u200D\u2642\uFE0F": "man bowing: medium-dark skin tone", "\u{1F647}\u{1F3FE}\u200D\u2642": "man bowing: medium-dark skin tone", "\u{1F647}\u{1F3FF}\u200D\u2642\uFE0F": "man bowing: dark skin tone", "\u{1F647}\u{1F3FF}\u200D\u2642": "man bowing: dark skin tone", "\u{1F647}\u200D\u2640\uFE0F": "woman bowing", "\u{1F647}\u200D\u2640": "woman bowing", "\u{1F647}\u{1F3FB}\u200D\u2640\uFE0F": "woman bowing: light skin tone", "\u{1F647}\u{1F3FB}\u200D\u2640": "woman bowing: light skin tone", "\u{1F647}\u{1F3FC}\u200D\u2640\uFE0F": "woman bowing: medium-light skin tone", "\u{1F647}\u{1F3FC}\u200D\u2640": "woman bowing: medium-light skin tone", "\u{1F647}\u{1F3FD}\u200D\u2640\uFE0F": "woman bowing: medium skin tone", "\u{1F647}\u{1F3FD}\u200D\u2640": "woman bowing: medium skin tone", "\u{1F647}\u{1F3FE}\u200D\u2640\uFE0F": "woman bowing: medium-dark skin tone", "\u{1F647}\u{1F3FE}\u200D\u2640": "woman bowing: medium-dark skin tone", "\u{1F647}\u{1F3FF}\u200D\u2640\uFE0F": "woman bowing: dark skin tone", "\u{1F647}\u{1F3FF}\u200D\u2640": "woman bowing: dark skin tone", "\u{1F926}": "person facepalming", "\u{1F926}\u{1F3FB}": "person facepalming: light skin tone", "\u{1F926}\u{1F3FC}": "person facepalming: medium-light skin tone", "\u{1F926}\u{1F3FD}": "person facepalming: medium skin tone", "\u{1F926}\u{1F3FE}": "person facepalming: medium-dark skin tone", "\u{1F926}\u{1F3FF}": "person facepalming: dark skin tone", "\u{1F926}\u200D\u2642\uFE0F": "man facepalming", "\u{1F926}\u200D\u2642": "man facepalming", "\u{1F926}\u{1F3FB}\u200D\u2642\uFE0F": "man facepalming: light skin tone", "\u{1F926}\u{1F3FB}\u200D\u2642": "man facepalming: light skin tone", "\u{1F926}\u{1F3FC}\u200D\u2642\uFE0F": "man facepalming: medium-light skin tone", "\u{1F926}\u{1F3FC}\u200D\u2642": "man facepalming: medium-light skin tone", "\u{1F926}\u{1F3FD}\u200D\u2642\uFE0F": "man facepalming: medium skin tone", "\u{1F926}\u{1F3FD}\u200D\u2642": "man facepalming: medium skin tone", "\u{1F926}\u{1F3FE}\u200D\u2642\uFE0F": "man facepalming: medium-dark skin tone", "\u{1F926}\u{1F3FE}\u200D\u2642": "man facepalming: medium-dark skin tone", "\u{1F926}\u{1F3FF}\u200D\u2642\uFE0F": "man facepalming: dark skin tone", "\u{1F926}\u{1F3FF}\u200D\u2642": "man facepalming: dark skin tone", "\u{1F926}\u200D\u2640\uFE0F": "woman facepalming", "\u{1F926}\u200D\u2640": "woman facepalming", "\u{1F926}\u{1F3FB}\u200D\u2640\uFE0F": "woman facepalming: light skin tone", "\u{1F926}\u{1F3FB}\u200D\u2640": "woman facepalming: light skin tone", "\u{1F926}\u{1F3FC}\u200D\u2640\uFE0F": "woman facepalming: medium-light skin tone", "\u{1F926}\u{1F3FC}\u200D\u2640": "woman facepalming: medium-light skin tone", "\u{1F926}\u{1F3FD}\u200D\u2640\uFE0F": "woman facepalming: medium skin tone", "\u{1F926}\u{1F3FD}\u200D\u2640": "woman facepalming: medium skin tone", "\u{1F926}\u{1F3FE}\u200D\u2640\uFE0F": "woman facepalming: medium-dark skin tone", "\u{1F926}\u{1F3FE}\u200D\u2640": "woman facepalming: medium-dark skin tone", "\u{1F926}\u{1F3FF}\u200D\u2640\uFE0F": "woman facepalming: dark skin tone", "\u{1F926}\u{1F3FF}\u200D\u2640": "woman facepalming: dark skin tone", "\u{1F937}": "person shrugging", "\u{1F937}\u{1F3FB}": "person shrugging: light skin tone", "\u{1F937}\u{1F3FC}": "person shrugging: medium-light skin tone", "\u{1F937}\u{1F3FD}": "person shrugging: medium skin tone", "\u{1F937}\u{1F3FE}": "person shrugging: medium-dark skin tone", "\u{1F937}\u{1F3FF}": "person shrugging: dark skin tone", "\u{1F937}\u200D\u2642\uFE0F": "man shrugging", "\u{1F937}\u200D\u2642": "man shrugging", "\u{1F937}\u{1F3FB}\u200D\u2642\uFE0F": "man shrugging: light skin tone", "\u{1F937}\u{1F3FB}\u200D\u2642": "man shrugging: light skin tone", "\u{1F937}\u{1F3FC}\u200D\u2642\uFE0F": "man shrugging: medium-light skin tone", "\u{1F937}\u{1F3FC}\u200D\u2642": "man shrugging: medium-light skin tone", "\u{1F937}\u{1F3FD}\u200D\u2642\uFE0F": "man shrugging: medium skin tone", "\u{1F937}\u{1F3FD}\u200D\u2642": "man shrugging: medium skin tone", "\u{1F937}\u{1F3FE}\u200D\u2642\uFE0F": "man shrugging: medium-dark skin tone", "\u{1F937}\u{1F3FE}\u200D\u2642": "man shrugging: medium-dark skin tone", "\u{1F937}\u{1F3FF}\u200D\u2642\uFE0F": "man shrugging: dark skin tone", "\u{1F937}\u{1F3FF}\u200D\u2642": "man shrugging: dark skin tone", "\u{1F937}\u200D\u2640\uFE0F": "woman shrugging", "\u{1F937}\u200D\u2640": "woman shrugging", "\u{1F937}\u{1F3FB}\u200D\u2640\uFE0F": "woman shrugging: light skin tone", "\u{1F937}\u{1F3FB}\u200D\u2640": "woman shrugging: light skin tone", "\u{1F937}\u{1F3FC}\u200D\u2640\uFE0F": "woman shrugging: medium-light skin tone", "\u{1F937}\u{1F3FC}\u200D\u2640": "woman shrugging: medium-light skin tone", "\u{1F937}\u{1F3FD}\u200D\u2640\uFE0F": "woman shrugging: medium skin tone", "\u{1F937}\u{1F3FD}\u200D\u2640": "woman shrugging: medium skin tone", "\u{1F937}\u{1F3FE}\u200D\u2640\uFE0F": "woman shrugging: medium-dark skin tone", "\u{1F937}\u{1F3FE}\u200D\u2640": "woman shrugging: medium-dark skin tone", "\u{1F937}\u{1F3FF}\u200D\u2640\uFE0F": "woman shrugging: dark skin tone", "\u{1F937}\u{1F3FF}\u200D\u2640": "woman shrugging: dark skin tone", "\u{1F486}": "person getting massage", "\u{1F486}\u{1F3FB}": "person getting massage: light skin tone", "\u{1F486}\u{1F3FC}": "person getting massage: medium-light skin tone", "\u{1F486}\u{1F3FD}": "person getting massage: medium skin tone", "\u{1F486}\u{1F3FE}": "person getting massage: medium-dark skin tone", "\u{1F486}\u{1F3FF}": "person getting massage: dark skin tone", "\u{1F486}\u200D\u2642\uFE0F": "man getting massage", "\u{1F486}\u200D\u2642": "man getting massage", "\u{1F486}\u{1F3FB}\u200D\u2642\uFE0F": "man getting massage: light skin tone", "\u{1F486}\u{1F3FB}\u200D\u2642": "man getting massage: light skin tone", "\u{1F486}\u{1F3FC}\u200D\u2642\uFE0F": "man getting massage: medium-light skin tone", "\u{1F486}\u{1F3FC}\u200D\u2642": "man getting massage: medium-light skin tone", "\u{1F486}\u{1F3FD}\u200D\u2642\uFE0F": "man getting massage: medium skin tone", "\u{1F486}\u{1F3FD}\u200D\u2642": "man getting massage: medium skin tone", "\u{1F486}\u{1F3FE}\u200D\u2642\uFE0F": "man getting massage: medium-dark skin tone", "\u{1F486}\u{1F3FE}\u200D\u2642": "man getting massage: medium-dark skin tone", "\u{1F486}\u{1F3FF}\u200D\u2642\uFE0F": "man getting massage: dark skin tone", "\u{1F486}\u{1F3FF}\u200D\u2642": "man getting massage: dark skin tone", "\u{1F486}\u200D\u2640\uFE0F": "woman getting massage", "\u{1F486}\u200D\u2640": "woman getting massage", "\u{1F486}\u{1F3FB}\u200D\u2640\uFE0F": "woman getting massage: light skin tone", "\u{1F486}\u{1F3FB}\u200D\u2640": "woman getting massage: light skin tone", "\u{1F486}\u{1F3FC}\u200D\u2640\uFE0F": "woman getting massage: medium-light skin tone", "\u{1F486}\u{1F3FC}\u200D\u2640": "woman getting massage: medium-light skin tone", "\u{1F486}\u{1F3FD}\u200D\u2640\uFE0F": "woman getting massage: medium skin tone", "\u{1F486}\u{1F3FD}\u200D\u2640": "woman getting massage: medium skin tone", "\u{1F486}\u{1F3FE}\u200D\u2640\uFE0F": "woman getting massage: medium-dark skin tone", "\u{1F486}\u{1F3FE}\u200D\u2640": "woman getting massage: medium-dark skin tone", "\u{1F486}\u{1F3FF}\u200D\u2640\uFE0F": "woman getting massage: dark skin tone", "\u{1F486}\u{1F3FF}\u200D\u2640": "woman getting massage: dark skin tone", "\u{1F487}": "person getting haircut", "\u{1F487}\u{1F3FB}": "person getting haircut: light skin tone", "\u{1F487}\u{1F3FC}": "person getting haircut: medium-light skin tone", "\u{1F487}\u{1F3FD}": "person getting haircut: medium skin tone", "\u{1F487}\u{1F3FE}": "person getting haircut: medium-dark skin tone", "\u{1F487}\u{1F3FF}": "person getting haircut: dark skin tone", "\u{1F487}\u200D\u2642\uFE0F": "man getting haircut", "\u{1F487}\u200D\u2642": "man getting haircut", "\u{1F487}\u{1F3FB}\u200D\u2642\uFE0F": "man getting haircut: light skin tone", "\u{1F487}\u{1F3FB}\u200D\u2642": "man getting haircut: light skin tone", "\u{1F487}\u{1F3FC}\u200D\u2642\uFE0F": "man getting haircut: medium-light skin tone", "\u{1F487}\u{1F3FC}\u200D\u2642": "man getting haircut: medium-light skin tone", "\u{1F487}\u{1F3FD}\u200D\u2642\uFE0F": "man getting haircut: medium skin tone", "\u{1F487}\u{1F3FD}\u200D\u2642": "man getting haircut: medium skin tone", "\u{1F487}\u{1F3FE}\u200D\u2642\uFE0F": "man getting haircut: medium-dark skin tone", "\u{1F487}\u{1F3FE}\u200D\u2642": "man getting haircut: medium-dark skin tone", "\u{1F487}\u{1F3FF}\u200D\u2642\uFE0F": "man getting haircut: dark skin tone", "\u{1F487}\u{1F3FF}\u200D\u2642": "man getting haircut: dark skin tone", "\u{1F487}\u200D\u2640\uFE0F": "woman getting haircut", "\u{1F487}\u200D\u2640": "woman getting haircut", "\u{1F487}\u{1F3FB}\u200D\u2640\uFE0F": "woman getting haircut: light skin tone", "\u{1F487}\u{1F3FB}\u200D\u2640": "woman getting haircut: light skin tone", "\u{1F487}\u{1F3FC}\u200D\u2640\uFE0F": "woman getting haircut: medium-light skin tone", "\u{1F487}\u{1F3FC}\u200D\u2640": "woman getting haircut: medium-light skin tone", "\u{1F487}\u{1F3FD}\u200D\u2640\uFE0F": "woman getting haircut: medium skin tone", "\u{1F487}\u{1F3FD}\u200D\u2640": "woman getting haircut: medium skin tone", "\u{1F487}\u{1F3FE}\u200D\u2640\uFE0F": "woman getting haircut: medium-dark skin tone", "\u{1F487}\u{1F3FE}\u200D\u2640": "woman getting haircut: medium-dark skin tone", "\u{1F487}\u{1F3FF}\u200D\u2640\uFE0F": "woman getting haircut: dark skin tone", "\u{1F487}\u{1F3FF}\u200D\u2640": "woman getting haircut: dark skin tone", "\u{1F6B6}": "person walking", "\u{1F6B6}\u{1F3FB}": "person walking: light skin tone", "\u{1F6B6}\u{1F3FC}": "person walking: medium-light skin tone", "\u{1F6B6}\u{1F3FD}": "person walking: medium skin tone", "\u{1F6B6}\u{1F3FE}": "person walking: medium-dark skin tone", "\u{1F6B6}\u{1F3FF}": "person walking: dark skin tone", "\u{1F6B6}\u200D\u2642\uFE0F": "man walking", "\u{1F6B6}\u200D\u2642": "man walking", "\u{1F6B6}\u{1F3FB}\u200D\u2642\uFE0F": "man walking: light skin tone", "\u{1F6B6}\u{1F3FB}\u200D\u2642": "man walking: light skin tone", "\u{1F6B6}\u{1F3FC}\u200D\u2642\uFE0F": "man walking: medium-light skin tone", "\u{1F6B6}\u{1F3FC}\u200D\u2642": "man walking: medium-light skin tone", "\u{1F6B6}\u{1F3FD}\u200D\u2642\uFE0F": "man walking: medium skin tone", "\u{1F6B6}\u{1F3FD}\u200D\u2642": "man walking: medium skin tone", "\u{1F6B6}\u{1F3FE}\u200D\u2642\uFE0F": "man walking: medium-dark skin tone", "\u{1F6B6}\u{1F3FE}\u200D\u2642": "man walking: medium-dark skin tone", "\u{1F6B6}\u{1F3FF}\u200D\u2642\uFE0F": "man walking: dark skin tone", "\u{1F6B6}\u{1F3FF}\u200D\u2642": "man walking: dark skin tone", "\u{1F6B6}\u200D\u2640\uFE0F": "woman walking", "\u{1F6B6}\u200D\u2640": "woman walking", "\u{1F6B6}\u{1F3FB}\u200D\u2640\uFE0F": "woman walking: light skin tone", "\u{1F6B6}\u{1F3FB}\u200D\u2640": "woman walking: light skin tone", "\u{1F6B6}\u{1F3FC}\u200D\u2640\uFE0F": "woman walking: medium-light skin tone", "\u{1F6B6}\u{1F3FC}\u200D\u2640": "woman walking: medium-light skin tone", "\u{1F6B6}\u{1F3FD}\u200D\u2640\uFE0F": "woman walking: medium skin tone", "\u{1F6B6}\u{1F3FD}\u200D\u2640": "woman walking: medium skin tone", "\u{1F6B6}\u{1F3FE}\u200D\u2640\uFE0F": "woman walking: medium-dark skin tone", "\u{1F6B6}\u{1F3FE}\u200D\u2640": "woman walking: medium-dark skin tone", "\u{1F6B6}\u{1F3FF}\u200D\u2640\uFE0F": "woman walking: dark skin tone", "\u{1F6B6}\u{1F3FF}\u200D\u2640": "woman walking: dark skin tone", "\u{1F3C3}": "person running", "\u{1F3C3}\u{1F3FB}": "person running: light skin tone", "\u{1F3C3}\u{1F3FC}": "person running: medium-light skin tone", "\u{1F3C3}\u{1F3FD}": "person running: medium skin tone", "\u{1F3C3}\u{1F3FE}": "person running: medium-dark skin tone", "\u{1F3C3}\u{1F3FF}": "person running: dark skin tone", "\u{1F3C3}\u200D\u2642\uFE0F": "man running", "\u{1F3C3}\u200D\u2642": "man running", "\u{1F3C3}\u{1F3FB}\u200D\u2642\uFE0F": "man running: light skin tone", "\u{1F3C3}\u{1F3FB}\u200D\u2642": "man running: light skin tone", "\u{1F3C3}\u{1F3FC}\u200D\u2642\uFE0F": "man running: medium-light skin tone", "\u{1F3C3}\u{1F3FC}\u200D\u2642": "man running: medium-light skin tone", "\u{1F3C3}\u{1F3FD}\u200D\u2642\uFE0F": "man running: medium skin tone", "\u{1F3C3}\u{1F3FD}\u200D\u2642": "man running: medium skin tone", "\u{1F3C3}\u{1F3FE}\u200D\u2642\uFE0F": "man running: medium-dark skin tone", "\u{1F3C3}\u{1F3FE}\u200D\u2642": "man running: medium-dark skin tone", "\u{1F3C3}\u{1F3FF}\u200D\u2642\uFE0F": "man running: dark skin tone", "\u{1F3C3}\u{1F3FF}\u200D\u2642": "man running: dark skin tone", "\u{1F3C3}\u200D\u2640\uFE0F": "woman running", "\u{1F3C3}\u200D\u2640": "woman running", "\u{1F3C3}\u{1F3FB}\u200D\u2640\uFE0F": "woman running: light skin tone", "\u{1F3C3}\u{1F3FB}\u200D\u2640": "woman running: light skin tone", "\u{1F3C3}\u{1F3FC}\u200D\u2640\uFE0F": "woman running: medium-light skin tone", "\u{1F3C3}\u{1F3FC}\u200D\u2640": "woman running: medium-light skin tone", "\u{1F3C3}\u{1F3FD}\u200D\u2640\uFE0F": "woman running: medium skin tone", "\u{1F3C3}\u{1F3FD}\u200D\u2640": "woman running: medium skin tone", "\u{1F3C3}\u{1F3FE}\u200D\u2640\uFE0F": "woman running: medium-dark skin tone", "\u{1F3C3}\u{1F3FE}\u200D\u2640": "woman running: medium-dark skin tone", "\u{1F3C3}\u{1F3FF}\u200D\u2640\uFE0F": "woman running: dark skin tone", "\u{1F3C3}\u{1F3FF}\u200D\u2640": "woman running: dark skin tone", "\u{1F483}": "woman dancing", "\u{1F483}\u{1F3FB}": "woman dancing: light skin tone", "\u{1F483}\u{1F3FC}": "woman dancing: medium-light skin tone", "\u{1F483}\u{1F3FD}": "woman dancing: medium skin tone", "\u{1F483}\u{1F3FE}": "woman dancing: medium-dark skin tone", "\u{1F483}\u{1F3FF}": "woman dancing: dark skin tone", "\u{1F57A}": "man dancing", "\u{1F57A}\u{1F3FB}": "man dancing: light skin tone", "\u{1F57A}\u{1F3FC}": "man dancing: medium-light skin tone", "\u{1F57A}\u{1F3FD}": "man dancing: medium skin tone", "\u{1F57A}\u{1F3FE}": "man dancing: medium-dark skin tone", "\u{1F57A}\u{1F3FF}": "man dancing: dark skin tone", "\u{1F46F}": "people with bunny ears", "\u{1F46F}\u200D\u2642\uFE0F": "men with bunny ears", "\u{1F46F}\u200D\u2642": "men with bunny ears", "\u{1F46F}\u200D\u2640\uFE0F": "women with bunny ears", "\u{1F46F}\u200D\u2640": "women with bunny ears", "\u{1F9D6}": "person in steamy room", "\u{1F9D6}\u{1F3FB}": "person in steamy room: light skin tone", "\u{1F9D6}\u{1F3FC}": "person in steamy room: medium-light skin tone", "\u{1F9D6}\u{1F3FD}": "person in steamy room: medium skin tone", "\u{1F9D6}\u{1F3FE}": "person in steamy room: medium-dark skin tone", "\u{1F9D6}\u{1F3FF}": "person in steamy room: dark skin tone", "\u{1F9D6}\u200D\u2640\uFE0F": "woman in steamy room", "\u{1F9D6}\u200D\u2640": "woman in steamy room", "\u{1F9D6}\u{1F3FB}\u200D\u2640\uFE0F": "woman in steamy room: light skin tone", "\u{1F9D6}\u{1F3FB}\u200D\u2640": "woman in steamy room: light skin tone", "\u{1F9D6}\u{1F3FC}\u200D\u2640\uFE0F": "woman in steamy room: medium-light skin tone", "\u{1F9D6}\u{1F3FC}\u200D\u2640": "woman in steamy room: medium-light skin tone", "\u{1F9D6}\u{1F3FD}\u200D\u2640\uFE0F": "woman in steamy room: medium skin tone", "\u{1F9D6}\u{1F3FD}\u200D\u2640": "woman in steamy room: medium skin tone", "\u{1F9D6}\u{1F3FE}\u200D\u2640\uFE0F": "woman in steamy room: medium-dark skin tone", "\u{1F9D6}\u{1F3FE}\u200D\u2640": "woman in steamy room: medium-dark skin tone", "\u{1F9D6}\u{1F3FF}\u200D\u2640\uFE0F": "woman in steamy room: dark skin tone", "\u{1F9D6}\u{1F3FF}\u200D\u2640": "woman in steamy room: dark skin tone", "\u{1F9D6}\u200D\u2642\uFE0F": "man in steamy room", "\u{1F9D6}\u200D\u2642": "man in steamy room", "\u{1F9D6}\u{1F3FB}\u200D\u2642\uFE0F": "man in steamy room: light skin tone", "\u{1F9D6}\u{1F3FB}\u200D\u2642": "man in steamy room: light skin tone", "\u{1F9D6}\u{1F3FC}\u200D\u2642\uFE0F": "man in steamy room: medium-light skin tone", "\u{1F9D6}\u{1F3FC}\u200D\u2642": "man in steamy room: medium-light skin tone", "\u{1F9D6}\u{1F3FD}\u200D\u2642\uFE0F": "man in steamy room: medium skin tone", "\u{1F9D6}\u{1F3FD}\u200D\u2642": "man in steamy room: medium skin tone", "\u{1F9D6}\u{1F3FE}\u200D\u2642\uFE0F": "man in steamy room: medium-dark skin tone", "\u{1F9D6}\u{1F3FE}\u200D\u2642": "man in steamy room: medium-dark skin tone", "\u{1F9D6}\u{1F3FF}\u200D\u2642\uFE0F": "man in steamy room: dark skin tone", "\u{1F9D6}\u{1F3FF}\u200D\u2642": "man in steamy room: dark skin tone", "\u{1F9D7}": "person climbing", "\u{1F9D7}\u{1F3FB}": "person climbing: light skin tone", "\u{1F9D7}\u{1F3FC}": "person climbing: medium-light skin tone", "\u{1F9D7}\u{1F3FD}": "person climbing: medium skin tone", "\u{1F9D7}\u{1F3FE}": "person climbing: medium-dark skin tone", "\u{1F9D7}\u{1F3FF}": "person climbing: dark skin tone", "\u{1F9D7}\u200D\u2640\uFE0F": "woman climbing", "\u{1F9D7}\u200D\u2640": "woman climbing", "\u{1F9D7}\u{1F3FB}\u200D\u2640\uFE0F": "woman climbing: light skin tone", "\u{1F9D7}\u{1F3FB}\u200D\u2640": "woman climbing: light skin tone", "\u{1F9D7}\u{1F3FC}\u200D\u2640\uFE0F": "woman climbing: medium-light skin tone", "\u{1F9D7}\u{1F3FC}\u200D\u2640": "woman climbing: medium-light skin tone", "\u{1F9D7}\u{1F3FD}\u200D\u2640\uFE0F": "woman climbing: medium skin tone", "\u{1F9D7}\u{1F3FD}\u200D\u2640": "woman climbing: medium skin tone", "\u{1F9D7}\u{1F3FE}\u200D\u2640\uFE0F": "woman climbing: medium-dark skin tone", "\u{1F9D7}\u{1F3FE}\u200D\u2640": "woman climbing: medium-dark skin tone", "\u{1F9D7}\u{1F3FF}\u200D\u2640\uFE0F": "woman climbing: dark skin tone", "\u{1F9D7}\u{1F3FF}\u200D\u2640": "woman climbing: dark skin tone", "\u{1F9D7}\u200D\u2642\uFE0F": "man climbing", "\u{1F9D7}\u200D\u2642": "man climbing", "\u{1F9D7}\u{1F3FB}\u200D\u2642\uFE0F": "man climbing: light skin tone", "\u{1F9D7}\u{1F3FB}\u200D\u2642": "man climbing: light skin tone", "\u{1F9D7}\u{1F3FC}\u200D\u2642\uFE0F": "man climbing: medium-light skin tone", "\u{1F9D7}\u{1F3FC}\u200D\u2642": "man climbing: medium-light skin tone", "\u{1F9D7}\u{1F3FD}\u200D\u2642\uFE0F": "man climbing: medium skin tone", "\u{1F9D7}\u{1F3FD}\u200D\u2642": "man climbing: medium skin tone", "\u{1F9D7}\u{1F3FE}\u200D\u2642\uFE0F": "man climbing: medium-dark skin tone", "\u{1F9D7}\u{1F3FE}\u200D\u2642": "man climbing: medium-dark skin tone", "\u{1F9D7}\u{1F3FF}\u200D\u2642\uFE0F": "man climbing: dark skin tone", "\u{1F9D7}\u{1F3FF}\u200D\u2642": "man climbing: dark skin tone", "\u{1F9D8}": "person in lotus position", "\u{1F9D8}\u{1F3FB}": "person in lotus position: light skin tone", "\u{1F9D8}\u{1F3FC}": "person in lotus position: medium-light skin tone", "\u{1F9D8}\u{1F3FD}": "person in lotus position: medium skin tone", "\u{1F9D8}\u{1F3FE}": "person in lotus position: medium-dark skin tone", "\u{1F9D8}\u{1F3FF}": "person in lotus position: dark skin tone", "\u{1F9D8}\u200D\u2640\uFE0F": "woman in lotus position", "\u{1F9D8}\u200D\u2640": "woman in lotus position", "\u{1F9D8}\u{1F3FB}\u200D\u2640\uFE0F": "woman in lotus position: light skin tone", "\u{1F9D8}\u{1F3FB}\u200D\u2640": "woman in lotus position: light skin tone", "\u{1F9D8}\u{1F3FC}\u200D\u2640\uFE0F": "woman in lotus position: medium-light skin tone", "\u{1F9D8}\u{1F3FC}\u200D\u2640": "woman in lotus position: medium-light skin tone", "\u{1F9D8}\u{1F3FD}\u200D\u2640\uFE0F": "woman in lotus position: medium skin tone", "\u{1F9D8}\u{1F3FD}\u200D\u2640": "woman in lotus position: medium skin tone", "\u{1F9D8}\u{1F3FE}\u200D\u2640\uFE0F": "woman in lotus position: medium-dark skin tone", "\u{1F9D8}\u{1F3FE}\u200D\u2640": "woman in lotus position: medium-dark skin tone", "\u{1F9D8}\u{1F3FF}\u200D\u2640\uFE0F": "woman in lotus position: dark skin tone", "\u{1F9D8}\u{1F3FF}\u200D\u2640": "woman in lotus position: dark skin tone", "\u{1F9D8}\u200D\u2642\uFE0F": "man in lotus position", "\u{1F9D8}\u200D\u2642": "man in lotus position", "\u{1F9D8}\u{1F3FB}\u200D\u2642\uFE0F": "man in lotus position: light skin tone", "\u{1F9D8}\u{1F3FB}\u200D\u2642": "man in lotus position: light skin tone", "\u{1F9D8}\u{1F3FC}\u200D\u2642\uFE0F": "man in lotus position: medium-light skin tone", "\u{1F9D8}\u{1F3FC}\u200D\u2642": "man in lotus position: medium-light skin tone", "\u{1F9D8}\u{1F3FD}\u200D\u2642\uFE0F": "man in lotus position: medium skin tone", "\u{1F9D8}\u{1F3FD}\u200D\u2642": "man in lotus position: medium skin tone", "\u{1F9D8}\u{1F3FE}\u200D\u2642\uFE0F": "man in lotus position: medium-dark skin tone", "\u{1F9D8}\u{1F3FE}\u200D\u2642": "man in lotus position: medium-dark skin tone", "\u{1F9D8}\u{1F3FF}\u200D\u2642\uFE0F": "man in lotus position: dark skin tone", "\u{1F9D8}\u{1F3FF}\u200D\u2642": "man in lotus position: dark skin tone", "\u{1F6C0}": "person taking bath", "\u{1F6C0}\u{1F3FB}": "person taking bath: light skin tone", "\u{1F6C0}\u{1F3FC}": "person taking bath: medium-light skin tone", "\u{1F6C0}\u{1F3FD}": "person taking bath: medium skin tone", "\u{1F6C0}\u{1F3FE}": "person taking bath: medium-dark skin tone", "\u{1F6C0}\u{1F3FF}": "person taking bath: dark skin tone", "\u{1F6CC}": "person in bed", "\u{1F6CC}\u{1F3FB}": "person in bed: light skin tone", "\u{1F6CC}\u{1F3FC}": "person in bed: medium-light skin tone", "\u{1F6CC}\u{1F3FD}": "person in bed: medium skin tone", "\u{1F6CC}\u{1F3FE}": "person in bed: medium-dark skin tone", "\u{1F6CC}\u{1F3FF}": "person in bed: dark skin tone", "\u{1F574}\uFE0F": "man in suit levitating", "\u{1F574}": "man in suit levitating", "\u{1F574}\u{1F3FB}": "man in suit levitating: light skin tone", "\u{1F574}\u{1F3FC}": "man in suit levitating: medium-light skin tone", "\u{1F574}\u{1F3FD}": "man in suit levitating: medium skin tone", "\u{1F574}\u{1F3FE}": "man in suit levitating: medium-dark skin tone", "\u{1F574}\u{1F3FF}": "man in suit levitating: dark skin tone", "\u{1F5E3}\uFE0F": "speaking head", "\u{1F5E3}": "speaking head", "\u{1F464}": "bust in silhouette", "\u{1F465}": "busts in silhouette", "\u{1F93A}": "person fencing", "\u{1F3C7}": "horse racing", "\u{1F3C7}\u{1F3FB}": "horse racing: light skin tone", "\u{1F3C7}\u{1F3FC}": "horse racing: medium-light skin tone", "\u{1F3C7}\u{1F3FD}": "horse racing: medium skin tone", "\u{1F3C7}\u{1F3FE}": "horse racing: medium-dark skin tone", "\u{1F3C7}\u{1F3FF}": "horse racing: dark skin tone", "\u26F7\uFE0F": "skier", "\u26F7": "skier", "\u{1F3C2}": "snowboarder", "\u{1F3C2}\u{1F3FB}": "snowboarder: light skin tone", "\u{1F3C2}\u{1F3FC}": "snowboarder: medium-light skin tone", "\u{1F3C2}\u{1F3FD}": "snowboarder: medium skin tone", "\u{1F3C2}\u{1F3FE}": "snowboarder: medium-dark skin tone", "\u{1F3C2}\u{1F3FF}": "snowboarder: dark skin tone", "\u{1F3CC}\uFE0F": "person golfing", "\u{1F3CC}": "person golfing", "\u{1F3CC}\u{1F3FB}": "person golfing: light skin tone", "\u{1F3CC}\u{1F3FC}": "person golfing: medium-light skin tone", "\u{1F3CC}\u{1F3FD}": "person golfing: medium skin tone", "\u{1F3CC}\u{1F3FE}": "person golfing: medium-dark skin tone", "\u{1F3CC}\u{1F3FF}": "person golfing: dark skin tone", "\u{1F3CC}\uFE0F\u200D\u2642\uFE0F": "man golfing", "\u{1F3CC}\u200D\u2642\uFE0F": "man golfing", "\u{1F3CC}\uFE0F\u200D\u2642": "man golfing", "\u{1F3CC}\u200D\u2642": "man golfing", "\u{1F3CC}\u{1F3FB}\u200D\u2642\uFE0F": "man golfing: light skin tone", "\u{1F3CC}\u{1F3FB}\u200D\u2642": "man golfing: light skin tone", "\u{1F3CC}\u{1F3FC}\u200D\u2642\uFE0F": "man golfing: medium-light skin tone", "\u{1F3CC}\u{1F3FC}\u200D\u2642": "man golfing: medium-light skin tone", "\u{1F3CC}\u{1F3FD}\u200D\u2642\uFE0F": "man golfing: medium skin tone", "\u{1F3CC}\u{1F3FD}\u200D\u2642": "man golfing: medium skin tone", "\u{1F3CC}\u{1F3FE}\u200D\u2642\uFE0F": "man golfing: medium-dark skin tone", "\u{1F3CC}\u{1F3FE}\u200D\u2642": "man golfing: medium-dark skin tone", "\u{1F3CC}\u{1F3FF}\u200D\u2642\uFE0F": "man golfing: dark skin tone", "\u{1F3CC}\u{1F3FF}\u200D\u2642": "man golfing: dark skin tone", "\u{1F3CC}\uFE0F\u200D\u2640\uFE0F": "woman golfing", "\u{1F3CC}\u200D\u2640\uFE0F": "woman golfing", "\u{1F3CC}\uFE0F\u200D\u2640": "woman golfing", "\u{1F3CC}\u200D\u2640": "woman golfing", "\u{1F3CC}\u{1F3FB}\u200D\u2640\uFE0F": "woman golfing: light skin tone", "\u{1F3CC}\u{1F3FB}\u200D\u2640": "woman golfing: light skin tone", "\u{1F3CC}\u{1F3FC}\u200D\u2640\uFE0F": "woman golfing: medium-light skin tone", "\u{1F3CC}\u{1F3FC}\u200D\u2640": "woman golfing: medium-light skin tone", "\u{1F3CC}\u{1F3FD}\u200D\u2640\uFE0F": "woman golfing: medium skin tone", "\u{1F3CC}\u{1F3FD}\u200D\u2640": "woman golfing: medium skin tone", "\u{1F3CC}\u{1F3FE}\u200D\u2640\uFE0F": "woman golfing: medium-dark skin tone", "\u{1F3CC}\u{1F3FE}\u200D\u2640": "woman golfing: medium-dark skin tone", "\u{1F3CC}\u{1F3FF}\u200D\u2640\uFE0F": "woman golfing: dark skin tone", "\u{1F3CC}\u{1F3FF}\u200D\u2640": "woman golfing: dark skin tone", "\u{1F3C4}": "person surfing", "\u{1F3C4}\u{1F3FB}": "person surfing: light skin tone", "\u{1F3C4}\u{1F3FC}": "person surfing: medium-light skin tone", "\u{1F3C4}\u{1F3FD}": "person surfing: medium skin tone", "\u{1F3C4}\u{1F3FE}": "person surfing: medium-dark skin tone", "\u{1F3C4}\u{1F3FF}": "person surfing: dark skin tone", "\u{1F3C4}\u200D\u2642\uFE0F": "man surfing", "\u{1F3C4}\u200D\u2642": "man surfing", "\u{1F3C4}\u{1F3FB}\u200D\u2642\uFE0F": "man surfing: light skin tone", "\u{1F3C4}\u{1F3FB}\u200D\u2642": "man surfing: light skin tone", "\u{1F3C4}\u{1F3FC}\u200D\u2642\uFE0F": "man surfing: medium-light skin tone", "\u{1F3C4}\u{1F3FC}\u200D\u2642": "man surfing: medium-light skin tone", "\u{1F3C4}\u{1F3FD}\u200D\u2642\uFE0F": "man surfing: medium skin tone", "\u{1F3C4}\u{1F3FD}\u200D\u2642": "man surfing: medium skin tone", "\u{1F3C4}\u{1F3FE}\u200D\u2642\uFE0F": "man surfing: medium-dark skin tone", "\u{1F3C4}\u{1F3FE}\u200D\u2642": "man surfing: medium-dark skin tone", "\u{1F3C4}\u{1F3FF}\u200D\u2642\uFE0F": "man surfing: dark skin tone", "\u{1F3C4}\u{1F3FF}\u200D\u2642": "man surfing: dark skin tone", "\u{1F3C4}\u200D\u2640\uFE0F": "woman surfing", "\u{1F3C4}\u200D\u2640": "woman surfing", "\u{1F3C4}\u{1F3FB}\u200D\u2640\uFE0F": "woman surfing: light skin tone", "\u{1F3C4}\u{1F3FB}\u200D\u2640": "woman surfing: light skin tone", "\u{1F3C4}\u{1F3FC}\u200D\u2640\uFE0F": "woman surfing: medium-light skin tone", "\u{1F3C4}\u{1F3FC}\u200D\u2640": "woman surfing: medium-light skin tone", "\u{1F3C4}\u{1F3FD}\u200D\u2640\uFE0F": "woman surfing: medium skin tone", "\u{1F3C4}\u{1F3FD}\u200D\u2640": "woman surfing: medium skin tone", "\u{1F3C4}\u{1F3FE}\u200D\u2640\uFE0F": "woman surfing: medium-dark skin tone", "\u{1F3C4}\u{1F3FE}\u200D\u2640": "woman surfing: medium-dark skin tone", "\u{1F3C4}\u{1F3FF}\u200D\u2640\uFE0F": "woman surfing: dark skin tone", "\u{1F3C4}\u{1F3FF}\u200D\u2640": "woman surfing: dark skin tone", "\u{1F6A3}": "person rowing boat", "\u{1F6A3}\u{1F3FB}": "person rowing boat: light skin tone", "\u{1F6A3}\u{1F3FC}": "person rowing boat: medium-light skin tone", "\u{1F6A3}\u{1F3FD}": "person rowing boat: medium skin tone", "\u{1F6A3}\u{1F3FE}": "person rowing boat: medium-dark skin tone", "\u{1F6A3}\u{1F3FF}": "person rowing boat: dark skin tone", "\u{1F6A3}\u200D\u2642\uFE0F": "man rowing boat", "\u{1F6A3}\u200D\u2642": "man rowing boat", "\u{1F6A3}\u{1F3FB}\u200D\u2642\uFE0F": "man rowing boat: light skin tone", "\u{1F6A3}\u{1F3FB}\u200D\u2642": "man rowing boat: light skin tone", "\u{1F6A3}\u{1F3FC}\u200D\u2642\uFE0F": "man rowing boat: medium-light skin tone", "\u{1F6A3}\u{1F3FC}\u200D\u2642": "man rowing boat: medium-light skin tone", "\u{1F6A3}\u{1F3FD}\u200D\u2642\uFE0F": "man rowing boat: medium skin tone", "\u{1F6A3}\u{1F3FD}\u200D\u2642": "man rowing boat: medium skin tone", "\u{1F6A3}\u{1F3FE}\u200D\u2642\uFE0F": "man rowing boat: medium-dark skin tone", "\u{1F6A3}\u{1F3FE}\u200D\u2642": "man rowing boat: medium-dark skin tone", "\u{1F6A3}\u{1F3FF}\u200D\u2642\uFE0F": "man rowing boat: dark skin tone", "\u{1F6A3}\u{1F3FF}\u200D\u2642": "man rowing boat: dark skin tone", "\u{1F6A3}\u200D\u2640\uFE0F": "woman rowing boat", "\u{1F6A3}\u200D\u2640": "woman rowing boat", "\u{1F6A3}\u{1F3FB}\u200D\u2640\uFE0F": "woman rowing boat: light skin tone", "\u{1F6A3}\u{1F3FB}\u200D\u2640": "woman rowing boat: light skin tone", "\u{1F6A3}\u{1F3FC}\u200D\u2640\uFE0F": "woman rowing boat: medium-light skin tone", "\u{1F6A3}\u{1F3FC}\u200D\u2640": "woman rowing boat: medium-light skin tone", "\u{1F6A3}\u{1F3FD}\u200D\u2640\uFE0F": "woman rowing boat: medium skin tone", "\u{1F6A3}\u{1F3FD}\u200D\u2640": "woman rowing boat: medium skin tone", "\u{1F6A3}\u{1F3FE}\u200D\u2640\uFE0F": "woman rowing boat: medium-dark skin tone", "\u{1F6A3}\u{1F3FE}\u200D\u2640": "woman rowing boat: medium-dark skin tone", "\u{1F6A3}\u{1F3FF}\u200D\u2640\uFE0F": "woman rowing boat: dark skin tone", "\u{1F6A3}\u{1F3FF}\u200D\u2640": "woman rowing boat: dark skin tone", "\u{1F3CA}": "person swimming", "\u{1F3CA}\u{1F3FB}": "person swimming: light skin tone", "\u{1F3CA}\u{1F3FC}": "person swimming: medium-light skin tone", "\u{1F3CA}\u{1F3FD}": "person swimming: medium skin tone", "\u{1F3CA}\u{1F3FE}": "person swimming: medium-dark skin tone", "\u{1F3CA}\u{1F3FF}": "person swimming: dark skin tone", "\u{1F3CA}\u200D\u2642\uFE0F": "man swimming", "\u{1F3CA}\u200D\u2642": "man swimming", "\u{1F3CA}\u{1F3FB}\u200D\u2642\uFE0F": "man swimming: light skin tone", "\u{1F3CA}\u{1F3FB}\u200D\u2642": "man swimming: light skin tone", "\u{1F3CA}\u{1F3FC}\u200D\u2642\uFE0F": "man swimming: medium-light skin tone", "\u{1F3CA}\u{1F3FC}\u200D\u2642": "man swimming: medium-light skin tone", "\u{1F3CA}\u{1F3FD}\u200D\u2642\uFE0F": "man swimming: medium skin tone", "\u{1F3CA}\u{1F3FD}\u200D\u2642": "man swimming: medium skin tone", "\u{1F3CA}\u{1F3FE}\u200D\u2642\uFE0F": "man swimming: medium-dark skin tone", "\u{1F3CA}\u{1F3FE}\u200D\u2642": "man swimming: medium-dark skin tone", "\u{1F3CA}\u{1F3FF}\u200D\u2642\uFE0F": "man swimming: dark skin tone", "\u{1F3CA}\u{1F3FF}\u200D\u2642": "man swimming: dark skin tone", "\u{1F3CA}\u200D\u2640\uFE0F": "woman swimming", "\u{1F3CA}\u200D\u2640": "woman swimming", "\u{1F3CA}\u{1F3FB}\u200D\u2640\uFE0F": "woman swimming: light skin tone", "\u{1F3CA}\u{1F3FB}\u200D\u2640": "woman swimming: light skin tone", "\u{1F3CA}\u{1F3FC}\u200D\u2640\uFE0F": "woman swimming: medium-light skin tone", "\u{1F3CA}\u{1F3FC}\u200D\u2640": "woman swimming: medium-light skin tone", "\u{1F3CA}\u{1F3FD}\u200D\u2640\uFE0F": "woman swimming: medium skin tone", "\u{1F3CA}\u{1F3FD}\u200D\u2640": "woman swimming: medium skin tone", "\u{1F3CA}\u{1F3FE}\u200D\u2640\uFE0F": "woman swimming: medium-dark skin tone", "\u{1F3CA}\u{1F3FE}\u200D\u2640": "woman swimming: medium-dark skin tone", "\u{1F3CA}\u{1F3FF}\u200D\u2640\uFE0F": "woman swimming: dark skin tone", "\u{1F3CA}\u{1F3FF}\u200D\u2640": "woman swimming: dark skin tone", "\u26F9\uFE0F": "person bouncing ball", "\u26F9": "person bouncing ball", "\u26F9\u{1F3FB}": "person bouncing ball: light skin tone", "\u26F9\u{1F3FC}": "person bouncing ball: medium-light skin tone", "\u26F9\u{1F3FD}": "person bouncing ball: medium skin tone", "\u26F9\u{1F3FE}": "person bouncing ball: medium-dark skin tone", "\u26F9\u{1F3FF}": "person bouncing ball: dark skin tone", "\u26F9\uFE0F\u200D\u2642\uFE0F": "man bouncing ball", "\u26F9\u200D\u2642\uFE0F": "man bouncing ball", "\u26F9\uFE0F\u200D\u2642": "man bouncing ball", "\u26F9\u200D\u2642": "man bouncing ball", "\u26F9\u{1F3FB}\u200D\u2642\uFE0F": "man bouncing ball: light skin tone", "\u26F9\u{1F3FB}\u200D\u2642": "man bouncing ball: light skin tone", "\u26F9\u{1F3FC}\u200D\u2642\uFE0F": "man bouncing ball: medium-light skin tone", "\u26F9\u{1F3FC}\u200D\u2642": "man bouncing ball: medium-light skin tone", "\u26F9\u{1F3FD}\u200D\u2642\uFE0F": "man bouncing ball: medium skin tone", "\u26F9\u{1F3FD}\u200D\u2642": "man bouncing ball: medium skin tone", "\u26F9\u{1F3FE}\u200D\u2642\uFE0F": "man bouncing ball: medium-dark skin tone", "\u26F9\u{1F3FE}\u200D\u2642": "man bouncing ball: medium-dark skin tone", "\u26F9\u{1F3FF}\u200D\u2642\uFE0F": "man bouncing ball: dark skin tone", "\u26F9\u{1F3FF}\u200D\u2642": "man bouncing ball: dark skin tone", "\u26F9\uFE0F\u200D\u2640\uFE0F": "woman bouncing ball", "\u26F9\u200D\u2640\uFE0F": "woman bouncing ball", "\u26F9\uFE0F\u200D\u2640": "woman bouncing ball", "\u26F9\u200D\u2640": "woman bouncing ball", "\u26F9\u{1F3FB}\u200D\u2640\uFE0F": "woman bouncing ball: light skin tone", "\u26F9\u{1F3FB}\u200D\u2640": "woman bouncing ball: light skin tone", "\u26F9\u{1F3FC}\u200D\u2640\uFE0F": "woman bouncing ball: medium-light skin tone", "\u26F9\u{1F3FC}\u200D\u2640": "woman bouncing ball: medium-light skin tone", "\u26F9\u{1F3FD}\u200D\u2640\uFE0F": "woman bouncing ball: medium skin tone", "\u26F9\u{1F3FD}\u200D\u2640": "woman bouncing ball: medium skin tone", "\u26F9\u{1F3FE}\u200D\u2640\uFE0F": "woman bouncing ball: medium-dark skin tone", "\u26F9\u{1F3FE}\u200D\u2640": "woman bouncing ball: medium-dark skin tone", "\u26F9\u{1F3FF}\u200D\u2640\uFE0F": "woman bouncing ball: dark skin tone", "\u26F9\u{1F3FF}\u200D\u2640": "woman bouncing ball: dark skin tone", "\u{1F3CB}\uFE0F": "person lifting weights", "\u{1F3CB}": "person lifting weights", "\u{1F3CB}\u{1F3FB}": "person lifting weights: light skin tone", "\u{1F3CB}\u{1F3FC}": "person lifting weights: medium-light skin tone", "\u{1F3CB}\u{1F3FD}": "person lifting weights: medium skin tone", "\u{1F3CB}\u{1F3FE}": "person lifting weights: medium-dark skin tone", "\u{1F3CB}\u{1F3FF}": "person lifting weights: dark skin tone", "\u{1F3CB}\uFE0F\u200D\u2642\uFE0F": "man lifting weights", "\u{1F3CB}\u200D\u2642\uFE0F": "man lifting weights", "\u{1F3CB}\uFE0F\u200D\u2642": "man lifting weights", "\u{1F3CB}\u200D\u2642": "man lifting weights", "\u{1F3CB}\u{1F3FB}\u200D\u2642\uFE0F": "man lifting weights: light skin tone", "\u{1F3CB}\u{1F3FB}\u200D\u2642": "man lifting weights: light skin tone", "\u{1F3CB}\u{1F3FC}\u200D\u2642\uFE0F": "man lifting weights: medium-light skin tone", "\u{1F3CB}\u{1F3FC}\u200D\u2642": "man lifting weights: medium-light skin tone", "\u{1F3CB}\u{1F3FD}\u200D\u2642\uFE0F": "man lifting weights: medium skin tone", "\u{1F3CB}\u{1F3FD}\u200D\u2642": "man lifting weights: medium skin tone", "\u{1F3CB}\u{1F3FE}\u200D\u2642\uFE0F": "man lifting weights: medium-dark skin tone", "\u{1F3CB}\u{1F3FE}\u200D\u2642": "man lifting weights: medium-dark skin tone", "\u{1F3CB}\u{1F3FF}\u200D\u2642\uFE0F": "man lifting weights: dark skin tone", "\u{1F3CB}\u{1F3FF}\u200D\u2642": "man lifting weights: dark skin tone", "\u{1F3CB}\uFE0F\u200D\u2640\uFE0F": "woman lifting weights", "\u{1F3CB}\u200D\u2640\uFE0F": "woman lifting weights", "\u{1F3CB}\uFE0F\u200D\u2640": "woman lifting weights", "\u{1F3CB}\u200D\u2640": "woman lifting weights", "\u{1F3CB}\u{1F3FB}\u200D\u2640\uFE0F": "woman lifting weights: light skin tone", "\u{1F3CB}\u{1F3FB}\u200D\u2640": "woman lifting weights: light skin tone", "\u{1F3CB}\u{1F3FC}\u200D\u2640\uFE0F": "woman lifting weights: medium-light skin tone", "\u{1F3CB}\u{1F3FC}\u200D\u2640": "woman lifting weights: medium-light skin tone", "\u{1F3CB}\u{1F3FD}\u200D\u2640\uFE0F": "woman lifting weights: medium skin tone", "\u{1F3CB}\u{1F3FD}\u200D\u2640": "woman lifting weights: medium skin tone", "\u{1F3CB}\u{1F3FE}\u200D\u2640\uFE0F": "woman lifting weights: medium-dark skin tone", "\u{1F3CB}\u{1F3FE}\u200D\u2640": "woman lifting weights: medium-dark skin tone", "\u{1F3CB}\u{1F3FF}\u200D\u2640\uFE0F": "woman lifting weights: dark skin tone", "\u{1F3CB}\u{1F3FF}\u200D\u2640": "woman lifting weights: dark skin tone", "\u{1F6B4}": "person biking", "\u{1F6B4}\u{1F3FB}": "person biking: light skin tone", "\u{1F6B4}\u{1F3FC}": "person biking: medium-light skin tone", "\u{1F6B4}\u{1F3FD}": "person biking: medium skin tone", "\u{1F6B4}\u{1F3FE}": "person biking: medium-dark skin tone", "\u{1F6B4}\u{1F3FF}": "person biking: dark skin tone", "\u{1F6B4}\u200D\u2642\uFE0F": "man biking", "\u{1F6B4}\u200D\u2642": "man biking", "\u{1F6B4}\u{1F3FB}\u200D\u2642\uFE0F": "man biking: light skin tone", "\u{1F6B4}\u{1F3FB}\u200D\u2642": "man biking: light skin tone", "\u{1F6B4}\u{1F3FC}\u200D\u2642\uFE0F": "man biking: medium-light skin tone", "\u{1F6B4}\u{1F3FC}\u200D\u2642": "man biking: medium-light skin tone", "\u{1F6B4}\u{1F3FD}\u200D\u2642\uFE0F": "man biking: medium skin tone", "\u{1F6B4}\u{1F3FD}\u200D\u2642": "man biking: medium skin tone", "\u{1F6B4}\u{1F3FE}\u200D\u2642\uFE0F": "man biking: medium-dark skin tone", "\u{1F6B4}\u{1F3FE}\u200D\u2642": "man biking: medium-dark skin tone", "\u{1F6B4}\u{1F3FF}\u200D\u2642\uFE0F": "man biking: dark skin tone", "\u{1F6B4}\u{1F3FF}\u200D\u2642": "man biking: dark skin tone", "\u{1F6B4}\u200D\u2640\uFE0F": "woman biking", "\u{1F6B4}\u200D\u2640": "woman biking", "\u{1F6B4}\u{1F3FB}\u200D\u2640\uFE0F": "woman biking: light skin tone", "\u{1F6B4}\u{1F3FB}\u200D\u2640": "woman biking: light skin tone", "\u{1F6B4}\u{1F3FC}\u200D\u2640\uFE0F": "woman biking: medium-light skin tone", "\u{1F6B4}\u{1F3FC}\u200D\u2640": "woman biking: medium-light skin tone", "\u{1F6B4}\u{1F3FD}\u200D\u2640\uFE0F": "woman biking: medium skin tone", "\u{1F6B4}\u{1F3FD}\u200D\u2640": "woman biking: medium skin tone", "\u{1F6B4}\u{1F3FE}\u200D\u2640\uFE0F": "woman biking: medium-dark skin tone", "\u{1F6B4}\u{1F3FE}\u200D\u2640": "woman biking: medium-dark skin tone", "\u{1F6B4}\u{1F3FF}\u200D\u2640\uFE0F": "woman biking: dark skin tone", "\u{1F6B4}\u{1F3FF}\u200D\u2640": "woman biking: dark skin tone", "\u{1F6B5}": "person mountain biking", "\u{1F6B5}\u{1F3FB}": "person mountain biking: light skin tone", "\u{1F6B5}\u{1F3FC}": "person mountain biking: medium-light skin tone", "\u{1F6B5}\u{1F3FD}": "person mountain biking: medium skin tone", "\u{1F6B5}\u{1F3FE}": "person mountain biking: medium-dark skin tone", "\u{1F6B5}\u{1F3FF}": "person mountain biking: dark skin tone", "\u{1F6B5}\u200D\u2642\uFE0F": "man mountain biking", "\u{1F6B5}\u200D\u2642": "man mountain biking", "\u{1F6B5}\u{1F3FB}\u200D\u2642\uFE0F": "man mountain biking: light skin tone", "\u{1F6B5}\u{1F3FB}\u200D\u2642": "man mountain biking: light skin tone", "\u{1F6B5}\u{1F3FC}\u200D\u2642\uFE0F": "man mountain biking: medium-light skin tone", "\u{1F6B5}\u{1F3FC}\u200D\u2642": "man mountain biking: medium-light skin tone", "\u{1F6B5}\u{1F3FD}\u200D\u2642\uFE0F": "man mountain biking: medium skin tone", "\u{1F6B5}\u{1F3FD}\u200D\u2642": "man mountain biking: medium skin tone", "\u{1F6B5}\u{1F3FE}\u200D\u2642\uFE0F": "man mountain biking: medium-dark skin tone", "\u{1F6B5}\u{1F3FE}\u200D\u2642": "man mountain biking: medium-dark skin tone", "\u{1F6B5}\u{1F3FF}\u200D\u2642\uFE0F": "man mountain biking: dark skin tone", "\u{1F6B5}\u{1F3FF}\u200D\u2642": "man mountain biking: dark skin tone", "\u{1F6B5}\u200D\u2640\uFE0F": "woman mountain biking", "\u{1F6B5}\u200D\u2640": "woman mountain biking", "\u{1F6B5}\u{1F3FB}\u200D\u2640\uFE0F": "woman mountain biking: light skin tone", "\u{1F6B5}\u{1F3FB}\u200D\u2640": "woman mountain biking: light skin tone", "\u{1F6B5}\u{1F3FC}\u200D\u2640\uFE0F": "woman mountain biking: medium-light skin tone", "\u{1F6B5}\u{1F3FC}\u200D\u2640": "woman mountain biking: medium-light skin tone", "\u{1F6B5}\u{1F3FD}\u200D\u2640\uFE0F": "woman mountain biking: medium skin tone", "\u{1F6B5}\u{1F3FD}\u200D\u2640": "woman mountain biking: medium skin tone", "\u{1F6B5}\u{1F3FE}\u200D\u2640\uFE0F": "woman mountain biking: medium-dark skin tone", "\u{1F6B5}\u{1F3FE}\u200D\u2640": "woman mountain biking: medium-dark skin tone", "\u{1F6B5}\u{1F3FF}\u200D\u2640\uFE0F": "woman mountain biking: dark skin tone", "\u{1F6B5}\u{1F3FF}\u200D\u2640": "woman mountain biking: dark skin tone", "\u{1F3CE}\uFE0F": "racing car", "\u{1F3CE}": "racing car", "\u{1F3CD}\uFE0F": "motorcycle", "\u{1F3CD}": "motorcycle", "\u{1F938}": "person cartwheeling", "\u{1F938}\u{1F3FB}": "person cartwheeling: light skin tone", "\u{1F938}\u{1F3FC}": "person cartwheeling: medium-light skin tone", "\u{1F938}\u{1F3FD}": "person cartwheeling: medium skin tone", "\u{1F938}\u{1F3FE}": "person cartwheeling: medium-dark skin tone", "\u{1F938}\u{1F3FF}": "person cartwheeling: dark skin tone", "\u{1F938}\u200D\u2642\uFE0F": "man cartwheeling", "\u{1F938}\u200D\u2642": "man cartwheeling", "\u{1F938}\u{1F3FB}\u200D\u2642\uFE0F": "man cartwheeling: light skin tone", "\u{1F938}\u{1F3FB}\u200D\u2642": "man cartwheeling: light skin tone", "\u{1F938}\u{1F3FC}\u200D\u2642\uFE0F": "man cartwheeling: medium-light skin tone", "\u{1F938}\u{1F3FC}\u200D\u2642": "man cartwheeling: medium-light skin tone", "\u{1F938}\u{1F3FD}\u200D\u2642\uFE0F": "man cartwheeling: medium skin tone", "\u{1F938}\u{1F3FD}\u200D\u2642": "man cartwheeling: medium skin tone", "\u{1F938}\u{1F3FE}\u200D\u2642\uFE0F": "man cartwheeling: medium-dark skin tone", "\u{1F938}\u{1F3FE}\u200D\u2642": "man cartwheeling: medium-dark skin tone", "\u{1F938}\u{1F3FF}\u200D\u2642\uFE0F": "man cartwheeling: dark skin tone", "\u{1F938}\u{1F3FF}\u200D\u2642": "man cartwheeling: dark skin tone", "\u{1F938}\u200D\u2640\uFE0F": "woman cartwheeling", "\u{1F938}\u200D\u2640": "woman cartwheeling", "\u{1F938}\u{1F3FB}\u200D\u2640\uFE0F": "woman cartwheeling: light skin tone", "\u{1F938}\u{1F3FB}\u200D\u2640": "woman cartwheeling: light skin tone", "\u{1F938}\u{1F3FC}\u200D\u2640\uFE0F": "woman cartwheeling: medium-light skin tone", "\u{1F938}\u{1F3FC}\u200D\u2640": "woman cartwheeling: medium-light skin tone", "\u{1F938}\u{1F3FD}\u200D\u2640\uFE0F": "woman cartwheeling: medium skin tone", "\u{1F938}\u{1F3FD}\u200D\u2640": "woman cartwheeling: medium skin tone", "\u{1F938}\u{1F3FE}\u200D\u2640\uFE0F": "woman cartwheeling: medium-dark skin tone", "\u{1F938}\u{1F3FE}\u200D\u2640": "woman cartwheeling: medium-dark skin tone", "\u{1F938}\u{1F3FF}\u200D\u2640\uFE0F": "woman cartwheeling: dark skin tone", "\u{1F938}\u{1F3FF}\u200D\u2640": "woman cartwheeling: dark skin tone", "\u{1F93C}": "people wrestling", "\u{1F93C}\u200D\u2642\uFE0F": "men wrestling", "\u{1F93C}\u200D\u2642": "men wrestling", "\u{1F93C}\u200D\u2640\uFE0F": "women wrestling", "\u{1F93C}\u200D\u2640": "women wrestling", "\u{1F93D}": "person playing water polo", "\u{1F93D}\u{1F3FB}": "person playing water polo: light skin tone", "\u{1F93D}\u{1F3FC}": "person playing water polo: medium-light skin tone", "\u{1F93D}\u{1F3FD}": "person playing water polo: medium skin tone", "\u{1F93D}\u{1F3FE}": "person playing water polo: medium-dark skin tone", "\u{1F93D}\u{1F3FF}": "person playing water polo: dark skin tone", "\u{1F93D}\u200D\u2642\uFE0F": "man playing water polo", "\u{1F93D}\u200D\u2642": "man playing water polo", "\u{1F93D}\u{1F3FB}\u200D\u2642\uFE0F": "man playing water polo: light skin tone", "\u{1F93D}\u{1F3FB}\u200D\u2642": "man playing water polo: light skin tone", "\u{1F93D}\u{1F3FC}\u200D\u2642\uFE0F": "man playing water polo: medium-light skin tone", "\u{1F93D}\u{1F3FC}\u200D\u2642": "man playing water polo: medium-light skin tone", "\u{1F93D}\u{1F3FD}\u200D\u2642\uFE0F": "man playing water polo: medium skin tone", "\u{1F93D}\u{1F3FD}\u200D\u2642": "man playing water polo: medium skin tone", "\u{1F93D}\u{1F3FE}\u200D\u2642\uFE0F": "man playing water polo: medium-dark skin tone", "\u{1F93D}\u{1F3FE}\u200D\u2642": "man playing water polo: medium-dark skin tone", "\u{1F93D}\u{1F3FF}\u200D\u2642\uFE0F": "man playing water polo: dark skin tone", "\u{1F93D}\u{1F3FF}\u200D\u2642": "man playing water polo: dark skin tone", "\u{1F93D}\u200D\u2640\uFE0F": "woman playing water polo", "\u{1F93D}\u200D\u2640": "woman playing water polo", "\u{1F93D}\u{1F3FB}\u200D\u2640\uFE0F": "woman playing water polo: light skin tone", "\u{1F93D}\u{1F3FB}\u200D\u2640": "woman playing water polo: light skin tone", "\u{1F93D}\u{1F3FC}\u200D\u2640\uFE0F": "woman playing water polo: medium-light skin tone", "\u{1F93D}\u{1F3FC}\u200D\u2640": "woman playing water polo: medium-light skin tone", "\u{1F93D}\u{1F3FD}\u200D\u2640\uFE0F": "woman playing water polo: medium skin tone", "\u{1F93D}\u{1F3FD}\u200D\u2640": "woman playing water polo: medium skin tone", "\u{1F93D}\u{1F3FE}\u200D\u2640\uFE0F": "woman playing water polo: medium-dark skin tone", "\u{1F93D}\u{1F3FE}\u200D\u2640": "woman playing water polo: medium-dark skin tone", "\u{1F93D}\u{1F3FF}\u200D\u2640\uFE0F": "woman playing water polo: dark skin tone", "\u{1F93D}\u{1F3FF}\u200D\u2640": "woman playing water polo: dark skin tone", "\u{1F93E}": "person playing handball", "\u{1F93E}\u{1F3FB}": "person playing handball: light skin tone", "\u{1F93E}\u{1F3FC}": "person playing handball: medium-light skin tone", "\u{1F93E}\u{1F3FD}": "person playing handball: medium skin tone", "\u{1F93E}\u{1F3FE}": "person playing handball: medium-dark skin tone", "\u{1F93E}\u{1F3FF}": "person playing handball: dark skin tone", "\u{1F93E}\u200D\u2642\uFE0F": "man playing handball", "\u{1F93E}\u200D\u2642": "man playing handball", "\u{1F93E}\u{1F3FB}\u200D\u2642\uFE0F": "man playing handball: light skin tone", "\u{1F93E}\u{1F3FB}\u200D\u2642": "man playing handball: light skin tone", "\u{1F93E}\u{1F3FC}\u200D\u2642\uFE0F": "man playing handball: medium-light skin tone", "\u{1F93E}\u{1F3FC}\u200D\u2642": "man playing handball: medium-light skin tone", "\u{1F93E}\u{1F3FD}\u200D\u2642\uFE0F": "man playing handball: medium skin tone", "\u{1F93E}\u{1F3FD}\u200D\u2642": "man playing handball: medium skin tone", "\u{1F93E}\u{1F3FE}\u200D\u2642\uFE0F": "man playing handball: medium-dark skin tone", "\u{1F93E}\u{1F3FE}\u200D\u2642": "man playing handball: medium-dark skin tone", "\u{1F93E}\u{1F3FF}\u200D\u2642\uFE0F": "man playing handball: dark skin tone", "\u{1F93E}\u{1F3FF}\u200D\u2642": "man playing handball: dark skin tone", "\u{1F93E}\u200D\u2640\uFE0F": "woman playing handball", "\u{1F93E}\u200D\u2640": "woman playing handball", "\u{1F93E}\u{1F3FB}\u200D\u2640\uFE0F": "woman playing handball: light skin tone", "\u{1F93E}\u{1F3FB}\u200D\u2640": "woman playing handball: light skin tone", "\u{1F93E}\u{1F3FC}\u200D\u2640\uFE0F": "woman playing handball: medium-light skin tone", "\u{1F93E}\u{1F3FC}\u200D\u2640": "woman playing handball: medium-light skin tone", "\u{1F93E}\u{1F3FD}\u200D\u2640\uFE0F": "woman playing handball: medium skin tone", "\u{1F93E}\u{1F3FD}\u200D\u2640": "woman playing handball: medium skin tone", "\u{1F93E}\u{1F3FE}\u200D\u2640\uFE0F": "woman playing handball: medium-dark skin tone", "\u{1F93E}\u{1F3FE}\u200D\u2640": "woman playing handball: medium-dark skin tone", "\u{1F93E}\u{1F3FF}\u200D\u2640\uFE0F": "woman playing handball: dark skin tone", "\u{1F93E}\u{1F3FF}\u200D\u2640": "woman playing handball: dark skin tone", "\u{1F939}": "person juggling", "\u{1F939}\u{1F3FB}": "person juggling: light skin tone", "\u{1F939}\u{1F3FC}": "person juggling: medium-light skin tone", "\u{1F939}\u{1F3FD}": "person juggling: medium skin tone", "\u{1F939}\u{1F3FE}": "person juggling: medium-dark skin tone", "\u{1F939}\u{1F3FF}": "person juggling: dark skin tone", "\u{1F939}\u200D\u2642\uFE0F": "man juggling", "\u{1F939}\u200D\u2642": "man juggling", "\u{1F939}\u{1F3FB}\u200D\u2642\uFE0F": "man juggling: light skin tone", "\u{1F939}\u{1F3FB}\u200D\u2642": "man juggling: light skin tone", "\u{1F939}\u{1F3FC}\u200D\u2642\uFE0F": "man juggling: medium-light skin tone", "\u{1F939}\u{1F3FC}\u200D\u2642": "man juggling: medium-light skin tone", "\u{1F939}\u{1F3FD}\u200D\u2642\uFE0F": "man juggling: medium skin tone", "\u{1F939}\u{1F3FD}\u200D\u2642": "man juggling: medium skin tone", "\u{1F939}\u{1F3FE}\u200D\u2642\uFE0F": "man juggling: medium-dark skin tone", "\u{1F939}\u{1F3FE}\u200D\u2642": "man juggling: medium-dark skin tone", "\u{1F939}\u{1F3FF}\u200D\u2642\uFE0F": "man juggling: dark skin tone", "\u{1F939}\u{1F3FF}\u200D\u2642": "man juggling: dark skin tone", "\u{1F939}\u200D\u2640\uFE0F": "woman juggling", "\u{1F939}\u200D\u2640": "woman juggling", "\u{1F939}\u{1F3FB}\u200D\u2640\uFE0F": "woman juggling: light skin tone", "\u{1F939}\u{1F3FB}\u200D\u2640": "woman juggling: light skin tone", "\u{1F939}\u{1F3FC}\u200D\u2640\uFE0F": "woman juggling: medium-light skin tone", "\u{1F939}\u{1F3FC}\u200D\u2640": "woman juggling: medium-light skin tone", "\u{1F939}\u{1F3FD}\u200D\u2640\uFE0F": "woman juggling: medium skin tone", "\u{1F939}\u{1F3FD}\u200D\u2640": "woman juggling: medium skin tone", "\u{1F939}\u{1F3FE}\u200D\u2640\uFE0F": "woman juggling: medium-dark skin tone", "\u{1F939}\u{1F3FE}\u200D\u2640": "woman juggling: medium-dark skin tone", "\u{1F939}\u{1F3FF}\u200D\u2640\uFE0F": "woman juggling: dark skin tone", "\u{1F939}\u{1F3FF}\u200D\u2640": "woman juggling: dark skin tone", "\u{1F46B}": "man and woman holding hands", "\u{1F46C}": "two men holding hands", "\u{1F46D}": "two women holding hands", "\u{1F48F}": "kiss", "\u{1F469}\u200D\u2764\uFE0F\u200D\u{1F48B}\u200D\u{1F468}": "kiss: woman, man", "\u{1F469}\u200D\u2764\u200D\u{1F48B}\u200D\u{1F468}": "kiss: woman, man", "\u{1F468}\u200D\u2764\uFE0F\u200D\u{1F48B}\u200D\u{1F468}": "kiss: man, man", "\u{1F468}\u200D\u2764\u200D\u{1F48B}\u200D\u{1F468}": "kiss: man, man", "\u{1F469}\u200D\u2764\uFE0F\u200D\u{1F48B}\u200D\u{1F469}": "kiss: woman, woman", "\u{1F469}\u200D\u2764\u200D\u{1F48B}\u200D\u{1F469}": "kiss: woman, woman", "\u{1F491}": "couple with heart", "\u{1F469}\u200D\u2764\uFE0F\u200D\u{1F468}": "couple with heart: woman, man", "\u{1F469}\u200D\u2764\u200D\u{1F468}": "couple with heart: woman, man", "\u{1F468}\u200D\u2764\uFE0F\u200D\u{1F468}": "couple with heart: man, man", "\u{1F468}\u200D\u2764\u200D\u{1F468}": "couple with heart: man, man", "\u{1F469}\u200D\u2764\uFE0F\u200D\u{1F469}": "couple with heart: woman, woman", "\u{1F469}\u200D\u2764\u200D\u{1F469}": "couple with heart: woman, woman", "\u{1F46A}": "family", "\u{1F468}\u200D\u{1F469}\u200D\u{1F466}": "family: man, woman, boy", "\u{1F468}\u200D\u{1F469}\u200D\u{1F467}": "family: man, woman, girl", "\u{1F468}\u200D\u{1F469}\u200D\u{1F467}\u200D\u{1F466}": "family: man, woman, girl, boy", "\u{1F468}\u200D\u{1F469}\u200D\u{1F466}\u200D\u{1F466}": "family: man, woman, boy, boy", "\u{1F468}\u200D\u{1F469}\u200D\u{1F467}\u200D\u{1F467}": "family: man, woman, girl, girl", "\u{1F468}\u200D\u{1F468}\u200D\u{1F466}": "family: man, man, boy", "\u{1F468}\u200D\u{1F468}\u200D\u{1F467}": "family: man, man, girl", "\u{1F468}\u200D\u{1F468}\u200D\u{1F467}\u200D\u{1F466}": "family: man, man, girl, boy", "\u{1F468}\u200D\u{1F468}\u200D\u{1F466}\u200D\u{1F466}": "family: man, man, boy, boy", "\u{1F468}\u200D\u{1F468}\u200D\u{1F467}\u200D\u{1F467}": "family: man, man, girl, girl", "\u{1F469}\u200D\u{1F469}\u200D\u{1F466}": "family: woman, woman, boy", "\u{1F469}\u200D\u{1F469}\u200D\u{1F467}": "family: woman, woman, girl", "\u{1F469}\u200D\u{1F469}\u200D\u{1F467}\u200D\u{1F466}": "family: woman, woman, girl, boy", "\u{1F469}\u200D\u{1F469}\u200D\u{1F466}\u200D\u{1F466}": "family: woman, woman, boy, boy", "\u{1F469}\u200D\u{1F469}\u200D\u{1F467}\u200D\u{1F467}": "family: woman, woman, girl, girl", "\u{1F468}\u200D\u{1F466}": "family: man, boy", "\u{1F468}\u200D\u{1F466}\u200D\u{1F466}": "family: man, boy, boy", "\u{1F468}\u200D\u{1F467}": "family: man, girl", "\u{1F468}\u200D\u{1F467}\u200D\u{1F466}": "family: man, girl, boy", "\u{1F468}\u200D\u{1F467}\u200D\u{1F467}": "family: man, girl, girl", "\u{1F469}\u200D\u{1F466}": "family: woman, boy", "\u{1F469}\u200D\u{1F466}\u200D\u{1F466}": "family: woman, boy, boy", "\u{1F469}\u200D\u{1F467}": "family: woman, girl", "\u{1F469}\u200D\u{1F467}\u200D\u{1F466}": "family: woman, girl, boy", "\u{1F469}\u200D\u{1F467}\u200D\u{1F467}": "family: woman, girl, girl", "\u{1F933}": "selfie", "\u{1F933}\u{1F3FB}": "selfie: light skin tone", "\u{1F933}\u{1F3FC}": "selfie: medium-light skin tone", "\u{1F933}\u{1F3FD}": "selfie: medium skin tone", "\u{1F933}\u{1F3FE}": "selfie: medium-dark skin tone", "\u{1F933}\u{1F3FF}": "selfie: dark skin tone", "\u{1F4AA}": "flexed biceps", "\u{1F4AA}\u{1F3FB}": "flexed biceps: light skin tone", "\u{1F4AA}\u{1F3FC}": "flexed biceps: medium-light skin tone", "\u{1F4AA}\u{1F3FD}": "flexed biceps: medium skin tone", "\u{1F4AA}\u{1F3FE}": "flexed biceps: medium-dark skin tone", "\u{1F4AA}\u{1F3FF}": "flexed biceps: dark skin tone", "\u{1F9B5}": "leg", "\u{1F9B5}\u{1F3FB}": "leg: light skin tone", "\u{1F9B5}\u{1F3FC}": "leg: medium-light skin tone", "\u{1F9B5}\u{1F3FD}": "leg: medium skin tone", "\u{1F9B5}\u{1F3FE}": "leg: medium-dark skin tone", "\u{1F9B5}\u{1F3FF}": "leg: dark skin tone", "\u{1F9B6}": "foot", "\u{1F9B6}\u{1F3FB}": "foot: light skin tone", "\u{1F9B6}\u{1F3FC}": "foot: medium-light skin tone", "\u{1F9B6}\u{1F3FD}": "foot: medium skin tone", "\u{1F9B6}\u{1F3FE}": "foot: medium-dark skin tone", "\u{1F9B6}\u{1F3FF}": "foot: dark skin tone", "\u{1F448}": "backhand index pointing left", "\u{1F448}\u{1F3FB}": "backhand index pointing left: light skin tone", "\u{1F448}\u{1F3FC}": "backhand index pointing left: medium-light skin tone", "\u{1F448}\u{1F3FD}": "backhand index pointing left: medium skin tone", "\u{1F448}\u{1F3FE}": "backhand index pointing left: medium-dark skin tone", "\u{1F448}\u{1F3FF}": "backhand index pointing left: dark skin tone", "\u{1F449}": "backhand index pointing right", "\u{1F449}\u{1F3FB}": "backhand index pointing right: light skin tone", "\u{1F449}\u{1F3FC}": "backhand index pointing right: medium-light skin tone", "\u{1F449}\u{1F3FD}": "backhand index pointing right: medium skin tone", "\u{1F449}\u{1F3FE}": "backhand index pointing right: medium-dark skin tone", "\u{1F449}\u{1F3FF}": "backhand index pointing right: dark skin tone", "\u261D\uFE0F": "index pointing up", "\u261D": "index pointing up", "\u261D\u{1F3FB}": "index pointing up: light skin tone", "\u261D\u{1F3FC}": "index pointing up: medium-light skin tone", "\u261D\u{1F3FD}": "index pointing up: medium skin tone", "\u261D\u{1F3FE}": "index pointing up: medium-dark skin tone", "\u261D\u{1F3FF}": "index pointing up: dark skin tone", "\u{1F446}": "backhand index pointing up", "\u{1F446}\u{1F3FB}": "backhand index pointing up: light skin tone", "\u{1F446}\u{1F3FC}": "backhand index pointing up: medium-light skin tone", "\u{1F446}\u{1F3FD}": "backhand index pointing up: medium skin tone", "\u{1F446}\u{1F3FE}": "backhand index pointing up: medium-dark skin tone", "\u{1F446}\u{1F3FF}": "backhand index pointing up: dark skin tone", "\u{1F595}": "middle finger", "\u{1F595}\u{1F3FB}": "middle finger: light skin tone", "\u{1F595}\u{1F3FC}": "middle finger: medium-light skin tone", "\u{1F595}\u{1F3FD}": "middle finger: medium skin tone", "\u{1F595}\u{1F3FE}": "middle finger: medium-dark skin tone", "\u{1F595}\u{1F3FF}": "middle finger: dark skin tone", "\u{1F447}": "backhand index pointing down", "\u{1F447}\u{1F3FB}": "backhand index pointing down: light skin tone", "\u{1F447}\u{1F3FC}": "backhand index pointing down: medium-light skin tone", "\u{1F447}\u{1F3FD}": "backhand index pointing down: medium skin tone", "\u{1F447}\u{1F3FE}": "backhand index pointing down: medium-dark skin tone", "\u{1F447}\u{1F3FF}": "backhand index pointing down: dark skin tone", "\u270C\uFE0F": "victory hand", "\u270C": "victory hand", "\u270C\u{1F3FB}": "victory hand: light skin tone", "\u270C\u{1F3FC}": "victory hand: medium-light skin tone", "\u270C\u{1F3FD}": "victory hand: medium skin tone", "\u270C\u{1F3FE}": "victory hand: medium-dark skin tone", "\u270C\u{1F3FF}": "victory hand: dark skin tone", "\u{1F91E}": "crossed fingers", "\u{1F91E}\u{1F3FB}": "crossed fingers: light skin tone", "\u{1F91E}\u{1F3FC}": "crossed fingers: medium-light skin tone", "\u{1F91E}\u{1F3FD}": "crossed fingers: medium skin tone", "\u{1F91E}\u{1F3FE}": "crossed fingers: medium-dark skin tone", "\u{1F91E}\u{1F3FF}": "crossed fingers: dark skin tone", "\u{1F596}": "vulcan salute", "\u{1F596}\u{1F3FB}": "vulcan salute: light skin tone", "\u{1F596}\u{1F3FC}": "vulcan salute: medium-light skin tone", "\u{1F596}\u{1F3FD}": "vulcan salute: medium skin tone", "\u{1F596}\u{1F3FE}": "vulcan salute: medium-dark skin tone", "\u{1F596}\u{1F3FF}": "vulcan salute: dark skin tone", "\u{1F918}": "sign of the horns", "\u{1F918}\u{1F3FB}": "sign of the horns: light skin tone", "\u{1F918}\u{1F3FC}": "sign of the horns: medium-light skin tone", "\u{1F918}\u{1F3FD}": "sign of the horns: medium skin tone", "\u{1F918}\u{1F3FE}": "sign of the horns: medium-dark skin tone", "\u{1F918}\u{1F3FF}": "sign of the horns: dark skin tone", "\u{1F919}": "call me hand", "\u{1F919}\u{1F3FB}": "call me hand: light skin tone", "\u{1F919}\u{1F3FC}": "call me hand: medium-light skin tone", "\u{1F919}\u{1F3FD}": "call me hand: medium skin tone", "\u{1F919}\u{1F3FE}": "call me hand: medium-dark skin tone", "\u{1F919}\u{1F3FF}": "call me hand: dark skin tone", "\u{1F590}\uFE0F": "hand with fingers splayed", "\u{1F590}": "hand with fingers splayed", "\u{1F590}\u{1F3FB}": "hand with fingers splayed: light skin tone", "\u{1F590}\u{1F3FC}": "hand with fingers splayed: medium-light skin tone", "\u{1F590}\u{1F3FD}": "hand with fingers splayed: medium skin tone", "\u{1F590}\u{1F3FE}": "hand with fingers splayed: medium-dark skin tone", "\u{1F590}\u{1F3FF}": "hand with fingers splayed: dark skin tone", "\u270B": "raised hand", "\u270B\u{1F3FB}": "raised hand: light skin tone", "\u270B\u{1F3FC}": "raised hand: medium-light skin tone", "\u270B\u{1F3FD}": "raised hand: medium skin tone", "\u270B\u{1F3FE}": "raised hand: medium-dark skin tone", "\u270B\u{1F3FF}": "raised hand: dark skin tone", "\u{1F44C}": "OK hand", "\u{1F44C}\u{1F3FB}": "OK hand: light skin tone", "\u{1F44C}\u{1F3FC}": "OK hand: medium-light skin tone", "\u{1F44C}\u{1F3FD}": "OK hand: medium skin tone", "\u{1F44C}\u{1F3FE}": "OK hand: medium-dark skin tone", "\u{1F44C}\u{1F3FF}": "OK hand: dark skin tone", "\u{1F44D}": "thumbs up", "\u{1F44D}\u{1F3FB}": "thumbs up: light skin tone", "\u{1F44D}\u{1F3FC}": "thumbs up: medium-light skin tone", "\u{1F44D}\u{1F3FD}": "thumbs up: medium skin tone", "\u{1F44D}\u{1F3FE}": "thumbs up: medium-dark skin tone", "\u{1F44D}\u{1F3FF}": "thumbs up: dark skin tone", "\u{1F44E}": "thumbs down", "\u{1F44E}\u{1F3FB}": "thumbs down: light skin tone", "\u{1F44E}\u{1F3FC}": "thumbs down: medium-light skin tone", "\u{1F44E}\u{1F3FD}": "thumbs down: medium skin tone", "\u{1F44E}\u{1F3FE}": "thumbs down: medium-dark skin tone", "\u{1F44E}\u{1F3FF}": "thumbs down: dark skin tone", "\u270A": "raised fist", "\u270A\u{1F3FB}": "raised fist: light skin tone", "\u270A\u{1F3FC}": "raised fist: medium-light skin tone", "\u270A\u{1F3FD}": "raised fist: medium skin tone", "\u270A\u{1F3FE}": "raised fist: medium-dark skin tone", "\u270A\u{1F3FF}": "raised fist: dark skin tone", "\u{1F44A}": "oncoming fist", "\u{1F44A}\u{1F3FB}": "oncoming fist: light skin tone", "\u{1F44A}\u{1F3FC}": "oncoming fist: medium-light skin tone", "\u{1F44A}\u{1F3FD}": "oncoming fist: medium skin tone", "\u{1F44A}\u{1F3FE}": "oncoming fist: medium-dark skin tone", "\u{1F44A}\u{1F3FF}": "oncoming fist: dark skin tone", "\u{1F91B}": "left-facing fist", "\u{1F91B}\u{1F3FB}": "left-facing fist: light skin tone", "\u{1F91B}\u{1F3FC}": "left-facing fist: medium-light skin tone", "\u{1F91B}\u{1F3FD}": "left-facing fist: medium skin tone", "\u{1F91B}\u{1F3FE}": "left-facing fist: medium-dark skin tone", "\u{1F91B}\u{1F3FF}": "left-facing fist: dark skin tone", "\u{1F91C}": "right-facing fist", "\u{1F91C}\u{1F3FB}": "right-facing fist: light skin tone", "\u{1F91C}\u{1F3FC}": "right-facing fist: medium-light skin tone", "\u{1F91C}\u{1F3FD}": "right-facing fist: medium skin tone", "\u{1F91C}\u{1F3FE}": "right-facing fist: medium-dark skin tone", "\u{1F91C}\u{1F3FF}": "right-facing fist: dark skin tone", "\u{1F91A}": "raised back of hand", "\u{1F91A}\u{1F3FB}": "raised back of hand: light skin tone", "\u{1F91A}\u{1F3FC}": "raised back of hand: medium-light skin tone", "\u{1F91A}\u{1F3FD}": "raised back of hand: medium skin tone", "\u{1F91A}\u{1F3FE}": "raised back of hand: medium-dark skin tone", "\u{1F91A}\u{1F3FF}": "raised back of hand: dark skin tone", "\u{1F44B}": "waving hand", "\u{1F44B}\u{1F3FB}": "waving hand: light skin tone", "\u{1F44B}\u{1F3FC}": "waving hand: medium-light skin tone", "\u{1F44B}\u{1F3FD}": "waving hand: medium skin tone", "\u{1F44B}\u{1F3FE}": "waving hand: medium-dark skin tone", "\u{1F44B}\u{1F3FF}": "waving hand: dark skin tone", "\u{1F91F}": "love-you gesture", "\u{1F91F}\u{1F3FB}": "love-you gesture: light skin tone", "\u{1F91F}\u{1F3FC}": "love-you gesture: medium-light skin tone", "\u{1F91F}\u{1F3FD}": "love-you gesture: medium skin tone", "\u{1F91F}\u{1F3FE}": "love-you gesture: medium-dark skin tone", "\u{1F91F}\u{1F3FF}": "love-you gesture: dark skin tone", "\u270D\uFE0F": "writing hand", "\u270D": "writing hand", "\u270D\u{1F3FB}": "writing hand: light skin tone", "\u270D\u{1F3FC}": "writing hand: medium-light skin tone", "\u270D\u{1F3FD}": "writing hand: medium skin tone", "\u270D\u{1F3FE}": "writing hand: medium-dark skin tone", "\u270D\u{1F3FF}": "writing hand: dark skin tone", "\u{1F44F}": "clapping hands", "\u{1F44F}\u{1F3FB}": "clapping hands: light skin tone", "\u{1F44F}\u{1F3FC}": "clapping hands: medium-light skin tone", "\u{1F44F}\u{1F3FD}": "clapping hands: medium skin tone", "\u{1F44F}\u{1F3FE}": "clapping hands: medium-dark skin tone", "\u{1F44F}\u{1F3FF}": "clapping hands: dark skin tone", "\u{1F450}": "open hands", "\u{1F450}\u{1F3FB}": "open hands: light skin tone", "\u{1F450}\u{1F3FC}": "open hands: medium-light skin tone", "\u{1F450}\u{1F3FD}": "open hands: medium skin tone", "\u{1F450}\u{1F3FE}": "open hands: medium-dark skin tone", "\u{1F450}\u{1F3FF}": "open hands: dark skin tone", "\u{1F64C}": "raising hands", "\u{1F64C}\u{1F3FB}": "raising hands: light skin tone", "\u{1F64C}\u{1F3FC}": "raising hands: medium-light skin tone", "\u{1F64C}\u{1F3FD}": "raising hands: medium skin tone", "\u{1F64C}\u{1F3FE}": "raising hands: medium-dark skin tone", "\u{1F64C}\u{1F3FF}": "raising hands: dark skin tone", "\u{1F932}": "palms up together", "\u{1F932}\u{1F3FB}": "palms up together: light skin tone", "\u{1F932}\u{1F3FC}": "palms up together: medium-light skin tone", "\u{1F932}\u{1F3FD}": "palms up together: medium skin tone", "\u{1F932}\u{1F3FE}": "palms up together: medium-dark skin tone", "\u{1F932}\u{1F3FF}": "palms up together: dark skin tone", "\u{1F64F}": "folded hands", "\u{1F64F}\u{1F3FB}": "folded hands: light skin tone", "\u{1F64F}\u{1F3FC}": "folded hands: medium-light skin tone", "\u{1F64F}\u{1F3FD}": "folded hands: medium skin tone", "\u{1F64F}\u{1F3FE}": "folded hands: medium-dark skin tone", "\u{1F64F}\u{1F3FF}": "folded hands: dark skin tone", "\u{1F91D}": "handshake", "\u{1F485}": "nail polish", "\u{1F485}\u{1F3FB}": "nail polish: light skin tone", "\u{1F485}\u{1F3FC}": "nail polish: medium-light skin tone", "\u{1F485}\u{1F3FD}": "nail polish: medium skin tone", "\u{1F485}\u{1F3FE}": "nail polish: medium-dark skin tone", "\u{1F485}\u{1F3FF}": "nail polish: dark skin tone", "\u{1F442}": "ear", "\u{1F442}\u{1F3FB}": "ear: light skin tone", "\u{1F442}\u{1F3FC}": "ear: medium-light skin tone", "\u{1F442}\u{1F3FD}": "ear: medium skin tone", "\u{1F442}\u{1F3FE}": "ear: medium-dark skin tone", "\u{1F442}\u{1F3FF}": "ear: dark skin tone", "\u{1F443}": "nose", "\u{1F443}\u{1F3FB}": "nose: light skin tone", "\u{1F443}\u{1F3FC}": "nose: medium-light skin tone", "\u{1F443}\u{1F3FD}": "nose: medium skin tone", "\u{1F443}\u{1F3FE}": "nose: medium-dark skin tone", "\u{1F443}\u{1F3FF}": "nose: dark skin tone", "\u{1F9B0}": "red-haired", "\u{1F9B1}": "curly-haired", "\u{1F9B2}": "bald", "\u{1F9B3}": "white-haired", "\u{1F463}": "footprints", "\u{1F440}": "eyes", "\u{1F441}\uFE0F": "eye", "\u{1F441}": "eye", "\u{1F441}\uFE0F\u200D\u{1F5E8}\uFE0F": "eye in speech bubble", "\u{1F441}\u200D\u{1F5E8}\uFE0F": "eye in speech bubble", "\u{1F441}\uFE0F\u200D\u{1F5E8}": "eye in speech bubble", "\u{1F441}\u200D\u{1F5E8}": "eye in speech bubble", "\u{1F9E0}": "brain", "\u{1F9B4}": "bone", "\u{1F9B7}": "tooth", "\u{1F445}": "tongue", "\u{1F444}": "mouth", "\u{1F48B}": "kiss mark", "\u{1F498}": "heart with arrow", "\u2764\uFE0F": "red heart", "\u2764": "red heart", "\u{1F493}": "beating heart", "\u{1F494}": "broken heart", "\u{1F495}": "two hearts", "\u{1F496}": "sparkling heart", "\u{1F497}": "growing heart", "\u{1F499}": "blue heart", "\u{1F49A}": "green heart", "\u{1F49B}": "yellow heart", "\u{1F9E1}": "orange heart", "\u{1F49C}": "purple heart", "\u{1F5A4}": "black heart", "\u{1F49D}": "heart with ribbon", "\u{1F49E}": "revolving hearts", "\u{1F49F}": "heart decoration", "\u2763\uFE0F": "heavy heart exclamation", "\u2763": "heavy heart exclamation", "\u{1F48C}": "love letter", "\u{1F4A4}": "zzz", "\u{1F4A2}": "anger symbol", "\u{1F4A3}": "bomb", "\u{1F4A5}": "collision", "\u{1F4A6}": "sweat droplets", "\u{1F4A8}": "dashing away", "\u{1F4AB}": "dizzy", "\u{1F4AC}": "speech balloon", "\u{1F5E8}\uFE0F": "left speech bubble", "\u{1F5E8}": "left speech bubble", "\u{1F5EF}\uFE0F": "right anger bubble", "\u{1F5EF}": "right anger bubble", "\u{1F4AD}": "thought balloon", "\u{1F573}\uFE0F": "hole", "\u{1F573}": "hole", "\u{1F453}": "glasses", "\u{1F576}\uFE0F": "sunglasses", "\u{1F576}": "sunglasses", "\u{1F97D}": "goggles", "\u{1F97C}": "lab coat", "\u{1F454}": "necktie", "\u{1F455}": "t-shirt", "\u{1F456}": "jeans", "\u{1F9E3}": "scarf", "\u{1F9E4}": "gloves", "\u{1F9E5}": "coat", "\u{1F9E6}": "socks", "\u{1F457}": "dress", "\u{1F458}": "kimono", "\u{1F459}": "bikini", "\u{1F45A}": "woman\u2019s clothes", "\u{1F45B}": "purse", "\u{1F45C}": "handbag", "\u{1F45D}": "clutch bag", "\u{1F6CD}\uFE0F": "shopping bags", "\u{1F6CD}": "shopping bags", "\u{1F392}": "school backpack", "\u{1F45E}": "man\u2019s shoe", "\u{1F45F}": "running shoe", "\u{1F97E}": "hiking boot", "\u{1F97F}": "woman\u2019s flat shoe", "\u{1F460}": "high-heeled shoe", "\u{1F461}": "woman\u2019s sandal", "\u{1F462}": "woman\u2019s boot", "\u{1F451}": "crown", "\u{1F452}": "woman\u2019s hat", "\u{1F3A9}": "top hat", "\u{1F393}": "graduation cap", "\u{1F9E2}": "billed cap", "\u26D1\uFE0F": "rescue worker\u2019s helmet", "\u26D1": "rescue worker\u2019s helmet", "\u{1F4FF}": "prayer beads", "\u{1F484}": "lipstick", "\u{1F48D}": "ring", "\u{1F48E}": "gem stone", "\u{1F435}": "monkey face", "\u{1F412}": "monkey", "\u{1F98D}": "gorilla", "\u{1F436}": "dog face", "\u{1F415}": "dog", "\u{1F429}": "poodle", "\u{1F43A}": "wolf face", "\u{1F98A}": "fox face", "\u{1F99D}": "raccoon", "\u{1F431}": "cat face", "\u{1F408}": "cat", "\u{1F981}": "lion face", "\u{1F42F}": "tiger face", "\u{1F405}": "tiger", "\u{1F406}": "leopard", "\u{1F434}": "horse face", "\u{1F40E}": "horse", "\u{1F984}": "unicorn face", "\u{1F993}": "zebra", "\u{1F98C}": "deer", "\u{1F42E}": "cow face", "\u{1F402}": "ox", "\u{1F403}": "water buffalo", "\u{1F404}": "cow", "\u{1F437}": "pig face", "\u{1F416}": "pig", "\u{1F417}": "boar", "\u{1F43D}": "pig nose", "\u{1F40F}": "ram", "\u{1F411}": "ewe", "\u{1F410}": "goat", "\u{1F42A}": "camel", "\u{1F42B}": "two-hump camel", "\u{1F999}": "llama", "\u{1F992}": "giraffe", "\u{1F418}": "elephant", "\u{1F98F}": "rhinoceros", "\u{1F99B}": "hippopotamus", "\u{1F42D}": "mouse face", "\u{1F401}": "mouse", "\u{1F400}": "rat", "\u{1F439}": "hamster face", "\u{1F430}": "rabbit face", "\u{1F407}": "rabbit", "\u{1F43F}\uFE0F": "chipmunk", "\u{1F43F}": "chipmunk", "\u{1F994}": "hedgehog", "\u{1F987}": "bat", "\u{1F43B}": "bear face", "\u{1F428}": "koala", "\u{1F43C}": "panda face", "\u{1F998}": "kangaroo", "\u{1F9A1}": "badger", "\u{1F43E}": "paw prints", "\u{1F983}": "turkey", "\u{1F414}": "chicken", "\u{1F413}": "rooster", "\u{1F423}": "hatching chick", "\u{1F424}": "baby chick", "\u{1F425}": "front-facing baby chick", "\u{1F426}": "bird", "\u{1F427}": "penguin", "\u{1F54A}\uFE0F": "dove", "\u{1F54A}": "dove", "\u{1F985}": "eagle", "\u{1F986}": "duck", "\u{1F9A2}": "swan", "\u{1F989}": "owl", "\u{1F99A}": "peacock", "\u{1F99C}": "parrot", "\u{1F438}": "frog face", "\u{1F40A}": "crocodile", "\u{1F422}": "turtle", "\u{1F98E}": "lizard", "\u{1F40D}": "snake", "\u{1F432}": "dragon face", "\u{1F409}": "dragon", "\u{1F995}": "sauropod", "\u{1F996}": "T-Rex", "\u{1F433}": "spouting whale", "\u{1F40B}": "whale", "\u{1F42C}": "dolphin", "\u{1F41F}": "fish", "\u{1F420}": "tropical fish", "\u{1F421}": "blowfish", "\u{1F988}": "shark", "\u{1F419}": "octopus", "\u{1F41A}": "spiral shell", "\u{1F980}": "crab", "\u{1F99E}": "lobster", "\u{1F990}": "shrimp", "\u{1F991}": "squid", "\u{1F40C}": "snail", "\u{1F98B}": "butterfly", "\u{1F41B}": "bug", "\u{1F41C}": "ant", "\u{1F41D}": "honeybee", "\u{1F41E}": "lady beetle", "\u{1F997}": "cricket", "\u{1F577}\uFE0F": "spider", "\u{1F577}": "spider", "\u{1F578}\uFE0F": "spider web", "\u{1F578}": "spider web", "\u{1F982}": "scorpion", "\u{1F99F}": "mosquito", "\u{1F9A0}": "microbe", "\u{1F490}": "bouquet", "\u{1F338}": "cherry blossom", "\u{1F4AE}": "white flower", "\u{1F3F5}\uFE0F": "rosette", "\u{1F3F5}": "rosette", "\u{1F339}": "rose", "\u{1F940}": "wilted flower", "\u{1F33A}": "hibiscus", "\u{1F33B}": "sunflower", "\u{1F33C}": "blossom", "\u{1F337}": "tulip", "\u{1F331}": "seedling", "\u{1F332}": "evergreen tree", "\u{1F333}": "deciduous tree", "\u{1F334}": "palm tree", "\u{1F335}": "cactus", "\u{1F33E}": "sheaf of rice", "\u{1F33F}": "herb", "\u2618\uFE0F": "shamrock", "\u2618": "shamrock", "\u{1F340}": "four leaf clover", "\u{1F341}": "maple leaf", "\u{1F342}": "fallen leaf", "\u{1F343}": "leaf fluttering in wind", "\u{1F347}": "grapes", "\u{1F348}": "melon", "\u{1F349}": "watermelon", "\u{1F34A}": "tangerine", "\u{1F34B}": "lemon", "\u{1F34C}": "banana", "\u{1F34D}": "pineapple", "\u{1F96D}": "mango", "\u{1F34E}": "red apple", "\u{1F34F}": "green apple", "\u{1F350}": "pear", "\u{1F351}": "peach", "\u{1F352}": "cherries", "\u{1F353}": "strawberry", "\u{1F95D}": "kiwi fruit", "\u{1F345}": "tomato", "\u{1F965}": "coconut", "\u{1F951}": "avocado", "\u{1F346}": "eggplant", "\u{1F954}": "potato", "\u{1F955}": "carrot", "\u{1F33D}": "ear of corn", "\u{1F336}\uFE0F": "hot pepper", "\u{1F336}": "hot pepper", "\u{1F952}": "cucumber", "\u{1F96C}": "leafy green", "\u{1F966}": "broccoli", "\u{1F344}": "mushroom", "\u{1F95C}": "peanuts", "\u{1F330}": "chestnut", "\u{1F35E}": "bread", "\u{1F950}": "croissant", "\u{1F956}": "baguette bread", "\u{1F968}": "pretzel", "\u{1F96F}": "bagel", "\u{1F95E}": "pancakes", "\u{1F9C0}": "cheese wedge", "\u{1F356}": "meat on bone", "\u{1F357}": "poultry leg", "\u{1F969}": "cut of meat", "\u{1F953}": "bacon", "\u{1F354}": "hamburger", "\u{1F35F}": "french fries", "\u{1F355}": "pizza", "\u{1F32D}": "hot dog", "\u{1F96A}": "sandwich", "\u{1F32E}": "taco", "\u{1F32F}": "burrito", "\u{1F959}": "stuffed flatbread", "\u{1F95A}": "egg", "\u{1F373}": "cooking", "\u{1F958}": "shallow pan of food", "\u{1F372}": "pot of food", "\u{1F963}": "bowl with spoon", "\u{1F957}": "green salad", "\u{1F37F}": "popcorn", "\u{1F9C2}": "salt", "\u{1F96B}": "canned food", "\u{1F371}": "bento box", "\u{1F358}": "rice cracker", "\u{1F359}": "rice ball", "\u{1F35A}": "cooked rice", "\u{1F35B}": "curry rice", "\u{1F35C}": "steaming bowl", "\u{1F35D}": "spaghetti", "\u{1F360}": "roasted sweet potato", "\u{1F362}": "oden", "\u{1F363}": "sushi", "\u{1F364}": "fried shrimp", "\u{1F365}": "fish cake with swirl", "\u{1F96E}": "moon cake", "\u{1F361}": "dango", "\u{1F95F}": "dumpling", "\u{1F960}": "fortune cookie", "\u{1F961}": "takeout box", "\u{1F366}": "soft ice cream", "\u{1F367}": "shaved ice", "\u{1F368}": "ice cream", "\u{1F369}": "doughnut", "\u{1F36A}": "cookie", "\u{1F382}": "birthday cake", "\u{1F370}": "shortcake", "\u{1F9C1}": "cupcake", "\u{1F967}": "pie", "\u{1F36B}": "chocolate bar", "\u{1F36C}": "candy", "\u{1F36D}": "lollipop", "\u{1F36E}": "custard", "\u{1F36F}": "honey pot", "\u{1F37C}": "baby bottle", "\u{1F95B}": "glass of milk", "\u2615": "hot beverage", "\u{1F375}": "teacup without handle", "\u{1F376}": "sake", "\u{1F37E}": "bottle with popping cork", "\u{1F377}": "wine glass", "\u{1F378}": "cocktail glass", "\u{1F379}": "tropical drink", "\u{1F37A}": "beer mug", "\u{1F37B}": "clinking beer mugs", "\u{1F942}": "clinking glasses", "\u{1F943}": "tumbler glass", "\u{1F964}": "cup with straw", "\u{1F962}": "chopsticks", "\u{1F37D}\uFE0F": "fork and knife with plate", "\u{1F37D}": "fork and knife with plate", "\u{1F374}": "fork and knife", "\u{1F944}": "spoon", "\u{1F52A}": "kitchen knife", "\u{1F3FA}": "amphora", "\u{1F30D}": "globe showing Europe-Africa", "\u{1F30E}": "globe showing Americas", "\u{1F30F}": "globe showing Asia-Australia", "\u{1F310}": "globe with meridians", "\u{1F5FA}\uFE0F": "world map", "\u{1F5FA}": "world map", "\u{1F5FE}": "map of Japan", "\u{1F9ED}": "compass", "\u{1F3D4}\uFE0F": "snow-capped mountain", "\u{1F3D4}": "snow-capped mountain", "\u26F0\uFE0F": "mountain", "\u26F0": "mountain", "\u{1F30B}": "volcano", "\u{1F5FB}": "mount fuji", "\u{1F3D5}\uFE0F": "camping", "\u{1F3D5}": "camping", "\u{1F3D6}\uFE0F": "beach with umbrella", "\u{1F3D6}": "beach with umbrella", "\u{1F3DC}\uFE0F": "desert", "\u{1F3DC}": "desert", "\u{1F3DD}\uFE0F": "desert island", "\u{1F3DD}": "desert island", "\u{1F3DE}\uFE0F": "national park", "\u{1F3DE}": "national park", "\u{1F3DF}\uFE0F": "stadium", "\u{1F3DF}": "stadium", "\u{1F3DB}\uFE0F": "classical building", "\u{1F3DB}": "classical building", "\u{1F3D7}\uFE0F": "building construction", "\u{1F3D7}": "building construction", "\u{1F9F1}": "bricks", "\u{1F3D8}\uFE0F": "houses", "\u{1F3D8}": "houses", "\u{1F3DA}\uFE0F": "derelict house", "\u{1F3DA}": "derelict house", "\u{1F3E0}": "house", "\u{1F3E1}": "house with garden", "\u{1F3E2}": "office building", "\u{1F3E3}": "Japanese post office", "\u{1F3E4}": "post office", "\u{1F3E5}": "hospital", "\u{1F3E6}": "bank", "\u{1F3E8}": "hotel", "\u{1F3E9}": "love hotel", "\u{1F3EA}": "convenience store", "\u{1F3EB}": "school", "\u{1F3EC}": "department store", "\u{1F3ED}": "factory", "\u{1F3EF}": "Japanese castle", "\u{1F3F0}": "castle", "\u{1F492}": "wedding", "\u{1F5FC}": "Tokyo tower", "\u{1F5FD}": "Statue of Liberty", "\u26EA": "church", "\u{1F54C}": "mosque", "\u{1F54D}": "synagogue", "\u26E9\uFE0F": "shinto shrine", "\u26E9": "shinto shrine", "\u{1F54B}": "kaaba", "\u26F2": "fountain", "\u26FA": "tent", "\u{1F301}": "foggy", "\u{1F303}": "night with stars", "\u{1F3D9}\uFE0F": "cityscape", "\u{1F3D9}": "cityscape", "\u{1F304}": "sunrise over mountains", "\u{1F305}": "sunrise", "\u{1F306}": "cityscape at dusk", "\u{1F307}": "sunset", "\u{1F309}": "bridge at night", "\u2668\uFE0F": "hot springs", "\u2668": "hot springs", "\u{1F30C}": "milky way", "\u{1F3A0}": "carousel horse", "\u{1F3A1}": "ferris wheel", "\u{1F3A2}": "roller coaster", "\u{1F488}": "barber pole", "\u{1F3AA}": "circus tent", "\u{1F682}": "locomotive", "\u{1F683}": "railway car", "\u{1F684}": "high-speed train", "\u{1F685}": "bullet train", "\u{1F686}": "train", "\u{1F687}": "metro", "\u{1F688}": "light rail", "\u{1F689}": "station", "\u{1F68A}": "tram", "\u{1F69D}": "monorail", "\u{1F69E}": "mountain railway", "\u{1F68B}": "tram car", "\u{1F68C}": "bus", "\u{1F68D}": "oncoming bus", "\u{1F68E}": "trolleybus", "\u{1F690}": "minibus", "\u{1F691}": "ambulance", "\u{1F692}": "fire engine", "\u{1F693}": "police car", "\u{1F694}": "oncoming police car", "\u{1F695}": "taxi", "\u{1F696}": "oncoming taxi", "\u{1F697}": "automobile", "\u{1F698}": "oncoming automobile", "\u{1F699}": "sport utility vehicle", "\u{1F69A}": "delivery truck", "\u{1F69B}": "articulated lorry", "\u{1F69C}": "tractor", "\u{1F6B2}": "bicycle", "\u{1F6F4}": "kick scooter", "\u{1F6F9}": "skateboard", "\u{1F6F5}": "motor scooter", "\u{1F68F}": "bus stop", "\u{1F6E3}\uFE0F": "motorway", "\u{1F6E3}": "motorway", "\u{1F6E4}\uFE0F": "railway track", "\u{1F6E4}": "railway track", "\u{1F6E2}\uFE0F": "oil drum", "\u{1F6E2}": "oil drum", "\u26FD": "fuel pump", "\u{1F6A8}": "police car light", "\u{1F6A5}": "horizontal traffic light", "\u{1F6A6}": "vertical traffic light", "\u{1F6D1}": "stop sign", "\u{1F6A7}": "construction", "\u2693": "anchor", "\u26F5": "sailboat", "\u{1F6F6}": "canoe", "\u{1F6A4}": "speedboat", "\u{1F6F3}\uFE0F": "passenger ship", "\u{1F6F3}": "passenger ship", "\u26F4\uFE0F": "ferry", "\u26F4": "ferry", "\u{1F6E5}\uFE0F": "motor boat", "\u{1F6E5}": "motor boat", "\u{1F6A2}": "ship", "\u2708\uFE0F": "airplane", "\u2708": "airplane", "\u{1F6E9}\uFE0F": "small airplane", "\u{1F6E9}": "small airplane", "\u{1F6EB}": "airplane departure", "\u{1F6EC}": "airplane arrival", "\u{1F4BA}": "seat", "\u{1F681}": "helicopter", "\u{1F69F}": "suspension railway", "\u{1F6A0}": "mountain cableway", "\u{1F6A1}": "aerial tramway", "\u{1F6F0}\uFE0F": "satellite", "\u{1F6F0}": "satellite", "\u{1F680}": "rocket", "\u{1F6F8}": "flying saucer", "\u{1F6CE}\uFE0F": "bellhop bell", "\u{1F6CE}": "bellhop bell", "\u{1F9F3}": "luggage", "\u231B": "hourglass done", "\u23F3": "hourglass not done", "\u231A": "watch", "\u23F0": "alarm clock", "\u23F1\uFE0F": "stopwatch", "\u23F1": "stopwatch", "\u23F2\uFE0F": "timer clock", "\u23F2": "timer clock", "\u{1F570}\uFE0F": "mantelpiece clock", "\u{1F570}": "mantelpiece clock", "\u{1F55B}": "twelve o\u2019clock", "\u{1F567}": "twelve-thirty", "\u{1F550}": "one o\u2019clock", "\u{1F55C}": "one-thirty", "\u{1F551}": "two o\u2019clock", "\u{1F55D}": "two-thirty", "\u{1F552}": "three o\u2019clock", "\u{1F55E}": "three-thirty", "\u{1F553}": "four o\u2019clock", "\u{1F55F}": "four-thirty", "\u{1F554}": "five o\u2019clock", "\u{1F560}": "five-thirty", "\u{1F555}": "six o\u2019clock", "\u{1F561}": "six-thirty", "\u{1F556}": "seven o\u2019clock", "\u{1F562}": "seven-thirty", "\u{1F557}": "eight o\u2019clock", "\u{1F563}": "eight-thirty", "\u{1F558}": "nine o\u2019clock", "\u{1F564}": "nine-thirty", "\u{1F559}": "ten o\u2019clock", "\u{1F565}": "ten-thirty", "\u{1F55A}": "eleven o\u2019clock", "\u{1F566}": "eleven-thirty", "\u{1F311}": "new moon", "\u{1F312}": "waxing crescent moon", "\u{1F313}": "first quarter moon", "\u{1F314}": "waxing gibbous moon", "\u{1F315}": "full moon", "\u{1F316}": "waning gibbous moon", "\u{1F317}": "last quarter moon", "\u{1F318}": "waning crescent moon", "\u{1F319}": "crescent moon", "\u{1F31A}": "new moon face", "\u{1F31B}": "first quarter moon face", "\u{1F31C}": "last quarter moon face", "\u{1F321}\uFE0F": "thermometer", "\u{1F321}": "thermometer", "\u2600\uFE0F": "sun", "\u2600": "sun", "\u{1F31D}": "full moon face", "\u{1F31E}": "sun with face", "\u2B50": "star", "\u{1F31F}": "glowing star", "\u{1F320}": "shooting star", "\u2601\uFE0F": "cloud", "\u2601": "cloud", "\u26C5": "sun behind cloud", "\u26C8\uFE0F": "cloud with lightning and rain", "\u26C8": "cloud with lightning and rain", "\u{1F324}\uFE0F": "sun behind small cloud", "\u{1F324}": "sun behind small cloud", "\u{1F325}\uFE0F": "sun behind large cloud", "\u{1F325}": "sun behind large cloud", "\u{1F326}\uFE0F": "sun behind rain cloud", "\u{1F326}": "sun behind rain cloud", "\u{1F327}\uFE0F": "cloud with rain", "\u{1F327}": "cloud with rain", "\u{1F328}\uFE0F": "cloud with snow", "\u{1F328}": "cloud with snow", "\u{1F329}\uFE0F": "cloud with lightning", "\u{1F329}": "cloud with lightning", "\u{1F32A}\uFE0F": "tornado", "\u{1F32A}": "tornado", "\u{1F32B}\uFE0F": "fog", "\u{1F32B}": "fog", "\u{1F32C}\uFE0F": "wind face", "\u{1F32C}": "wind face", "\u{1F300}": "cyclone", "\u{1F308}": "rainbow", "\u{1F302}": "closed umbrella", "\u2602\uFE0F": "umbrella", "\u2602": "umbrella", "\u2614": "umbrella with rain drops", "\u26F1\uFE0F": "umbrella on ground", "\u26F1": "umbrella on ground", "\u26A1": "high voltage", "\u2744\uFE0F": "snowflake", "\u2744": "snowflake", "\u2603\uFE0F": "snowman", "\u2603": "snowman", "\u26C4": "snowman without snow", "\u2604\uFE0F": "comet", "\u2604": "comet", "\u{1F525}": "fire", "\u{1F4A7}": "droplet", "\u{1F30A}": "water wave", "\u{1F383}": "jack-o-lantern", "\u{1F384}": "Christmas tree", "\u{1F386}": "fireworks", "\u{1F387}": "sparkler", "\u{1F9E8}": "firecracker", "\u2728": "sparkles", "\u{1F388}": "balloon", "\u{1F389}": "party popper", "\u{1F38A}": "confetti ball", "\u{1F38B}": "tanabata tree", "\u{1F38D}": "pine decoration", "\u{1F38E}": "Japanese dolls", "\u{1F38F}": "carp streamer", "\u{1F390}": "wind chime", "\u{1F391}": "moon viewing ceremony", "\u{1F9E7}": "red envelope", "\u{1F380}": "ribbon", "\u{1F381}": "wrapped gift", "\u{1F397}\uFE0F": "reminder ribbon", "\u{1F397}": "reminder ribbon", "\u{1F39F}\uFE0F": "admission tickets", "\u{1F39F}": "admission tickets", "\u{1F3AB}": "ticket", "\u{1F396}\uFE0F": "military medal", "\u{1F396}": "military medal", "\u{1F3C6}": "trophy", "\u{1F3C5}": "sports medal", "\u{1F947}": "1st place medal", "\u{1F948}": "2nd place medal", "\u{1F949}": "3rd place medal", "\u26BD": "soccer ball", "\u26BE": "baseball", "\u{1F94E}": "softball", "\u{1F3C0}": "basketball", "\u{1F3D0}": "volleyball", "\u{1F3C8}": "american football", "\u{1F3C9}": "rugby football", "\u{1F3BE}": "tennis", "\u{1F94F}": "flying disc", "\u{1F3B3}": "bowling", "\u{1F3CF}": "cricket game", "\u{1F3D1}": "field hockey", "\u{1F3D2}": "ice hockey", "\u{1F94D}": "lacrosse", "\u{1F3D3}": "ping pong", "\u{1F3F8}": "badminton", "\u{1F94A}": "boxing glove", "\u{1F94B}": "martial arts uniform", "\u{1F945}": "goal net", "\u26F3": "flag in hole", "\u26F8\uFE0F": "ice skate", "\u26F8": "ice skate", "\u{1F3A3}": "fishing pole", "\u{1F3BD}": "running shirt", "\u{1F3BF}": "skis", "\u{1F6F7}": "sled", "\u{1F94C}": "curling stone", "\u{1F3AF}": "direct hit", "\u{1F3B1}": "pool 8 ball", "\u{1F52E}": "crystal ball", "\u{1F9FF}": "nazar amulet", "\u{1F3AE}": "video game", "\u{1F579}\uFE0F": "joystick", "\u{1F579}": "joystick", "\u{1F3B0}": "slot machine", "\u{1F3B2}": "game die", "\u{1F9E9}": "jigsaw", "\u{1F9F8}": "teddy bear", "\u2660\uFE0F": "spade suit", "\u2660": "spade suit", "\u2665\uFE0F": "heart suit", "\u2665": "heart suit", "\u2666\uFE0F": "diamond suit", "\u2666": "diamond suit", "\u2663\uFE0F": "club suit", "\u2663": "club suit", "\u265F\uFE0F": "chess pawn", "\u265F": "chess pawn", "\u{1F0CF}": "joker", "\u{1F004}": "mahjong red dragon", "\u{1F3B4}": "flower playing cards", "\u{1F3AD}": "performing arts", "\u{1F5BC}\uFE0F": "framed picture", "\u{1F5BC}": "framed picture", "\u{1F3A8}": "artist palette", "\u{1F9F5}": "thread", "\u{1F9F6}": "yarn", "\u{1F507}": "muted speaker", "\u{1F508}": "speaker low volume", "\u{1F509}": "speaker medium volume", "\u{1F50A}": "speaker high volume", "\u{1F4E2}": "loudspeaker", "\u{1F4E3}": "megaphone", "\u{1F4EF}": "postal horn", "\u{1F514}": "bell", "\u{1F515}": "bell with slash", "\u{1F3BC}": "musical score", "\u{1F3B5}": "musical note", "\u{1F3B6}": "musical notes", "\u{1F399}\uFE0F": "studio microphone", "\u{1F399}": "studio microphone", "\u{1F39A}\uFE0F": "level slider", "\u{1F39A}": "level slider", "\u{1F39B}\uFE0F": "control knobs", "\u{1F39B}": "control knobs", "\u{1F3A4}": "microphone", "\u{1F3A7}": "headphone", "\u{1F4FB}": "radio", "\u{1F3B7}": "saxophone", "\u{1F3B8}": "guitar", "\u{1F3B9}": "musical keyboard", "\u{1F3BA}": "trumpet", "\u{1F3BB}": "violin", "\u{1F941}": "drum", "\u{1F4F1}": "mobile phone", "\u{1F4F2}": "mobile phone with arrow", "\u260E\uFE0F": "telephone", "\u260E": "telephone", "\u{1F4DE}": "telephone receiver", "\u{1F4DF}": "pager", "\u{1F4E0}": "fax machine", "\u{1F50B}": "battery", "\u{1F50C}": "electric plug", "\u{1F4BB}": "laptop computer", "\u{1F5A5}\uFE0F": "desktop computer", "\u{1F5A5}": "desktop computer", "\u{1F5A8}\uFE0F": "printer", "\u{1F5A8}": "printer", "\u2328\uFE0F": "keyboard", "\u2328": "keyboard", "\u{1F5B1}\uFE0F": "computer mouse", "\u{1F5B1}": "computer mouse", "\u{1F5B2}\uFE0F": "trackball", "\u{1F5B2}": "trackball", "\u{1F4BD}": "computer disk", "\u{1F4BE}": "floppy disk", "\u{1F4BF}": "optical disk", "\u{1F4C0}": "dvd", "\u{1F9EE}": "abacus", "\u{1F3A5}": "movie camera", "\u{1F39E}\uFE0F": "film frames", "\u{1F39E}": "film frames", "\u{1F4FD}\uFE0F": "film projector", "\u{1F4FD}": "film projector", "\u{1F3AC}": "clapper board", "\u{1F4FA}": "television", "\u{1F4F7}": "camera", "\u{1F4F8}": "camera with flash", "\u{1F4F9}": "video camera", "\u{1F4FC}": "videocassette", "\u{1F50D}": "magnifying glass tilted left", "\u{1F50E}": "magnifying glass tilted right", "\u{1F56F}\uFE0F": "candle", "\u{1F56F}": "candle", "\u{1F4A1}": "light bulb", "\u{1F526}": "flashlight", "\u{1F3EE}": "red paper lantern", "\u{1F4D4}": "notebook with decorative cover", "\u{1F4D5}": "closed book", "\u{1F4D6}": "open book", "\u{1F4D7}": "green book", "\u{1F4D8}": "blue book", "\u{1F4D9}": "orange book", "\u{1F4DA}": "books", "\u{1F4D3}": "notebook", "\u{1F4D2}": "ledger", "\u{1F4C3}": "page with curl", "\u{1F4DC}": "scroll", "\u{1F4C4}": "page facing up", "\u{1F4F0}": "newspaper", "\u{1F5DE}\uFE0F": "rolled-up newspaper", "\u{1F5DE}": "rolled-up newspaper", "\u{1F4D1}": "bookmark tabs", "\u{1F516}": "bookmark", "\u{1F3F7}\uFE0F": "label", "\u{1F3F7}": "label", "\u{1F4B0}": "money bag", "\u{1F4B4}": "yen banknote", "\u{1F4B5}": "dollar banknote", "\u{1F4B6}": "euro banknote", "\u{1F4B7}": "pound banknote", "\u{1F4B8}": "money with wings", "\u{1F4B3}": "credit card", "\u{1F9FE}": "receipt", "\u{1F4B9}": "chart increasing with yen", "\u{1F4B1}": "currency exchange", "\u{1F4B2}": "heavy dollar sign", "\u2709\uFE0F": "envelope", "\u2709": "envelope", "\u{1F4E7}": "e-mail", "\u{1F4E8}": "incoming envelope", "\u{1F4E9}": "envelope with arrow", "\u{1F4E4}": "outbox tray", "\u{1F4E5}": "inbox tray", "\u{1F4E6}": "package", "\u{1F4EB}": "closed mailbox with raised flag", "\u{1F4EA}": "closed mailbox with lowered flag", "\u{1F4EC}": "open mailbox with raised flag", "\u{1F4ED}": "open mailbox with lowered flag", "\u{1F4EE}": "postbox", "\u{1F5F3}\uFE0F": "ballot box with ballot", "\u{1F5F3}": "ballot box with ballot", "\u270F\uFE0F": "pencil", "\u270F": "pencil", "\u2712\uFE0F": "black nib", "\u2712": "black nib", "\u{1F58B}\uFE0F": "fountain pen", "\u{1F58B}": "fountain pen", "\u{1F58A}\uFE0F": "pen", "\u{1F58A}": "pen", "\u{1F58C}\uFE0F": "paintbrush", "\u{1F58C}": "paintbrush", "\u{1F58D}\uFE0F": "crayon", "\u{1F58D}": "crayon", "\u{1F4DD}": "memo", "\u{1F4BC}": "briefcase", "\u{1F4C1}": "file folder", "\u{1F4C2}": "open file folder", "\u{1F5C2}\uFE0F": "card index dividers", "\u{1F5C2}": "card index dividers", "\u{1F4C5}": "calendar", "\u{1F4C6}": "tear-off calendar", "\u{1F5D2}\uFE0F": "spiral notepad", "\u{1F5D2}": "spiral notepad", "\u{1F5D3}\uFE0F": "spiral calendar", "\u{1F5D3}": "spiral calendar", "\u{1F4C7}": "card index", "\u{1F4C8}": "chart increasing", "\u{1F4C9}": "chart decreasing", "\u{1F4CA}": "bar chart", "\u{1F4CB}": "clipboard", "\u{1F4CC}": "pushpin", "\u{1F4CD}": "round pushpin", "\u{1F4CE}": "paperclip", "\u{1F587}\uFE0F": "linked paperclips", "\u{1F587}": "linked paperclips", "\u{1F4CF}": "straight ruler", "\u{1F4D0}": "triangular ruler", "\u2702\uFE0F": "scissors", "\u2702": "scissors", "\u{1F5C3}\uFE0F": "card file box", "\u{1F5C3}": "card file box", "\u{1F5C4}\uFE0F": "file cabinet", "\u{1F5C4}": "file cabinet", "\u{1F5D1}\uFE0F": "wastebasket", "\u{1F5D1}": "wastebasket", "\u{1F512}": "locked", "\u{1F513}": "unlocked", "\u{1F50F}": "locked with pen", "\u{1F510}": "locked with key", "\u{1F511}": "key", "\u{1F5DD}\uFE0F": "old key", "\u{1F5DD}": "old key", "\u{1F528}": "hammer", "\u26CF\uFE0F": "pick", "\u26CF": "pick", "\u2692\uFE0F": "hammer and pick", "\u2692": "hammer and pick", "\u{1F6E0}\uFE0F": "hammer and wrench", "\u{1F6E0}": "hammer and wrench", "\u{1F5E1}\uFE0F": "dagger", "\u{1F5E1}": "dagger", "\u2694\uFE0F": "crossed swords", "\u2694": "crossed swords", "\u{1F52B}": "pistol", "\u{1F3F9}": "bow and arrow", "\u{1F6E1}\uFE0F": "shield", "\u{1F6E1}": "shield", "\u{1F527}": "wrench", "\u{1F529}": "nut and bolt", "\u2699\uFE0F": "gear", "\u2699": "gear", "\u{1F5DC}\uFE0F": "clamp", "\u{1F5DC}": "clamp", "\u2696\uFE0F": "balance scale", "\u2696": "balance scale", "\u{1F517}": "link", "\u26D3\uFE0F": "chains", "\u26D3": "chains", "\u{1F9F0}": "toolbox", "\u{1F9F2}": "magnet", "\u2697\uFE0F": "alembic", "\u2697": "alembic", "\u{1F9EA}": "test tube", "\u{1F9EB}": "petri dish", "\u{1F9EC}": "dna", "\u{1F52C}": "microscope", "\u{1F52D}": "telescope", "\u{1F4E1}": "satellite antenna", "\u{1F489}": "syringe", "\u{1F48A}": "pill", "\u{1F6AA}": "door", "\u{1F6CF}\uFE0F": "bed", "\u{1F6CF}": "bed", "\u{1F6CB}\uFE0F": "couch and lamp", "\u{1F6CB}": "couch and lamp", "\u{1F6BD}": "toilet", "\u{1F6BF}": "shower", "\u{1F6C1}": "bathtub", "\u{1F9F4}": "lotion bottle", "\u{1F9F7}": "safety pin", "\u{1F9F9}": "broom", "\u{1F9FA}": "basket", "\u{1F9FB}": "roll of paper", "\u{1F9FC}": "soap", "\u{1F9FD}": "sponge", "\u{1F9EF}": "fire extinguisher", "\u{1F6D2}": "shopping cart", "\u{1F6AC}": "cigarette", "\u26B0\uFE0F": "coffin", "\u26B0": "coffin", "\u26B1\uFE0F": "funeral urn", "\u26B1": "funeral urn", "\u{1F5FF}": "moai", "\u{1F3E7}": "ATM sign", "\u{1F6AE}": "litter in bin sign", "\u{1F6B0}": "potable water", "\u267F": "wheelchair symbol", "\u{1F6B9}": "men\u2019s room", "\u{1F6BA}": "women\u2019s room", "\u{1F6BB}": "restroom", "\u{1F6BC}": "baby symbol", "\u{1F6BE}": "water closet", "\u{1F6C2}": "passport control", "\u{1F6C3}": "customs", "\u{1F6C4}": "baggage claim", "\u{1F6C5}": "left luggage", "\u26A0\uFE0F": "warning", "\u26A0": "warning", "\u{1F6B8}": "children crossing", "\u26D4": "no entry", "\u{1F6AB}": "prohibited", "\u{1F6B3}": "no bicycles", "\u{1F6AD}": "no smoking", "\u{1F6AF}": "no littering", "\u{1F6B1}": "non-potable water", "\u{1F6B7}": "no pedestrians", "\u{1F4F5}": "no mobile phones", "\u{1F51E}": "no one under eighteen", "\u2622\uFE0F": "radioactive", "\u2622": "radioactive", "\u2623\uFE0F": "biohazard", "\u2623": "biohazard", "\u2B06\uFE0F": "up arrow", "\u2B06": "up arrow", "\u2197\uFE0F": "up-right arrow", "\u2197": "up-right arrow", "\u27A1\uFE0F": "right arrow", "\u27A1": "right arrow", "\u2198\uFE0F": "down-right arrow", "\u2198": "down-right arrow", "\u2B07\uFE0F": "down arrow", "\u2B07": "down arrow", "\u2199\uFE0F": "down-left arrow", "\u2199": "down-left arrow", "\u2B05\uFE0F": "left arrow", "\u2B05": "left arrow", "\u2196\uFE0F": "up-left arrow", "\u2196": "up-left arrow", "\u2195\uFE0F": "up-down arrow", "\u2195": "up-down arrow", "\u2194\uFE0F": "left-right arrow", "\u2194": "left-right arrow", "\u21A9\uFE0F": "right arrow curving left", "\u21A9": "right arrow curving left", "\u21AA\uFE0F": "left arrow curving right", "\u21AA": "left arrow curving right", "\u2934\uFE0F": "right arrow curving up", "\u2934": "right arrow curving up", "\u2935\uFE0F": "right arrow curving down", "\u2935": "right arrow curving down", "\u{1F503}": "clockwise vertical arrows", "\u{1F504}": "counterclockwise arrows button", "\u{1F519}": "BACK arrow", "\u{1F51A}": "END arrow", "\u{1F51B}": "ON! arrow", "\u{1F51C}": "SOON arrow", "\u{1F51D}": "TOP arrow", "\u{1F6D0}": "place of worship", "\u269B\uFE0F": "atom symbol", "\u269B": "atom symbol", "\u{1F549}\uFE0F": "om", "\u{1F549}": "om", "\u2721\uFE0F": "star of David", "\u2721": "star of David", "\u2638\uFE0F": "wheel of dharma", "\u2638": "wheel of dharma", "\u262F\uFE0F": "yin yang", "\u262F": "yin yang", "\u271D\uFE0F": "latin cross", "\u271D": "latin cross", "\u2626\uFE0F": "orthodox cross", "\u2626": "orthodox cross", "\u262A\uFE0F": "star and crescent", "\u262A": "star and crescent", "\u262E\uFE0F": "peace symbol", "\u262E": "peace symbol", "\u{1F54E}": "menorah", "\u{1F52F}": "dotted six-pointed star", "\u2648": "Aries", "\u2649": "Taurus", "\u264A": "Gemini", "\u264B": "Cancer", "\u264C": "Leo", "\u264D": "Virgo", "\u264E": "Libra", "\u264F": "Scorpio", "\u2650": "Sagittarius", "\u2651": "Capricorn", "\u2652": "Aquarius", "\u2653": "Pisces", "\u26CE": "Ophiuchus", "\u{1F500}": "shuffle tracks button", "\u{1F501}": "repeat button", "\u{1F502}": "repeat single button", "\u25B6\uFE0F": "play button", "\u25B6": "play button", "\u23E9": "fast-forward button", "\u23ED\uFE0F": "next track button", "\u23ED": "next track button", "\u23EF\uFE0F": "play or pause button", "\u23EF": "play or pause button", "\u25C0\uFE0F": "reverse button", "\u25C0": "reverse button", "\u23EA": "fast reverse button", "\u23EE\uFE0F": "last track button", "\u23EE": "last track button", "\u{1F53C}": "upwards button", "\u23EB": "fast up button", "\u{1F53D}": "downwards button", "\u23EC": "fast down button", "\u23F8\uFE0F": "pause button", "\u23F8": "pause button", "\u23F9\uFE0F": "stop button", "\u23F9": "stop button", "\u23FA\uFE0F": "record button", "\u23FA": "record button", "\u23CF\uFE0F": "eject button", "\u23CF": "eject button", "\u{1F3A6}": "cinema", "\u{1F505}": "dim button", "\u{1F506}": "bright button", "\u{1F4F6}": "antenna bars", "\u{1F4F3}": "vibration mode", "\u{1F4F4}": "mobile phone off", "\u2640\uFE0F": "female sign", "\u2640": "female sign", "\u2642\uFE0F": "male sign", "\u2642": "male sign", "\u2695\uFE0F": "medical symbol", "\u2695": "medical symbol", "\u267E\uFE0F": "infinity", "\u267E": "infinity", "\u267B\uFE0F": "recycling symbol", "\u267B": "recycling symbol", "\u269C\uFE0F": "fleur-de-lis", "\u269C": "fleur-de-lis", "\u{1F531}": "trident emblem", "\u{1F4DB}": "name badge", "\u{1F530}": "Japanese symbol for beginner", "\u2B55": "heavy large circle", "\u2705": "white heavy check mark", "\u2611\uFE0F": "ballot box with check", "\u2611": "ballot box with check", "\u2714\uFE0F": "heavy check mark", "\u2714": "heavy check mark", "\u2716\uFE0F": "heavy multiplication x", "\u2716": "heavy multiplication x", "\u274C": "cross mark", "\u274E": "cross mark button", "\u2795": "heavy plus sign", "\u2796": "heavy minus sign", "\u2797": "heavy division sign", "\u27B0": "curly loop", "\u27BF": "double curly loop", "\u303D\uFE0F": "part alternation mark", "\u303D": "part alternation mark", "\u2733\uFE0F": "eight-spoked asterisk", "\u2733": "eight-spoked asterisk", "\u2734\uFE0F": "eight-pointed star", "\u2734": "eight-pointed star", "\u2747\uFE0F": "sparkle", "\u2747": "sparkle", "\u203C\uFE0F": "double exclamation mark", "\u203C": "double exclamation mark", "\u2049\uFE0F": "exclamation question mark", "\u2049": "exclamation question mark", "\u2753": "question mark", "\u2754": "white question mark", "\u2755": "white exclamation mark", "\u2757": "exclamation mark", "\u3030\uFE0F": "wavy dash", "\u3030": "wavy dash", "\xA9\uFE0F": "copyright", "\xA9": "copyright", "\xAE\uFE0F": "registered", "\xAE": "registered", "\u2122\uFE0F": "trade mark", "\u2122": "trade mark", "#\uFE0F\u20E3": "keycap: #", "#\u20E3": "keycap: #", "*\uFE0F\u20E3": "keycap: *", "*\u20E3": "keycap: *", "0\uFE0F\u20E3": "keycap: 0", "0\u20E3": "keycap: 0", "1\uFE0F\u20E3": "keycap: 1", "1\u20E3": "keycap: 1", "2\uFE0F\u20E3": "keycap: 2", "2\u20E3": "keycap: 2", "3\uFE0F\u20E3": "keycap: 3", "3\u20E3": "keycap: 3", "4\uFE0F\u20E3": "keycap: 4", "4\u20E3": "keycap: 4", "5\uFE0F\u20E3": "keycap: 5", "5\u20E3": "keycap: 5", "6\uFE0F\u20E3": "keycap: 6", "6\u20E3": "keycap: 6", "7\uFE0F\u20E3": "keycap: 7", "7\u20E3": "keycap: 7", "8\uFE0F\u20E3": "keycap: 8", "8\u20E3": "keycap: 8", "9\uFE0F\u20E3": "keycap: 9", "9\u20E3": "keycap: 9", "\u{1F51F}": "keycap: 10", "\u{1F4AF}": "hundred points", "\u{1F520}": "input latin uppercase", "\u{1F521}": "input latin lowercase", "\u{1F522}": "input numbers", "\u{1F523}": "input symbols", "\u{1F524}": "input latin letters", "\u{1F170}\uFE0F": "A button (blood type)", "\u{1F170}": "A button (blood type)", "\u{1F18E}": "AB button (blood type)", "\u{1F171}\uFE0F": "B button (blood type)", "\u{1F171}": "B button (blood type)", "\u{1F191}": "CL button", "\u{1F192}": "COOL button", "\u{1F193}": "FREE button", "\u2139\uFE0F": "information", \u2139: "information", "\u{1F194}": "ID button", "\u24C2\uFE0F": "circled M", "\u24C2": "circled M", "\u{1F195}": "NEW button", "\u{1F196}": "NG button", "\u{1F17E}\uFE0F": "O button (blood type)", "\u{1F17E}": "O button (blood type)", "\u{1F197}": "OK button", "\u{1F17F}\uFE0F": "P button", "\u{1F17F}": "P button", "\u{1F198}": "SOS button", "\u{1F199}": "UP! button", "\u{1F19A}": "VS button", "\u{1F201}": "Japanese \u201Chere\u201D button", "\u{1F202}\uFE0F": "Japanese \u201Cservice charge\u201D button", "\u{1F202}": "Japanese \u201Cservice charge\u201D button", "\u{1F237}\uFE0F": "Japanese \u201Cmonthly amount\u201D button", "\u{1F237}": "Japanese \u201Cmonthly amount\u201D button", "\u{1F236}": "Japanese \u201Cnot free of charge\u201D button", "\u{1F22F}": "Japanese \u201Creserved\u201D button", "\u{1F250}": "Japanese \u201Cbargain\u201D button", "\u{1F239}": "Japanese \u201Cdiscount\u201D button", "\u{1F21A}": "Japanese \u201Cfree of charge\u201D button", "\u{1F232}": "Japanese \u201Cprohibited\u201D button", "\u{1F251}": "Japanese \u201Cacceptable\u201D button", "\u{1F238}": "Japanese \u201Capplication\u201D button", "\u{1F234}": "Japanese \u201Cpassing grade\u201D button", "\u{1F233}": "Japanese \u201Cvacancy\u201D button", "\u3297\uFE0F": "Japanese \u201Ccongratulations\u201D button", "\u3297": "Japanese \u201Ccongratulations\u201D button", "\u3299\uFE0F": "Japanese \u201Csecret\u201D button", "\u3299": "Japanese \u201Csecret\u201D button", "\u{1F23A}": "Japanese \u201Copen for business\u201D button", "\u{1F235}": "Japanese \u201Cno vacancy\u201D button", "\u25AA\uFE0F": "black small square", "\u25AA": "black small square", "\u25AB\uFE0F": "white small square", "\u25AB": "white small square", "\u25FB\uFE0F": "white medium square", "\u25FB": "white medium square", "\u25FC\uFE0F": "black medium square", "\u25FC": "black medium square", "\u25FD": "white medium-small square", "\u25FE": "black medium-small square", "\u2B1B": "black large square", "\u2B1C": "white large square", "\u{1F536}": "large orange diamond", "\u{1F537}": "large blue diamond", "\u{1F538}": "small orange diamond", "\u{1F539}": "small blue diamond", "\u{1F53A}": "red triangle pointed up", "\u{1F53B}": "red triangle pointed down", "\u{1F4A0}": "diamond with a dot", "\u{1F518}": "radio button", "\u{1F532}": "black square button", "\u{1F533}": "white square button", "\u26AA": "white circle", "\u26AB": "black circle", "\u{1F534}": "red circle", "\u{1F535}": "blue circle", "\u{1F3C1}": "chequered flag", "\u{1F6A9}": "triangular flag", "\u{1F38C}": "crossed flags", "\u{1F3F4}": "black flag", "\u{1F3F3}\uFE0F": "white flag", "\u{1F3F3}": "white flag", "\u{1F3F3}\uFE0F\u200D\u{1F308}": "rainbow flag", "\u{1F3F3}\u200D\u{1F308}": "rainbow flag", "\u{1F3F4}\u200D\u2620\uFE0F": "pirate flag", "\u{1F3F4}\u200D\u2620": "pirate flag", "\u{1F1E6}\u{1F1E8}": "Ascension Island", "\u{1F1E6}\u{1F1E9}": "Andorra", "\u{1F1E6}\u{1F1EA}": "United Arab Emirates", "\u{1F1E6}\u{1F1EB}": "Afghanistan", "\u{1F1E6}\u{1F1EC}": "Antigua & Barbuda", "\u{1F1E6}\u{1F1EE}": "Anguilla", "\u{1F1E6}\u{1F1F1}": "Albania", "\u{1F1E6}\u{1F1F2}": "Armenia", "\u{1F1E6}\u{1F1F4}": "Angola", "\u{1F1E6}\u{1F1F6}": "Antarctica", "\u{1F1E6}\u{1F1F7}": "Argentina", "\u{1F1E6}\u{1F1F8}": "American Samoa", "\u{1F1E6}\u{1F1F9}": "Austria", "\u{1F1E6}\u{1F1FA}": "Australia", "\u{1F1E6}\u{1F1FC}": "Aruba", "\u{1F1E6}\u{1F1FD}": "\xC5land Islands", "\u{1F1E6}\u{1F1FF}": "Azerbaijan", "\u{1F1E7}\u{1F1E6}": "Bosnia & Herzegovina", "\u{1F1E7}\u{1F1E7}": "Barbados", "\u{1F1E7}\u{1F1E9}": "Bangladesh", "\u{1F1E7}\u{1F1EA}": "Belgium", "\u{1F1E7}\u{1F1EB}": "Burkina Faso", "\u{1F1E7}\u{1F1EC}": "Bulgaria", "\u{1F1E7}\u{1F1ED}": "Bahrain", "\u{1F1E7}\u{1F1EE}": "Burundi", "\u{1F1E7}\u{1F1EF}": "Benin", "\u{1F1E7}\u{1F1F1}": "St. Barth\xE9lemy", "\u{1F1E7}\u{1F1F2}": "Bermuda", "\u{1F1E7}\u{1F1F3}": "Brunei", "\u{1F1E7}\u{1F1F4}": "Bolivia", "\u{1F1E7}\u{1F1F6}": "Caribbean Netherlands", "\u{1F1E7}\u{1F1F7}": "Brazil", "\u{1F1E7}\u{1F1F8}": "Bahamas", "\u{1F1E7}\u{1F1F9}": "Bhutan", "\u{1F1E7}\u{1F1FB}": "Bouvet Island", "\u{1F1E7}\u{1F1FC}": "Botswana", "\u{1F1E7}\u{1F1FE}": "Belarus", "\u{1F1E7}\u{1F1FF}": "Belize", "\u{1F1E8}\u{1F1E6}": "Canada", "\u{1F1E8}\u{1F1E8}": "Cocos (Keeling) Islands", "\u{1F1E8}\u{1F1E9}": "Congo - Kinshasa", "\u{1F1E8}\u{1F1EB}": "Central African Republic", "\u{1F1E8}\u{1F1EC}": "Congo - Brazzaville", "\u{1F1E8}\u{1F1ED}": "Switzerland", "\u{1F1E8}\u{1F1EE}": "C\xF4te d\u2019Ivoire", "\u{1F1E8}\u{1F1F0}": "Cook Islands", "\u{1F1E8}\u{1F1F1}": "Chile", "\u{1F1E8}\u{1F1F2}": "Cameroon", "\u{1F1E8}\u{1F1F3}": "China", "\u{1F1E8}\u{1F1F4}": "Colombia", "\u{1F1E8}\u{1F1F5}": "Clipperton Island", "\u{1F1E8}\u{1F1F7}": "Costa Rica", "\u{1F1E8}\u{1F1FA}": "Cuba", "\u{1F1E8}\u{1F1FB}": "Cape Verde", "\u{1F1E8}\u{1F1FC}": "Cura\xE7ao", "\u{1F1E8}\u{1F1FD}": "Christmas Island", "\u{1F1E8}\u{1F1FE}": "Cyprus", "\u{1F1E8}\u{1F1FF}": "Czechia", "\u{1F1E9}\u{1F1EA}": "Germany", "\u{1F1E9}\u{1F1EC}": "Diego Garcia", "\u{1F1E9}\u{1F1EF}": "Djibouti", "\u{1F1E9}\u{1F1F0}": "Denmark", "\u{1F1E9}\u{1F1F2}": "Dominica", "\u{1F1E9}\u{1F1F4}": "Dominican Republic", "\u{1F1E9}\u{1F1FF}": "Algeria", "\u{1F1EA}\u{1F1E6}": "Ceuta & Melilla", "\u{1F1EA}\u{1F1E8}": "Ecuador", "\u{1F1EA}\u{1F1EA}": "Estonia", "\u{1F1EA}\u{1F1EC}": "Egypt", "\u{1F1EA}\u{1F1ED}": "Western Sahara", "\u{1F1EA}\u{1F1F7}": "Eritrea", "\u{1F1EA}\u{1F1F8}": "Spain", "\u{1F1EA}\u{1F1F9}": "Ethiopia", "\u{1F1EA}\u{1F1FA}": "European Union", "\u{1F1EB}\u{1F1EE}": "Finland", "\u{1F1EB}\u{1F1EF}": "Fiji", "\u{1F1EB}\u{1F1F0}": "Falkland Islands", "\u{1F1EB}\u{1F1F2}": "Micronesia", "\u{1F1EB}\u{1F1F4}": "Faroe Islands", "\u{1F1EB}\u{1F1F7}": "France", "\u{1F1EC}\u{1F1E6}": "Gabon", "\u{1F1EC}\u{1F1E7}": "United Kingdom", "\u{1F1EC}\u{1F1E9}": "Grenada", "\u{1F1EC}\u{1F1EA}": "Georgia", "\u{1F1EC}\u{1F1EB}": "French Guiana", "\u{1F1EC}\u{1F1EC}": "Guernsey", "\u{1F1EC}\u{1F1ED}": "Ghana", "\u{1F1EC}\u{1F1EE}": "Gibraltar", "\u{1F1EC}\u{1F1F1}": "Greenland", "\u{1F1EC}\u{1F1F2}": "Gambia", "\u{1F1EC}\u{1F1F3}": "Guinea", "\u{1F1EC}\u{1F1F5}": "Guadeloupe", "\u{1F1EC}\u{1F1F6}": "Equatorial Guinea", "\u{1F1EC}\u{1F1F7}": "Greece", "\u{1F1EC}\u{1F1F8}": "South Georgia & South Sandwich Islands", "\u{1F1EC}\u{1F1F9}": "Guatemala", "\u{1F1EC}\u{1F1FA}": "Guam", "\u{1F1EC}\u{1F1FC}": "Guinea-Bissau", "\u{1F1EC}\u{1F1FE}": "Guyana", "\u{1F1ED}\u{1F1F0}": "Hong Kong SAR China", "\u{1F1ED}\u{1F1F2}": "Heard & McDonald Islands", "\u{1F1ED}\u{1F1F3}": "Honduras", "\u{1F1ED}\u{1F1F7}": "Croatia", "\u{1F1ED}\u{1F1F9}": "Haiti", "\u{1F1ED}\u{1F1FA}": "Hungary", "\u{1F1EE}\u{1F1E8}": "Canary Islands", "\u{1F1EE}\u{1F1E9}": "Indonesia", "\u{1F1EE}\u{1F1EA}": "Ireland", "\u{1F1EE}\u{1F1F1}": "Israel", "\u{1F1EE}\u{1F1F2}": "Isle of Man", "\u{1F1EE}\u{1F1F3}": "India", "\u{1F1EE}\u{1F1F4}": "British Indian Ocean Territory", "\u{1F1EE}\u{1F1F6}": "Iraq", "\u{1F1EE}\u{1F1F7}": "Iran", "\u{1F1EE}\u{1F1F8}": "Iceland", "\u{1F1EE}\u{1F1F9}": "Italy", "\u{1F1EF}\u{1F1EA}": "Jersey", "\u{1F1EF}\u{1F1F2}": "Jamaica", "\u{1F1EF}\u{1F1F4}": "Jordan", "\u{1F1EF}\u{1F1F5}": "Japan", "\u{1F1F0}\u{1F1EA}": "Kenya", "\u{1F1F0}\u{1F1EC}": "Kyrgyzstan", "\u{1F1F0}\u{1F1ED}": "Cambodia", "\u{1F1F0}\u{1F1EE}": "Kiribati", "\u{1F1F0}\u{1F1F2}": "Comoros", "\u{1F1F0}\u{1F1F3}": "St. Kitts & Nevis", "\u{1F1F0}\u{1F1F5}": "North Korea", "\u{1F1F0}\u{1F1F7}": "South Korea", "\u{1F1F0}\u{1F1FC}": "Kuwait", "\u{1F1F0}\u{1F1FE}": "Cayman Islands", "\u{1F1F0}\u{1F1FF}": "Kazakhstan", "\u{1F1F1}\u{1F1E6}": "Laos", "\u{1F1F1}\u{1F1E7}": "Lebanon", "\u{1F1F1}\u{1F1E8}": "St. Lucia", "\u{1F1F1}\u{1F1EE}": "Liechtenstein", "\u{1F1F1}\u{1F1F0}": "Sri Lanka", "\u{1F1F1}\u{1F1F7}": "Liberia", "\u{1F1F1}\u{1F1F8}": "Lesotho", "\u{1F1F1}\u{1F1F9}": "Lithuania", "\u{1F1F1}\u{1F1FA}": "Luxembourg", "\u{1F1F1}\u{1F1FB}": "Latvia", "\u{1F1F1}\u{1F1FE}": "Libya", "\u{1F1F2}\u{1F1E6}": "Morocco", "\u{1F1F2}\u{1F1E8}": "Monaco", "\u{1F1F2}\u{1F1E9}": "Moldova", "\u{1F1F2}\u{1F1EA}": "Montenegro", "\u{1F1F2}\u{1F1EB}": "St. Martin", "\u{1F1F2}\u{1F1EC}": "Madagascar", "\u{1F1F2}\u{1F1ED}": "Marshall Islands", "\u{1F1F2}\u{1F1F0}": "Macedonia", "\u{1F1F2}\u{1F1F1}": "Mali", "\u{1F1F2}\u{1F1F2}": "Myanmar (Burma)", "\u{1F1F2}\u{1F1F3}": "Mongolia", "\u{1F1F2}\u{1F1F4}": "Macau SAR China", "\u{1F1F2}\u{1F1F5}": "Northern Mariana Islands", "\u{1F1F2}\u{1F1F6}": "Martinique", "\u{1F1F2}\u{1F1F7}": "Mauritania", "\u{1F1F2}\u{1F1F8}": "Montserrat", "\u{1F1F2}\u{1F1F9}": "Malta", "\u{1F1F2}\u{1F1FA}": "Mauritius", "\u{1F1F2}\u{1F1FB}": "Maldives", "\u{1F1F2}\u{1F1FC}": "Malawi", "\u{1F1F2}\u{1F1FD}": "Mexico", "\u{1F1F2}\u{1F1FE}": "Malaysia", "\u{1F1F2}\u{1F1FF}": "Mozambique", "\u{1F1F3}\u{1F1E6}": "Namibia", "\u{1F1F3}\u{1F1E8}": "New Caledonia", "\u{1F1F3}\u{1F1EA}": "Niger", "\u{1F1F3}\u{1F1EB}": "Norfolk Island", "\u{1F1F3}\u{1F1EC}": "Nigeria", "\u{1F1F3}\u{1F1EE}": "Nicaragua", "\u{1F1F3}\u{1F1F1}": "Netherlands", "\u{1F1F3}\u{1F1F4}": "Norway", "\u{1F1F3}\u{1F1F5}": "Nepal", "\u{1F1F3}\u{1F1F7}": "Nauru", "\u{1F1F3}\u{1F1FA}": "Niue", "\u{1F1F3}\u{1F1FF}": "New Zealand", "\u{1F1F4}\u{1F1F2}": "Oman", "\u{1F1F5}\u{1F1E6}": "Panama", "\u{1F1F5}\u{1F1EA}": "Peru", "\u{1F1F5}\u{1F1EB}": "French Polynesia", "\u{1F1F5}\u{1F1EC}": "Papua New Guinea", "\u{1F1F5}\u{1F1ED}": "Philippines", "\u{1F1F5}\u{1F1F0}": "Pakistan", "\u{1F1F5}\u{1F1F1}": "Poland", "\u{1F1F5}\u{1F1F2}": "St. Pierre & Miquelon", "\u{1F1F5}\u{1F1F3}": "Pitcairn Islands", "\u{1F1F5}\u{1F1F7}": "Puerto Rico", "\u{1F1F5}\u{1F1F8}": "Palestinian Territories", "\u{1F1F5}\u{1F1F9}": "Portugal", "\u{1F1F5}\u{1F1FC}": "Palau", "\u{1F1F5}\u{1F1FE}": "Paraguay", "\u{1F1F6}\u{1F1E6}": "Qatar", "\u{1F1F7}\u{1F1EA}": "R\xE9union", "\u{1F1F7}\u{1F1F4}": "Romania", "\u{1F1F7}\u{1F1F8}": "Serbia", "\u{1F1F7}\u{1F1FA}": "Russia", "\u{1F1F7}\u{1F1FC}": "Rwanda", "\u{1F1F8}\u{1F1E6}": "Saudi Arabia", "\u{1F1F8}\u{1F1E7}": "Solomon Islands", "\u{1F1F8}\u{1F1E8}": "Seychelles", "\u{1F1F8}\u{1F1E9}": "Sudan", "\u{1F1F8}\u{1F1EA}": "Sweden", "\u{1F1F8}\u{1F1EC}": "Singapore", "\u{1F1F8}\u{1F1ED}": "St. Helena", "\u{1F1F8}\u{1F1EE}": "Slovenia", "\u{1F1F8}\u{1F1EF}": "Svalbard & Jan Mayen", "\u{1F1F8}\u{1F1F0}": "Slovakia", "\u{1F1F8}\u{1F1F1}": "Sierra Leone", "\u{1F1F8}\u{1F1F2}": "San Marino", "\u{1F1F8}\u{1F1F3}": "Senegal", "\u{1F1F8}\u{1F1F4}": "Somalia", "\u{1F1F8}\u{1F1F7}": "Suriname", "\u{1F1F8}\u{1F1F8}": "South Sudan", "\u{1F1F8}\u{1F1F9}": "S\xE3o Tom\xE9 & Pr\xEDncipe", "\u{1F1F8}\u{1F1FB}": "El Salvador", "\u{1F1F8}\u{1F1FD}": "Sint Maarten", "\u{1F1F8}\u{1F1FE}": "Syria", "\u{1F1F8}\u{1F1FF}": "Swaziland", "\u{1F1F9}\u{1F1E6}": "Tristan da Cunha", "\u{1F1F9}\u{1F1E8}": "Turks & Caicos Islands", "\u{1F1F9}\u{1F1E9}": "Chad", "\u{1F1F9}\u{1F1EB}": "French Southern Territories", "\u{1F1F9}\u{1F1EC}": "Togo", "\u{1F1F9}\u{1F1ED}": "Thailand", "\u{1F1F9}\u{1F1EF}": "Tajikistan", "\u{1F1F9}\u{1F1F0}": "Tokelau", "\u{1F1F9}\u{1F1F1}": "Timor-Leste", "\u{1F1F9}\u{1F1F2}": "Turkmenistan", "\u{1F1F9}\u{1F1F3}": "Tunisia", "\u{1F1F9}\u{1F1F4}": "Tonga", "\u{1F1F9}\u{1F1F7}": "Turkey", "\u{1F1F9}\u{1F1F9}": "Trinidad & Tobago", "\u{1F1F9}\u{1F1FB}": "Tuvalu", "\u{1F1F9}\u{1F1FC}": "Taiwan", "\u{1F1F9}\u{1F1FF}": "Tanzania", "\u{1F1FA}\u{1F1E6}": "Ukraine", "\u{1F1FA}\u{1F1EC}": "Uganda", "\u{1F1FA}\u{1F1F2}": "U.S. Outlying Islands", "\u{1F1FA}\u{1F1F3}": "United Nations", "\u{1F1FA}\u{1F1F8}": "United States", "\u{1F1FA}\u{1F1FE}": "Uruguay", "\u{1F1FA}\u{1F1FF}": "Uzbekistan", "\u{1F1FB}\u{1F1E6}": "Vatican City", "\u{1F1FB}\u{1F1E8}": "St. Vincent & Grenadines", "\u{1F1FB}\u{1F1EA}": "Venezuela", "\u{1F1FB}\u{1F1EC}": "British Virgin Islands", "\u{1F1FB}\u{1F1EE}": "U.S. Virgin Islands", "\u{1F1FB}\u{1F1F3}": "Vietnam", "\u{1F1FB}\u{1F1FA}": "Vanuatu", "\u{1F1FC}\u{1F1EB}": "Wallis & Futuna", "\u{1F1FC}\u{1F1F8}": "Samoa", "\u{1F1FD}\u{1F1F0}": "Kosovo", "\u{1F1FE}\u{1F1EA}": "Yemen", "\u{1F1FE}\u{1F1F9}": "Mayotte", "\u{1F1FF}\u{1F1E6}": "South Africa", "\u{1F1FF}\u{1F1F2}": "Zambia", "\u{1F1FF}\u{1F1FC}": "Zimbabwe", "\u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F}": "England", "\u{1F3F4}\u{E0067}\u{E0062}\u{E0073}\u{E0063}\u{E0074}\u{E007F}": "Scotland", "\u{1F3F4}\u{E0067}\u{E0062}\u{E0077}\u{E006C}\u{E0073}\u{E007F}": "Wales" }, boosters: { absolutely: 0.293, amazingly: 0.293, awfully: 0.293, completely: 0.293, considerable: 0.293, considerably: 0.293, decidedly: 0.293, deeply: 0.293, effing: 0.293, enormous: 0.293, enormously: 0.293, entirely: 0.293, especially: 0.293, exceptional: 0.293, exceptionally: 0.293, extreme: 0.293, extremely: 0.293, fabulously: 0.293, flipping: 0.293, flippin: 0.293, frackin: 0.293, fracking: 0.293, fricking: 0.293, frickin: 0.293, frigging: 0.293, friggin: 0.293, fully: 0.293, fuckin: 0.293, fucking: 0.293, fuggin: 0.293, fugging: 0.293, greatly: 0.293, hella: 0.293, highly: 0.293, hugely: 0.293, incredible: 0.293, incredibly: 0.293, intensely: 0.293, major: 0.293, majorly: 0.293, more: 0.293, most: 0.293, particularly: 0.293, purely: 0.293, quite: 0.293, really: 0.293, remarkably: 0.293, so: 0.293, substantially: 0.293, thoroughly: 0.293, total: 0.293, totally: 0.293, tremendous: 0.293, tremendously: 0.293, uber: 0.293, unbelievably: 0.293, unusually: 0.293, utter: 0.293, utterly: 0.293, very: 0.293, almost: -0.293, barely: -0.293, hardly: -0.293, "just enough": -0.293, "kind of": -0.293, kinda: -0.293, kindof: -0.293, "kind-of": -0.293, less: -0.293, little: -0.293, marginal: -0.293, marginally: -0.293, occasional: -0.293, occasionally: -0.293, partly: -0.293, scarce: -0.293, scarcely: -0.293, slight: -0.293, slightly: -0.293, somewhat: -0.293, "sort of": -0.293, sorta: -0.293, sortof: -0.293, "sort-of": -0.293 }, negators: ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent", "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't", "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"], specialCases: { "the shit": 3, "the bomb": 3, "bad ass": 1.5, badass: 1.5, "bus stop": 0, "yeah right": -2, "kiss of death": -1.5, "to die for": 3, "beating heart": 3.5 } };

// web/lib/vader-sentiment.ts
var lexicon = vader_data_default.lexicon;
var emojis = vader_data_default.emojis;
var boosters = vader_data_default.boosters;
var specialCases = vader_data_default.specialCases;
var negators = new Set(vader_data_default.negators);
var N_SCALAR = -0.74;
var C_INCR = 0.733;
var has = (object, key) => Object.hasOwn(object, key);
var isUpper = (word) => word !== word.toLowerCase() && word === word.toUpperCase();
var negated = (word) => negators.has(word) || word.includes("n't");
function round(value, places) {
  const scale = 10 ** places;
  const scaled = Math.abs(value) * scale;
  if (scaled % 1 === 0.5) return Math.sign(value) * (Math.round(scaled / 2) * 2) / scale;
  return Number(value.toFixed(places));
}
function expandEmoji(text) {
  let output = "";
  let previousSpace = true;
  for (const character of text) {
    if (has(emojis, character)) {
      if (!previousSpace) output += " ";
      output += emojis[character];
      previousSpace = false;
    } else {
      output += character;
      previousSpace = character === " ";
    }
  }
  return output.trim();
}
function tokens(text) {
  return text ? text.split(/\s+/u).map((token) => {
    const stripped = token.replace(/^[!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~]+|[!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~]+$/gu, "");
    return stripped.length <= 2 ? token : stripped;
  }) : [];
}
function negationCheck(valence, words, distance, index) {
  if (distance === 1) {
    if (words[index - 2] === "never" && ["so", "this"].includes(words[index - 1])) return valence * 1.25;
    if (words[index - 2] === "without" && words[index - 1] === "doubt") return valence;
  }
  if (distance === 2) {
    if (words[index - 3] === "never" && ["so", "this"].includes(words[index - 2]) || ["so", "this"].includes(words[index - 1])) return valence * 1.25;
    if (words[index - 3] === "without" && [words[index - 2], words[index - 1]].includes("doubt")) return valence;
  }
  return negated(words[index - distance - 1]) ? valence * N_SCALAR : valence;
}
function idiomsCheck(valence, words, index) {
  const phrase = (from, to) => words.slice(from, to + 1).join(" ");
  const sequences = [
    phrase(index - 1, index),
    phrase(index - 2, index),
    phrase(index - 2, index - 1),
    phrase(index - 3, index - 1),
    phrase(index - 3, index - 2)
  ];
  for (const sequence of sequences) {
    if (has(specialCases, sequence)) {
      valence = specialCases[sequence];
      break;
    }
  }
  for (const length of [1, 2]) {
    if (words.length - 1 >= index + length) {
      const sequence = phrase(index, index + length);
      if (has(specialCases, sequence)) valence = specialCases[sequence];
    }
  }
  for (const sequence of [sequences[3], sequences[4], sequences[2]]) {
    if (has(boosters, sequence)) valence += boosters[sequence];
  }
  return valence;
}
function wordValence(original, words, index, capitalDifference) {
  const word = words[index];
  if (has(boosters, word) || word === "kind" && words[index + 1] === "of" || !has(lexicon, word)) return 0;
  let valence = lexicon[word];
  if (word === "no" && index !== words.length - 1 && has(lexicon, words[index + 1])) valence = 0;
  if (index > 0 && words[index - 1] === "no" || index > 1 && words[index - 2] === "no" || index > 2 && words[index - 3] === "no" && ["or", "nor"].includes(words[index - 1])) {
    valence = lexicon[word] * N_SCALAR;
  }
  if (isUpper(original[index]) && capitalDifference) valence += valence > 0 ? C_INCR : -C_INCR;
  for (let distance = 0; distance < 3; distance += 1) {
    if (index <= distance || has(lexicon, words[index - distance - 1])) continue;
    const preceding = words[index - distance - 1];
    let scalar = 0;
    if (has(boosters, preceding)) {
      scalar = boosters[preceding] * (valence < 0 ? -1 : 1);
      if (capitalDifference && isUpper(original[index - distance - 1])) scalar += valence > 0 ? C_INCR : -C_INCR;
    }
    if (distance === 1 && scalar !== 0) scalar *= 0.95;
    if (distance === 2 && scalar !== 0) scalar *= 0.9;
    valence += scalar;
    valence = negationCheck(valence, words, distance, index);
    if (distance === 2) valence = idiomsCheck(valence, words, index);
  }
  if (index > 0 && words[index - 1] === "least" && !has(lexicon, words[index - 1])) {
    if (index === 1 || !["at", "very"].includes(words[index - 2])) valence *= N_SCALAR;
  }
  return valence;
}
function scoreVaderSentiment(input) {
  const text = expandEmoji(input);
  const original = tokens(text);
  const words = original.map((word) => word.toLowerCase());
  const capitalCount = original.filter(isUpper).length;
  const capitalDifference = capitalCount > 0 && capitalCount < original.length;
  const sentiments = words.map((_, index) => wordValence(original, words, index, capitalDifference));
  if (!sentiments.length) return { neg: 0, neu: 0, pos: 0, compound: 0 };
  const but = words.indexOf("but");
  if (but !== -1) {
    for (let pass = 0; pass < sentiments.length; pass += 1) {
      const value = sentiments[pass];
      const index = sentiments.indexOf(value);
      if (index < but) sentiments[index] = value * 0.5;
      else if (index > but) sentiments[index] = value * 1.5;
    }
  }
  const questionCount = (text.match(/\?/gu) ?? []).length;
  const punctuation = Math.min((text.match(/!/gu) ?? []).length, 4) * 0.292 + (questionCount > 3 ? 0.96 : questionCount > 1 ? questionCount * 0.18 : 0);
  let sum = sentiments.reduce((total2, value) => total2 + value, 0);
  if (sum > 0) sum += punctuation;
  else if (sum < 0) sum -= punctuation;
  const compound = sum / Math.sqrt(sum * sum + 15);
  let positive = 0;
  let negative = 0;
  let neutral = 0;
  for (const value of sentiments) {
    if (value > 0) positive += value + 1;
    else if (value < 0) negative += value - 1;
    else neutral += 1;
  }
  if (positive > Math.abs(negative)) positive += punctuation;
  else if (positive < Math.abs(negative)) negative -= punctuation;
  const total = positive + Math.abs(negative) + neutral;
  return {
    neg: round(Math.abs(negative / total), 3),
    neu: round(neutral / total, 3),
    pos: round(positive / total, 3),
    compound: round(compound, 4)
  };
}

// web/lib/trajectory-featurizer.ts
var MOVE_DELTAS = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 }
};
function samePoint(left, right) {
  return left.x === right.x && left.y === right.y;
}
function recordTrajectoryStep(stateBefore, stateAfter, aiAction, humanAction, reward = stateAfter.score - stateBefore.score) {
  return { stateBefore, stateAfter, aiAction, humanAction, reward };
}
function featurizeTrajectorySteps(steps) {
  const counts = {};
  const indicators = /* @__PURE__ */ new Set();
  const add = (feature, value = 1) => {
    counts[feature] = (counts[feature] ?? 0) + value;
  };
  const indicate = (feature) => {
    if (indicators.has(feature)) return;
    indicators.add(feature);
    add(feature);
  };
  for (const step of steps) {
    if (step.aiAction === "stay") add("time_cost");
    const heldBefore = step.stateBefore.partner.held;
    const heldAfter = step.stateAfter.partner.held;
    if (heldAfter !== heldBefore) {
      if (heldAfter === "tomato") {
        add("ingredient_tomato");
        add("pick_tomato");
      } else if (heldAfter === "onion") {
        add("ingredient_onion");
        add("pick_onion");
      } else if (heldAfter === "dish") {
        add("pick_dish");
      } else if (heldAfter === "soup") {
        add("pick_ready_soup");
      }
      if (step.aiAction === "interact" && heldAfter === null) {
        if (heldBefore === "tomato") add("adds_needed_tomato");
        else if (heldBefore === "onion") add("adds_needed_onion");
        else if (heldBefore === "soup" && step.reward > 0) add("serve_ready_soup");
      }
    }
    if (step.stateAfter.pot.stage === "empty") indicate("pot_empty");
    if (step.stateAfter.pot.stage === "cooking") indicate("pot_cooking");
    if (step.stateAfter.pot.stage === "ready") {
      indicate("soup_ready");
      if (heldAfter !== "dish") indicate("dish_needed_for_ready_soup");
    }
    const humanDelta = MOVE_DELTAS[step.humanAction];
    if (humanDelta) {
      const humanTarget = {
        x: step.stateBefore.player.x + humanDelta.x,
        y: step.stateBefore.player.y + humanDelta.y
      };
      if (samePoint(step.stateAfter.player, step.stateBefore.player) && samePoint(step.stateBefore.partner, humanTarget)) {
        add("blocks_human_path");
        add("human_wait_cost");
      }
      if (samePoint(step.stateAfter.partner, humanTarget) && !samePoint(step.stateBefore.partner, humanTarget)) {
        add("cuts_in_front_of_human");
        add("collision_risk");
      }
    }
  }
  return counts;
}

// tmp/pragmatic-negative-audit.ts
var initial = createGameState("running");
var pickupBefore = { ...initial, partner: { ...initial.partner, x: 4, y: 4, facing: "down" } };
var pickupAfter = stepGame(pickupBefore, "interact", "stay");
if (pickupAfter.partner.held !== "onion") throw new Error("Real onion pickup failed");
var trajectoryFeatures = featurizeTrajectorySteps([recordTrajectoryStep(pickupBefore, pickupAfter, "interact", "stay")]);
var prior = () => createFullGaussianPrior([...REWARD_FEATURES]);
var weights = (mean) => Object.fromEntries(REWARD_FEATURES.map((feature, index) => [feature, mean[index]]));
var decision = (state, mean) => {
  const result = chooseAiDecision(state, mean ? weights(mean) : void 0);
  const actual = stepGame(state, result.action, "stay");
  const scores = Object.fromEntries(result.ranking.map((x) => [x.subgoal, x.score]));
  return {
    chosenSubgoal: result.chosenSubgoal,
    action: result.action,
    scores,
    onionMinusTomato: "GET_ONION" in scores && "GET_TOMATO" in scores ? scores.GET_ONION - scores.GET_TOMATO : null,
    actualPartnerAfter: actual.partner,
    actualEvent: actual.lastEventCode,
    ranking: result.ranking
  };
};
var snapshots = [
  ["empty_pot_tomato_station", { ...initial, partner: { ...initial.partner, x: 1, y: 4, facing: "left" } }],
  ["empty_pot_onion_station", pickupBefore],
  ["empty_pot_initial_position", initial],
  ["cooking_next_order_prefetch", { ...initial, partner: { ...initial.partner, x: 1, y: 4, facing: "left" }, pot: { stage: "cooking", tomatoes: 2, onions: 1, secondsRemaining: 15 } }]
];
var inputs = [
  { text: "That onion pickup was good.", label: "trajectory" },
  { text: "That onion pickup was bad.", label: "trajectory" },
  { text: "That onion pickup was terrible.", label: "trajectory" },
  { text: "Don't pick onions.", label: "action" }
];
var priors = [
  { id: "zero_prior", make: prior },
  { id: "synthetic_existing_progress_reward", make: () => {
    const p = prior();
    p.mean[REWARD_FEATURES.indexOf("moves_toward_needed_object")] = 30;
    return p;
  } }
];
var records = priors.flatMap(({ id: priorId, make }) => snapshots.flatMap(([snapshotId, state]) => inputs.map(({ text, label }) => {
  const grounding = groundRoute1Feedback({ text, grounding: { label, confidence: 0.99, modelHash: "fixed-correct-reference-for-updater-isolation" }, state, trajectoryFeatures });
  const sentiment = scoreVaderSentiment(text);
  const result = applyGroundedPragmaticFeedback(make(), { grounding, sentiment: sentiment.compound });
  const legacy = applyPragmaticRoute1Update(make(), { feedbackForm: label === "action" ? "imperative" : "evaluative", feedbackFormConfidence: 0.99, trajectoryFeatures: grounding.targetFeatures, actionFeatures: grounding.targetFeatures, sentiment: sentiment.compound });
  return {
    priorId,
    snapshotId,
    text,
    label,
    sentiment,
    grounding,
    status: result.status,
    effectiveValence: result.effectiveValence,
    valenceSource: result.valenceSource,
    before: decision(state, make().mean),
    literalOnly: decision(state, result.literal.state.mean),
    after: decision(state, result.state.mean),
    legacyGlobalComplement: decision(state, legacy.state.mean),
    delta: result.delta
  };
})));
var sourceFiles = ["web/lib/route1-grounding.ts", "web/lib/pragmatic-route1.ts", "web/lib/browser-models.ts", "web/lib/game.ts", "web/lib/subgoal-policy.ts", "web/lib/trajectory-featurizer.ts", "web/lib/vader-sentiment.ts", "web/lib/vendor/vader-data.json"];
var report = { schema: "pragmatic-negative-behavior-audit-v1", dataScope: "Synthetic only. Actual onion pickup transition replayed, then same frozen observed features paired with constructed feasible decision snapshots. Grounding label is fixed correct to isolate updater; this is not classifier evaluation. Existing-progress prior is an explicit synthetic Gaussian fixture, not a claimed trained posterior. legacyGlobalComplement uses original low-level API, so its prohibition row intentionally does not include the separate live prohibition interpretation.", priorDefinitions: Object.fromEntries(priors.map((p) => [p.id, p.make()])), sourceHashes: Object.fromEntries(sourceFiles.map((p) => [p, (0, import_node_crypto.createHash)("sha256").update((0, import_node_fs.readFileSync)(p)).digest("hex")])), pickupBefore: pickupBefore.partner, pickupAfter: pickupAfter.partner, trajectoryFeatures, records };
(0, import_node_fs.mkdirSync)("artifacts/pragmatic-negative-audit-20260906", { recursive: true });
(0, import_node_fs.writeFileSync)("artifacts/pragmatic-negative-audit-20260906/report-after-full53.json", JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ trajectoryFeatures, results: records.filter((r) => r.snapshotId === "cooking_next_order_prefetch").map((r) => ({ prior: r.priorId, snapshot: r.snapshotId, text: r.text, status: r.status, valence: r.effectiveValence, before: r.before.chosenSubgoal, literal: r.literalOnly.chosenSubgoal, after: r.after.chosenSubgoal, legacy: r.legacyGlobalComplement.chosenSubgoal, beforeMargin: r.before.onionMinusTomato, literalMargin: r.literalOnly.onionMinusTomato, afterMargin: r.after.onionMinusTomato, legacyMargin: r.legacyGlobalComplement.onionMinusTomato, action: r.after.action })) }, null, 2));
