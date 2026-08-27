import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import {
  applyRoute1PaperUpdate,
  applyRoute2Gaussian,
  classifierVariantFromSearch,
  createBrowserModels,
  createFullGaussianPrior,
  createIndependentGaussianPrior,
  loadBrowserModelsFromManifest,
  route2Tokenize,
} from "../lib/browser-models";

const root = join(import.meta.dirname, "..");
const legacyFeedbackArtifact = JSON.parse(
  readFileSync(join(root, "public", "models", "feedback-form-v3.json"), "utf8"),
);
const shadowFeedbackArtifact = JSON.parse(
  readFileSync(
    join(
      root,
      "public",
      "models",
      "feedback-form-boundary-shadow-v1-580ab91c21d988d4ea7e146b70e810b68d0e454372b5cf3cb0ea32abb7f81dc3.json",
    ),
    "utf8",
  ),
);
const route2Artifact = JSON.parse(
  readFileSync(join(root, "public", "models", "route2-v5.json"), "utf8"),
);
const legacyModels = createBrowserModels(legacyFeedbackArtifact, route2Artifact);
const models = createBrowserModels(shadowFeedbackArtifact, route2Artifact);

describe("browser model release manifest", () => {
  const manifestText = readFileSync(
    join(root, "public", "models", "manifest.json"),
    "utf8",
  );
  const shadowManifestText = readFileSync(
    join(root, "public", "models", "manifest-boundary-shadow-v1.json"),
    "utf8",
  );
  const legacyFeedbackBytes = readFileSync(
    join(root, "public", "models", "feedback-form-v3.json"),
  );
  const shadowFeedbackBytes = readFileSync(
    join(
      root,
      "public",
      "models",
      "feedback-form-boundary-shadow-v1-580ab91c21d988d4ea7e146b70e810b68d0e454372b5cf3cb0ea32abb7f81dc3.json",
    ),
  );
  const route2Bytes = readFileSync(join(root, "public", "models", "route2-v5.json"));

  function releaseFetcher(manifest = manifestText) {
    return async (input: RequestInfo | URL): Promise<Response> => {
      const path = String(input);
      if (path === "/models/manifest.json") return new Response(manifest);
      if (path === "/models/manifest-boundary-shadow-v1.json") {
        return new Response(shadowManifestText);
      }
      if (path === "/models/feedback-form-v3.json") {
        return new Response(new Uint8Array(legacyFeedbackBytes));
      }
      if (
        path ===
        "/models/feedback-form-boundary-shadow-v1-580ab91c21d988d4ea7e146b70e810b68d0e454372b5cf3cb0ea32abb7f81dc3.json"
      ) {
        return new Response(new Uint8Array(shadowFeedbackBytes));
      }
      if (path === "/models/route2-v5.json") {
        return new Response(new Uint8Array(route2Bytes));
      }
      return new Response("not found", { status: 404 });
    };
  }

  it("loads the production classifier by default after SHA-256 verification", async () => {
    expect(createHash("sha256").update(manifestText).digest("hex")).toBe(
      "ecb077b810cab4fb1949a25c948d65e5878a51164743cd3cc0c1986b70826f76",
    );
    const released = await loadBrowserModelsFromManifest("", releaseFetcher());
    expect(released.features).toHaveLength(53);
    expect(released.classify("Please take a dish.").modelHash).toBe(
      "3ce488daf88afb611fa5923ed21407ea344fbbb79c10fbc8ab3c39bc174d6e31",
    );
    expect(released.classifierMode).toBe("production");
  });

  it("loads the explicit shadow preview without changing the default channel", async () => {
    const released = await loadBrowserModelsFromManifest(
      "",
      releaseFetcher(),
      "boundary-shadow-v1",
    );
    expect(released.classify("you should not wait").modelHash).toBe(
      "40b03b2f9a88a71f7f5334d61f77176bcde01b86b7f8f277422c3038be4e1a55",
    );
    expect(released.classifierMode).toBe("experimental-shadow-preview");
  });

  it("rejects an artifact whose bytes do not match the release manifest", async () => {
    const fetcher = releaseFetcher();
    await expect(
      loadBrowserModelsFromManifest("", async (input) => {
        if (String(input) === "/models/feedback-form-v3.json") {
          return new Response("tampered");
        }
        return fetcher(input);
      }),
    ).rejects.toThrow("feedback-form model hash mismatch");
  });

  it("does not fall back when the explicit shadow manifest is missing", async () => {
    await expect(
      loadBrowserModelsFromManifest(
        "",
        async (input) => {
          if (String(input) === "/models/manifest-boundary-shadow-v1.json") {
            return new Response("not found", { status: 404 });
          }
          return releaseFetcher()(input);
        },
        "boundary-shadow-v1",
      ),
    ).rejects.toThrow("failed to load browser model manifest");
  });

  it("selects the shadow only for the exact opt-in query parameter", () => {
    expect(classifierVariantFromSearch("?classifier=boundary-shadow-v1")).toBe(
      "boundary-shadow-v1",
    );
    expect(classifierVariantFromSearch("")).toBe("production");
    expect(() => classifierVariantFromSearch("?classifier=unknown")).toThrow(
      "unsupported classifier query parameter",
    );
    expect(() =>
      classifierVariantFromSearch(
        "?classifier=boundary-shadow-v1&classifier=boundary-shadow-v1",
      ),
    ).toThrow("must appear exactly once");
  });
});

describe("rollback feedback-form v3 model parity", () => {
  const gold: Record<string, Record<string, number>> = {
    "That last route was bad.": {
      descriptive: 0.0007616129387273556,
      evaluative: 0.9987120181662493,
      imperative: 0.0005263688950233705,
    },
    "Please take a dish instead.": {
      descriptive: 0.18208058072521213,
      evaluative: 0.05712194603207744,
      imperative: 0.7607974732427105,
    },
    "There is a spare dish near the stove.": {
      descriptive: 0.45022840947515697,
      evaluative: 0.041866647848330656,
      imperative: 0.5079049426765124,
    },
    "What is the score?": {
      descriptive: 0.15589638941591147,
      evaluative: 0.7734005559401227,
      imperative: 0.07070305464396585,
    },
  };

  for (const [text, probabilities] of Object.entries(gold)) {
    it(`matches sklearn for ${text}`, () => {
      const prediction = legacyModels.classify(text);
      for (const [label, expected] of Object.entries(probabilities)) {
        expect(prediction.probabilities[label as keyof typeof prediction.probabilities]).toBeCloseTo(
          expected,
          9,
        );
      }
      expect(prediction.modelHash).toBe(
        "3ce488daf88afb611fa5923ed21407ea344fbbb79c10fbc8ab3c39bc174d6e31",
      );
    });
  }

  it("keeps the low-score top class but abstains", () => {
    const prediction = legacyModels.classify("There is a spare dish near the stove.");
    expect(prediction.label).toBe("imperative");
    expect(prediction.abstained).toBe(true);
    expect(prediction.threshold).toBe(0.55);
  });
});

describe("boundary shadow preview model parity", () => {
  const gold: Record<string, Record<string, number>> = {
    "you should not wait": {
      descriptive: 9.929842301860547e-18,
      evaluative: 4.331421478968129e-14,
      imperative: 0.9999999999999567,
    },
    "you need to pick up the tomato closest to you.": {
      descriptive: 2.6274986168347226e-19,
      evaluative: 2.6181990532796743e-12,
      imperative: 0.9999999999973819,
    },
    "There is a spare dish near the stove.": {
      descriptive: 0.9999329295030418,
      evaluative: 2.3639187171761516e-10,
      imperative: 6.707026056616814e-5,
    },
    "should not block my way": {
      descriptive: 1.0542149527233571e-20,
      evaluative: 4.228884786855795e-21,
      imperative: 1.0,
    },
  };

  for (const [text, probabilities] of Object.entries(gold)) {
    it(`matches shadow sklearn for ${text}`, () => {
      const prediction = models.classify(text);
      for (const [label, expected] of Object.entries(probabilities)) {
        expect(
          prediction.probabilities[label as keyof typeof prediction.probabilities],
        ).toBeCloseTo(expected, 9);
      }
      expect(prediction.modelHash).toBe(
        "40b03b2f9a88a71f7f5334d61f77176bcde01b86b7f8f277422c3038be4e1a55",
      );
      expect(prediction.scoreKind).toBe(
        "temperature_scaled_train_oof_probability",
      );
      expect(prediction.temperatureScaled).toBe(true);
      expect(prediction.independentlyCalibrated).toBe(false);
      expect(prediction.calibrated).toBe(false);
    });
  }

  it("classifies all three acceptance phrases as imperative", () => {
    expect(models.classify("you should not wait").label).toBe("imperative");
    expect(
      models.classify("you need to pick up the tomato closest to you.").label,
    ).toBe("imperative");
    expect(models.classify("should not block my way").label).toBe("imperative");
  });

  it("replays the frozen temperature, class order, diagnostic threshold, and FeatureUnion weight", () => {
    expect(shadowFeedbackArtifact.classes).toEqual([
      "descriptive",
      "evaluative",
      "imperative",
    ]);
    expect(shadowFeedbackArtifact.calibration.temperature).toBe(
      0.0823618560384671,
    );
    expect(shadowFeedbackArtifact.minimum_confidence).toBe(0.55);
    expect(shadowFeedbackArtifact.transformers[0].weight ?? 1).toBe(1);
    expect(shadowFeedbackArtifact.transformers[1].weight).toBe(0.7);
  });

  it("fails closed when the class order is changed", () => {
    const mutated = structuredClone(shadowFeedbackArtifact);
    mutated.classes = ["evaluative", "descriptive", "imperative"];
    expect(() => createBrowserModels(mutated, route2Artifact)).toThrow(
      "class order changed",
    );
  });
});

describe("browser Route2 model parity", () => {
  it("uses the paper-compatible noun tokenizer", () => {
    expect(route2Tokenize("Please take dishes, instead.")).toEqual([
      "please",
      "take",
      "dish",
      "instead",
    ]);
  });

  it("matches all 53 Python ensemble outputs", () => {
    const expected: Record<string, number> = {
      adds_extra_onion: -1.9229856371879577,
      adds_extra_tomato: -1.9079070568084717,
      adds_needed_onion: 1.8969232082366942,
      adds_needed_tomato: 1.898507046699524,
      avoids_duplicate_human_task: 2.850256931781769,
      avoids_human_shortest_path: 1.692656084895134,
      blocks_human_path: -2.532379126548767,
      blocks_partner_on_ring: -1.6736770898103714,
      blocks_serving_route: -2.529458302259445,
      breaks_recipe: -2.857763147354126,
      clears_human_path: 1.691334919631481,
      clears_human_shortest_path: 1.700957891345024,
      clears_serving_access: 1.6908294677734375,
      collision_risk: -1.6720532327890396,
      complementary_to_human: 3.554472804069519,
      completes_recipe: 2.8561413049697877,
      crowds_human_target: -2.913862204551697,
      cuts_in_front_of_human: -2.1203154385089875,
      delays_serving: -1.8699128389358521,
      dish_needed_for_ready_soup: 1.4161182522773743,
      distance_cost: -0.005774738790569245,
      duplicate_human_task: -3.557748818397522,
      frustrates_human: -2.902908682823181,
      human_wait_cost: -1.6933645576238632,
      ingredient_onion: -0.15319537073373796,
      ingredient_tomato: -0.1550318341702223,
      matches_current_order: 1.4258131861686707,
      moves_away_from_needed_object: -1.2291895031929017,
      moves_toward_needed_object: -0.785537774860859,
      near_dish_dispenser: 0.0004923107924696523,
      near_onion_dispenser: -0.004434481092903298,
      near_pot: -0.015463758274563588,
      near_serving_counter: -0.0003854137670714408,
      near_tomato_dispenser: 0.0006177791106892983,
      pick_dish: -0.4050505466759205,
      pick_onion: -0.2392043974250555,
      pick_ready_soup: 0.9682601392269135,
      pick_tomato: -0.23244672566652297,
      pot_cooking: 0.004273988170461962,
      pot_empty: 0.007477770690275065,
      pot_has_one_tomato: -0.0049985350441829725,
      pot_has_two_tomatoes: 0.0021348486492115625,
      pot_has_two_tomatoes_one_onion: 0.006655969098210335,
      recipe_needs_first_tomato: -0.003981267347262474,
      recipe_needs_onion: 0.9518010795116425,
      recipe_needs_second_tomato: -0.0009595436647941824,
      respects_human_intent: 2.8940253496170043,
      serve_ready_soup: 2.87085599899292,
      soup_ready: 0.0022272007202445822,
      steals_human_target: -3.6308570384979246,
      supports_serving: 2.4827208161354064,
      time_cost: -0.4923825293779373,
      wrong_ingredient: -1.9088760972023011,
    };
    const prediction = models.route2(
      "Please take a dish instead.",
      Array.from({ length: 53 }, () => 0),
    );
    expect(Object.keys(prediction.weights)).toHaveLength(53);
    for (const [feature, value] of Object.entries(expected)) {
      expect(Math.abs(prediction.weights[feature] - value)).toBeLessThan(1e-5);
    }
    expect(prediction.ensembleSize).toBe(10);
  });
});

describe("paper Bayesian updates", () => {
  it("Route2 updates every independent dimension with precision two and no f_G", () => {
    const prior = createIndependentGaussianPrior(["a", "b"]);
    const result = applyRoute2Gaussian(prior, { a: 3, b: -1 });
    expect(result.state.precision).toEqual([2.04, 2.04]);
    expect(result.state.mean[0]).toBeCloseTo(6 / 2.04, 12);
    expect(result.state.mean[1]).toBeCloseTo(-2 / 2.04, 12);
  });

  it("Route1 lets f_G own grounding and uses the rank-one paper multiply", () => {
    const prior = createFullGaussianPrior(["a", "b"]);
    const result = applyRoute1PaperUpdate(prior, {
      feedbackForm: "evaluative",
      feedbackFormConfidence: 1,
      requestedReferenceSubtype: "feature",
      trajectoryFeatures: { a: 1 },
      namedFeatures: { b: 1 },
      valence: 30,
    });
    expect(result.status).toBe("updated");
    expect(result.referenceConflict).toBe(true);
    expect(result.effectiveReferenceSubtype).toBe("trajectory");
    expect(result.targetFeatures).toEqual({ a: 1 });
    expect(result.state.mean[0]).toBeCloseTo(1500 / 51, 12);
    expect(result.state.mean[1]).toBe(0);
    expect(result.state.covariance[0][0]).toBeCloseTo(25 - 1250 / 51, 12);
  });

  it("Route1 rejects an abstained coarse form before changing weights", () => {
    const prior = createFullGaussianPrior(["a"]);
    const result = applyRoute1PaperUpdate(prior, {
      feedbackForm: "descriptive",
      feedbackFormConfidence: 0.4,
      feedbackFormAbstained: true,
      namedFeatures: { a: 1 },
      valence: -30,
    });
    expect(result.status).toBe("rejected");
    expect(result.state).toBe(prior);
    expect(result.delta).toEqual({});
  });
});
