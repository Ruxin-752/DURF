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
      "feedback-form-boundary-shadow-raw-v2-2f1e1f7ed8836680c4eebd6e6b89f28e565ef99f1dbbf1c6ed03023903719fef.json",
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
    join(root, "public", "models", "manifest-boundary-shadow-raw-v2.json"),
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
      "feedback-form-boundary-shadow-raw-v2-2f1e1f7ed8836680c4eebd6e6b89f28e565ef99f1dbbf1c6ed03023903719fef.json",
    ),
  );
  const route2Bytes = readFileSync(join(root, "public", "models", "route2-v5.json"));

  function releaseFetcher(manifest = manifestText) {
    return async (input: RequestInfo | URL): Promise<Response> => {
      const path = String(input);
      if (path === "/models/manifest.json") return new Response(manifest);
      if (path === "/models/manifest-boundary-shadow-raw-v2.json") {
        return new Response(shadowManifestText);
      }
      if (path === "/models/feedback-form-v3.json") {
        return new Response(new Uint8Array(legacyFeedbackBytes));
      }
      if (
        path ===
        "/models/feedback-form-boundary-shadow-raw-v2-2f1e1f7ed8836680c4eebd6e6b89f28e565ef99f1dbbf1c6ed03023903719fef.json"
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
      "40f5d51cd2fdcd6cb2c8f5dacf691b9f097a212b7a6a5373a2f3fb50237b0e1e",
    );
    const released = await loadBrowserModelsFromManifest("", releaseFetcher());
    expect(released.features).toHaveLength(53);
    expect(released.classify("Please take a dish.").modelHash).toBe(
      "3ce488daf88afb611fa5923ed21407ea344fbbb79c10fbc8ab3c39bc174d6e31",
    );
    expect(released.classifierMode).toBe("production");
  });

  it("loads the explicit shadow preview without changing the default channel", async () => {
    expect(createHash("sha256").update(shadowManifestText).digest("hex")).toBe(
      "9bd1f9411ab0cd16fe056d67dce00d15b4272a05b5acfa2f01251cc9e7be7e03",
    );
    const released = await loadBrowserModelsFromManifest(
      "",
      releaseFetcher(),
      "boundary-shadow-raw-v2",
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

  it("rejects a default manifest whose release status or promotion flag changes", async () => {
    const mutatedManifest = JSON.parse(manifestText);
    mutatedManifest.release.promotion_eligible = true;
    const fetcher = releaseFetcher();
    await expect(
      loadBrowserModelsFromManifest("", async (input) => {
        if (String(input) === "/models/manifest.json") {
          return new Response(JSON.stringify(mutatedManifest));
        }
        return fetcher(input);
      }),
    ).rejects.toThrow("release status or promotion contract changed");
  });

  it("does not fall back when the explicit shadow manifest is missing", async () => {
    await expect(
      loadBrowserModelsFromManifest(
        "",
        async (input) => {
          if (String(input) === "/models/manifest-boundary-shadow-raw-v2.json") {
            return new Response("not found", { status: 404 });
          }
          return releaseFetcher()(input);
        },
        "boundary-shadow-raw-v2",
      ),
    ).rejects.toThrow("failed to load browser model manifest");
  });

  it("selects the shadow only for the exact opt-in query parameter", () => {
    expect(classifierVariantFromSearch("?classifier=boundary-shadow-raw-v2")).toBe(
      "boundary-shadow-raw-v2",
    );
    expect(classifierVariantFromSearch("")).toBe("production");
    expect(() => classifierVariantFromSearch("?classifier=unknown")).toThrow(
      "unsupported classifier query parameter",
    );
    expect(() =>
      classifierVariantFromSearch(
        "?classifier=boundary-shadow-raw-v2&classifier=boundary-shadow-raw-v2",
      ),
    ).toThrow("must appear exactly once");
    expect(() => classifierVariantFromSearch("?classifier=boundary-shadow-v1")).toThrow(
      "unsupported classifier query parameter",
    );
  });
});

describe("rollback feedback-form v3 model parity", () => {
  it("binds every active report and release claim to the active source model", () => {
    const card = legacyFeedbackArtifact.model_card;
    expect(card.release).toMatchObject({
      status: "active_legacy_default",
      default_eligible: true,
      promotion_eligible: false,
      promotion_status: "legacy_active_not_requalified",
    });
    expect(card.claims.independent_current_player_accuracy).toBeNull();
    expect(card.reports.map((report: { id: string }) => report.id)).toEqual([
      "training",
      "paper_human_reference_proxy",
    ]);
    expect(
      card.reports.every(
        (report: { source_model_sha256: string }) =>
          report.source_model_sha256 === card.source_model_sha256,
      ),
    ).toBe(true);
  });

  it("exposes model-bound evidence without claiming current-player accuracy", () => {
    expect(legacyModels.classifierEvidence).toMatchObject({
      trained: true,
      frozen: true,
      modelHash: "3ce488daf88afb611fa5923ed21407ea344fbbb79c10fbc8ab3c39bc174d6e31",
      independentCurrentPlayerAccuracy: null,
      diagnosticAccuracy: 0.9166666666666666,
      diagnosticRows: 96,
      promotionEligible: false,
      promotionReason: "No independent current-player evaluation is bound to this model.",
      diagnosticPreviouslyExposed: true,
    });
  });

  const gold: Record<string, Record<string, number>> = {
    "That last route was bad.": {
      descriptive: 0.0007616129387273556,
      evaluative: 0.9987120181662493,
      imperative: 0.0005263688950233713,
    },
    "Please take a dish instead.": {
      descriptive: 0.1820805807252124,
      evaluative: 0.05712194603207763,
      imperative: 0.76079747324271,
    },
    "There is a spare dish near the stove.": {
      descriptive: 0.45022840947515697,
      evaluative: 0.04186664784833057,
      imperative: 0.5079049426765124,
    },
    "What is the score?": {
      descriptive: 0.1558963894159114,
      evaluative: 0.7734005559401227,
      imperative: 0.07070305464396585,
    },
  };

  for (const [text, probabilities] of Object.entries(gold)) {
    it(`matches sklearn for ${text}`, () => {
      const prediction = legacyModels.classify(text);
      for (const [label, expected] of Object.entries(probabilities)) {
        const actual = prediction.probabilities[
          label as keyof typeof prediction.probabilities
        ];
        expect(Object.is(actual, expected)).toBe(true);
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
  it("binds the failed one-shot diagnostic to the frozen shadow release", () => {
    expect(models.classifierEvidence).toMatchObject({
      modelHash: "40b03b2f9a88a71f7f5334d61f77176bcde01b86b7f8f277422c3038be4e1a55",
      independentCurrentPlayerAccuracy: null,
      diagnosticAccuracy: 0.8194444444444444,
      diagnosticRows: 72,
      requestedAccuracyTarget: 0.87,
      targetPassed: false,
      promotionEligible: false,
      promotionReason: "The requested 87% target was not met.",
      diagnosticPreviouslyExposed: false,
    });
  });

  const gold: Record<string, Record<string, number>> = {
    "you should not wait": {
      descriptive: 0.03554112555658123,
      evaluative: 0.07087656218712816,
      imperative: 0.8935823122562906,
    },
    "you need to pick up the tomato closest to you.": {
      descriptive: 0.025853134087664525,
      evaluative: 0.09748086343854041,
      imperative: 0.876666002473795,
    },
    "There is a spare dish near the stove.": {
      descriptive: 0.6194637132574359,
      evaluative: 0.09980970676859893,
      imperative: 0.28072657997396516,
    },
    "you should not block my way": {
      descriptive: 0.014481074279711606,
      evaluative: 0.028159406971222996,
      imperative: 0.9573595187490653,
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
      expect(prediction.scoreKind).toBe("raw_model_softmax_score");
      expect(prediction.temperatureScaled).toBe(false);
      expect(prediction.independentlyCalibrated).toBe(false);
      expect(prediction.calibrated).toBe(false);
      expect(prediction.calibrationVersion).toBeNull();
    });
  }

  it("classifies all three acceptance phrases as imperative", () => {
    expect(models.classify("you should not wait").label).toBe("imperative");
    expect(
      models.classify("you need to pick up the tomato closest to you.").label,
    ).toBe("imperative");
    expect(models.classify("you should not block my way").label).toBe("imperative");
  });

  it("uses the raw top score for the routing-policy rejection", () => {
    const prediction = models.classify("maybe");
    expect(prediction.confidence).toBeCloseTo(0.43225123, 7);
    expect(prediction.abstained).toBe(true);
    const prior = createFullGaussianPrior(["a"]);
    const result = applyRoute1PaperUpdate(prior, {
      feedbackForm: prediction.label,
      feedbackFormConfidence: prediction.confidence,
      feedbackFormThreshold: prediction.threshold,
      feedbackFormAbstained: prediction.abstained,
      actionFeatures: { a: 1 },
      valence: 1,
    });
    expect(result.status).toBe("rejected");
    expect(result.reason).toBe("feedback_form_low_confidence");
    expect(result.state).toBe(prior);
    expect(result.delta).toEqual({});
  });

  it("uses the paper's fixed precision after the classifier gate", () => {
    const prediction = models.classify("you should not wait");
    const result = applyRoute1PaperUpdate(createFullGaussianPrior(["a"]), {
      feedbackForm: prediction.label,
      feedbackFormConfidence: prediction.confidence,
      feedbackFormThreshold: prediction.threshold,
      feedbackFormAbstained: prediction.abstained,
      actionFeatures: { a: 1 },
      valence: 1,
    });
    expect(result.status).toBe("updated");
    expect(result.effectivePrecision).toBe(2);
  });

  it("replays raw-score policy, class order, routing threshold, and FeatureUnion weight", () => {
    expect(shadowFeedbackArtifact.classes).toEqual([
      "descriptive",
      "evaluative",
      "imperative",
    ]);
    expect(shadowFeedbackArtifact.calibration).toEqual({
      independently_validated: false,
      method: "none",
      scope: "raw_model_output",
    });
    expect(shadowFeedbackArtifact.score_policy).toEqual({
      independently_calibrated: false,
      kind: "raw_softmax",
      probability_of_correctness: false,
      routing_threshold: 0.55,
      temperature: 1,
      threshold_policy: "existing_web_preview_policy_not_validated_in_raw_score_space",
      version: "boundary-shadow-raw-softmax-v2",
    });
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

  it("fails closed when raw-score or threshold-policy claims change", () => {
    const scoreMutation = structuredClone(shadowFeedbackArtifact);
    scoreMutation.claim_scope.displayed_score_is_probability_of_correctness = true;
    expect(() => createBrowserModels(scoreMutation, route2Artifact)).toThrow(
      "shadow classifier release contract changed",
    );
    const thresholdMutation = structuredClone(shadowFeedbackArtifact);
    thresholdMutation.score_policy.threshold_policy = "validated";
    expect(() => createBrowserModels(thresholdMutation, route2Artifact)).toThrow(
      "shadow classifier release contract changed",
    );
  });

  it("fails closed when model-card report SHA binding or promotion changes", () => {
    const reportMutation = structuredClone(shadowFeedbackArtifact);
    reportMutation.model_card.reports[1].source_model_sha256 = "0".repeat(64);
    expect(() => createBrowserModels(reportMutation, route2Artifact)).toThrow(
      "model-card report binding changed",
    );
    const promotionMutation = structuredClone(shadowFeedbackArtifact);
    promotionMutation.model_card.release.promotion_eligible = true;
    expect(() => createBrowserModels(promotionMutation, route2Artifact)).toThrow(
      "release status or promotion contract changed",
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
  it("does not let classifier score rescale an accepted Route1 observation", () => {
    const lowScore = applyRoute1PaperUpdate(createFullGaussianPrior(["a"]), {
      feedbackForm: "imperative",
      feedbackFormConfidence: 0.6,
      feedbackFormThreshold: 0.55,
      actionFeatures: { a: 1 },
      valence: 1,
    });
    const highScore = applyRoute1PaperUpdate(createFullGaussianPrior(["a"]), {
      feedbackForm: "imperative",
      feedbackFormConfidence: 0.99,
      feedbackFormThreshold: 0.55,
      actionFeatures: { a: 1 },
      valence: 1,
    });
    expect(lowScore.effectivePrecision).toBe(2);
    expect(lowScore.state.mean).toEqual(highScore.state.mean);
  });

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
