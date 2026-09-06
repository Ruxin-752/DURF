import { chooseAiDecision, type GameState } from './game';
import { POLICY_ACTIVE_FEATURES, REWARD_FEATURES, type Subgoal } from './subgoal-policy';

/** Independent reference prediction: never substitute the UI speech-act label. */
export interface Route1GroundingPrediction {
  label: 'action' | 'feature' | 'trajectory';
  confidence: number;
  threshold?: number;
  abstained?: boolean;
  modelHash?: string;
}

export interface GroundedRoute1Feedback {
  status: 'grounded' | 'rejected';
  reason?: string;
  label: Route1GroundingPrediction['label'];
  confidence: number;
  threshold: number;
  modelHash?: string;
  targetFeatures: Record<string, number>;
  /** Original Pragmatic rule: every unreferenced reward dimension is contrasted. */
  pragmaticAlternatives: Record<string, number>;
  selectedSubgoal: Subgoal | null;
  directivePolarity: 'affirm' | 'prohibit' | null;
  adaptation: 'paper-full-feature-complement-v1';
}

export interface GroundRoute1Input {
  text: string;
  grounding: Route1GroundingPrediction;
  state: GameState;
  trajectoryFeatures?: Record<string, number>;
}

const activeFeatures = new Set<string>(POLICY_ACTIVE_FEATURES);
const rewardFeatures = new Set<string>(REWARD_FEATURES);
export const ROUTE1_APPLICABILITY_VERSION = 'kitchen-reference-applicability-v1';

function isUnresolvedQuestion(text: string, label: Route1GroundingPrediction['label']): boolean {
  // A polite action request is resolvable by the action binder. Factual,
  // speculative, and rhetorical questions are not reward observations.
  if (label === 'action' && /^(?:please\s+)?(?:can|could|would|will) you\s+(?:please\s+)?(?:get|grab|fetch|take|pick|collect|bring|add|put|place|serve|deliver|scoop|plate|ladle|move|go|walk|head|wait|stay|yield|stop|avoid|stash)\b/u.test(text)) return false;
  return /\?/u.test(text) ||
    /^(?:who|what|when|where|why|how|is|are|was|were|am|does|did|can|could|would|will|should|shall|may|might|have|has|had)\b/u.test(text) ||
    /^do\s+(?!not\b)/u.test(text);
}

function trajectoryApplicability(text: string, features: Record<string, number>): string | null {
  const plain = text.replace(/[.!]+$/u, '').trim();
  // These complete constructions clearly appraise the immediately preceding
  // play. An adjective embedded in an unrelated sentence is not sufficient.
  const appraisal = '(?:good|great|nice|excellent|awesome|fantastic|terrific|perfect|bad|poor|terrible|awful|wrong|correct|helpful|unhelpful)';
  const modifier = '(?:(?:really|very|so|quite|not)\\s+)*';
  const generic = new RegExp(`^(?:${modifier}${appraisal}(?: (?:job|work|move|choice|decision|play|teamwork))?|well done|(?:you|we) did (?:well|great|badly)|(?:that|this|it) (?:was|is) ${modifier}${appraisal}|(?:that|this) (?:move|choice|decision|play) (?:was|is) ${modifier}${appraisal})$`, 'u');
  if (generic.test(plain)) return null;

  const kitchenReference = /\b(?:onions?|tomato(?:es)?|soups?|dishes|dish|plates?|bowls?|pots?|recipe|serv(?:e|ed|ing)|pickup|pick(?:ed|ing)?|cook(?:ed|ing)?|block(?:ed|ing)?|paths?|routes?|waiting|teamwork)\b/u.test(text);
  const appraisalWord = new RegExp(`\\b(?:${appraisal}|well|better|worse|efficient|wasteful|slow|useful|quick|fast|mistake)\\b`, 'u');
  const pastBehavior = /\b(?:you|we|the ai|the partner|my partner)\s+(?:(?:just|already|really|actually|have|had)\s+)*(?:picked|grabbed|fetched|took|got|added|put|placed|served|delivered|scooped|collected|waited|moved|blocked|cleared|wasted|helped|dropped|stashed)\b/u.test(text);
  const taskThanks = /^(?:thanks|thank you) for\b/u.test(text);
  if (!kitchenReference || (!appraisalWord.test(text) && !pastBehavior && !taskThanks)) {
    return 'unresolved_trajectory_reference';
  }
  // If the appraisal names a kitchen object, the observed window must contain
  // evidence of that object. Never invent a cited event from an unrelated turn.
  const objects: Array<[RegExp, string[]]> = [
    [/\bonions?\b/u, ['ingredient_onion', 'pick_onion', 'adds_needed_onion', 'adds_extra_onion']],
    [/\btomato(?:es)?\b/u, ['ingredient_tomato', 'pick_tomato', 'adds_needed_tomato', 'adds_extra_tomato']],
    [/\b(?:dish|dishes|plates?|bowls?)\b/u, ['pick_dish', 'dish_needed_for_ready_soup']],
    [/\bsoups?\b/u, ['pick_ready_soup', 'serve_ready_soup', 'soup_ready', 'supports_serving']],
  ];
  for (const [mention, matching] of objects) {
    if (mention.test(text) && !matching.some((feature) => (features[feature] ?? 0) > 0)) return 'trajectory_reference_not_observed';
  }
  return null;
}

function knownPositiveFeatures(source: Record<string, number>): Record<string, number> {
  return Object.fromEntries(Object.entries(source).filter(
    ([feature, value]) => rewardFeatures.has(feature) && Number.isFinite(value) && value > 0,
  ));
}

function actionReference(text: string, state: GameState): {
  subgoals: Subgoal[];
  polarity: 'affirm' | 'prohibit';
  reason?: string;
} {
  // Prohibitions are a directive property, independent of VADER sentiment.
  const prohibitions = text.match(/\b(?:do not|don't|dont|never|avoid|stop|must not|should not|shouldn't)\b/gu) ?? [];
  const prohibit = prohibitions.length > 0;
  const polarity = prohibit ? 'prohibit' : 'affirm';
  if (prohibitions.length > 1 || /\b(?:not not|don't not|do not not)\b/u.test(text)) {
    return { subgoals: [], polarity, reason: 'unsupported_action_negation' };
  }
  // One phrase must select one counterfactual; a list is not one observation.
  if (/\b(?:and|then|instead|rather|except|unless|or)\b/u.test(text)) {
    return { subgoals: [], polarity, reason: 'ambiguous_action_reference' };
  }
  if (/\b(?:not|n't)\b/u.test(text) && !prohibit) {
    return { subgoals: [], polarity, reason: 'unsupported_action_negation' };
  }
  const onion = /\bonions?\b/u.test(text);
  const tomato = /\btomato(?:es)?\b/u.test(text);
  const dish = /\b(?:dishes|dish|plates?|bowls?)\b/u.test(text);
  const soup = /\bsoups?\b/u.test(text);
  if ([onion, tomato, dish, soup].filter(Boolean).length > 1 && !(dish && soup && !onion && !tomato)) {
    return { subgoals: [], polarity, reason: 'ambiguous_action_reference' };
  }
  const subgoals: Subgoal[] = [];
  const put = /\b(?:add(?:ing)?|put(?:ting)?|drop(?:ping)?|place|placing|load(?:ing)?|insert(?:ing)?|toss(?:ing)?)\b/u.test(text);
  const acquire = /\b(?:get(?:ting)?|grab(?:bing)?|fetch(?:ing)?|take|taking|pick(?:ing)?(?: up)?|collect(?:ing)?|bring(?:ing)?)\b/u.test(text);
  const approach = /\b(?:go|going|move|moving|head(?:ing)?|walk(?:ing)?)\b/u.test(text);
  if (put && /\b(?:onto|into|on|in|at)\b/u.test(text) && !/\b(?:pot|stove|counter)\b/u.test(text)) {
    return { subgoals: [], polarity, reason: 'unresolved_action_reference' };
  }
  if (/\b(?:yield(?:ing)?|step(?:ping)? aside|move out of (?:my|the) way|clear(?:ing)? (?:my|the) path)\b/u.test(text)) {
    subgoals.push('YIELD_PATH');
  } else if (/\b(?:wait(?:ing)?|stay(?:ing)? (?:still|there|put)|hold still|pause)\b/u.test(text)) {
    subgoals.push('WAIT');
  } else if (/\b(?:stash(?:ing)?|set(?:ting)? down)\b/u.test(text) || (put && /\bcounter\b/u.test(text))) {
    subgoals.push('STASH_HELD_OBJECT');
  } else if (/\b(?:serve|serving|deliver(?:ing)?|hand in)\b/u.test(text)) {
    subgoals.push('SERVE_SOUP');
  } else if (/\b(?:scoop(?:ing)?|plate|plating|ladle|ladling)\b/u.test(text) || (soup && acquire)) {
    subgoals.push('PICKUP_SOUP');
  } else if (put || (approach && /\b(?:pot|stove)\b/u.test(text))) {
    const ingredient = onion ? 'onion' : tomato ? 'tomato' : state.partner.held;
    if (ingredient === 'onion') subgoals.push('PUT_ONION_IN_POT');
    if (ingredient === 'tomato') subgoals.push('PUT_TOMATO_IN_POT');
    if (ingredient === 'dish' && approach) subgoals.push('PICKUP_SOUP');
  } else if (acquire || approach) {
    if (onion) subgoals.push('GET_ONION');
    if (tomato) subgoals.push('GET_TOMATO');
    if (dish) subgoals.push('GET_DISH');
  }
  return { subgoals, polarity };
}

/** Named reward attributes, not a bag of every action associated with each noun. */
function featureReference(text: string): Record<string, number> {
  const features: Record<string, number> = {};
  if (/\bonions?\b/u.test(text)) features.ingredient_onion = 1;
  if (/\btomato(?:es)?\b/u.test(text)) features.ingredient_tomato = 1;
  if (/\b(?:block(?:s|ed|ing)?|obstruct(?:s|ed|ing)?)\b.*\b(?:path|route|way)\b/u.test(text)) features.blocks_human_path = 1;
  if (/\b(?:clear|open|free)\b.*\b(?:path|route|way)\b/u.test(text)) features.clears_human_path = 1;
  if (/\b(?:duplicate|duplicating|same)\b.*\b(?:task|work|job)\b/u.test(text)) features.duplicate_human_task = 1;
  if (/\b(?:delay|delays|delaying|slow)\b.*\bserv(?:e|ing|ice)\b/u.test(text)) features.delays_serving = 1;
  if (/\b(?:help(?:s|ful)?|support(?:s)?)\b.*\bserv(?:e|ing|ice)\b/u.test(text)) features.supports_serving = 1;
  if (/\b(?:wast(?:e|es|ing)|too much)\b.*\btime\b/u.test(text)) features.time_cost = 1;
  // Location and current-state assertions carry no preference over decisions.
  // Do not turn 'the onion dispenser is on the left' into a pick-onion reward.
  if (/\b(?:dispenser|window|counter|station)\b/u.test(text) ||
      /\b(?:pot|stove)\b.*\b(?:has|contains|empty|cooking|ready|full)\b/u.test(text) ||
      /\b(?:on (?:the )?(?:left|right)|next to|beside|near|north|south)\b/u.test(text)) return {};
  return features;
}

/**
 * Bind one independently classified reference to the game's 53-feature contract.
 * Actions use the exact live-policy candidate vector, including geometry and
 * feasibility. Historical praise uses only the supplied recent trajectory.
 */
export function groundRoute1Feedback(input: GroundRoute1Input): GroundedRoute1Feedback {
  const { grounding, state } = input;
  const threshold = grounding.threshold ?? 0.55;
  const result: GroundedRoute1Feedback = {
    status: 'rejected', label: grounding.label, confidence: grounding.confidence,
    threshold, modelHash: grounding.modelHash, targetFeatures: {},
    pragmaticAlternatives: {}, selectedSubgoal: null, directivePolarity: null,
    adaptation: 'paper-full-feature-complement-v1',
  };
  const reject = (reason: string): GroundedRoute1Feedback => ({ ...result, reason });
  if (grounding.abstained || !Number.isFinite(threshold) || threshold < 0 || threshold > 1 ||
      !Number.isFinite(grounding.confidence) ||
      grounding.confidence < threshold || grounding.confidence > 1) return reject('grounding_low_confidence');
  const text = input.text.trim().toLowerCase().replace(/[’‘]/gu, "'");
  if (!text || /[^\p{Script=Latin}\p{Number}\p{Punctuation}\p{Separator}\p{Symbol}\s]/u.test(text)) return reject('unsupported_language');
  // Explicit off-game framing does not become reward evidence just because a
  // classifier finds a kitchen noun. This gate never changes its predicted label.
  if (/\b(?:movie|film|book|novel|song|music|chat(?:ting)?|weather|website|internet|vacation|holiday|yesterday|tomorrow)\b/u.test(text)) return reject('out_of_domain_reference');
  if (isUnresolvedQuestion(text, grounding.label)) return reject('unresolved_question_reference');
  const decision = chooseAiDecision(state);
  if (grounding.label === 'trajectory') {
    result.targetFeatures = knownPositiveFeatures(input.trajectoryFeatures ?? {});
    const applicabilityReason = trajectoryApplicability(text, result.targetFeatures);
    if (applicabilityReason) return reject(applicabilityReason);
  } else if (grounding.label === 'feature') {
    if (!/\b(?:is|are|was|were|has|have|contains?|needs?|requires?|prefer|important|useful|helpful|good|bad|wrong|correct|blocks?|blocked|blocking|clears?|clearing|delays?|delaying|wastes?|wasting)\b/u.test(text)) return reject('unresolved_feature_reference');
    result.targetFeatures = featureReference(text);
  } else {
    const reference = actionReference(text, state);
    result.directivePolarity = reference.polarity;
    if (reference.reason) return reject(reference.reason);
    if (reference.subgoals.length !== 1) return reject('unresolved_action_reference');
    result.selectedSubgoal = reference.subgoals[0];
    const candidate = decision.ranking.find((item) => item.subgoal === result.selectedSubgoal);
    if (!candidate) return reject('infeasible_action_reference');
    result.targetFeatures = knownPositiveFeatures(candidate.features);
  }
  if (!Object.keys(result.targetFeatures).length) return reject('empty_target_features');
  if (!Object.keys(result.targetFeatures).some((feature) => activeFeatures.has(feature))) return reject('no_policy_active_target');
  if (result.directivePolarity !== 'prohibit') {
    // Match the released paper implementation before its L1 normalization.
    // Restricting this complement to feasible features increases per-feature
    // negative evidence and can reward a criticized task relative to another.
    for (const feature of REWARD_FEATURES) {
      if (!(feature in result.targetFeatures)) result.pragmaticAlternatives[feature] = 1;
    }
  }
  return { ...result, status: 'grounded' };
}
