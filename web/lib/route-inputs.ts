import type { GameState } from './game';
import type {
  FeedbackFormPrediction,
  FeedbackLabel as ModelFeedbackLabel,
} from './browser-models';
import type {
  FeedbackRoute,
  FeedbackLabel,
  FeedbackProbabilities,
  PhraseResearchPrediction,
} from './research-types';

const LABEL_MAP: Record<ModelFeedbackLabel, FeedbackLabel> = {
  evaluative: 'Evaluative',
  imperative: 'Imperative',
  descriptive: 'Descriptive',
};

export function splitFeedbackPhrases(text: string): string[] {
  return text
    .split(
      /(?<=\.)\s+|\r?\n+|(?<=[。！？!?;；，,])|(?:但是|不过|然而|然后|接着)|\b(?:but|and then|however|then)\b/giu,
    )
    .map((part) =>
      part
        .trim()
        .replace(
          /^(?:(?:但是|不过|然而|然后|接着)\s*|(?:but|and then|however|then)\b\s*)+/iu,
          '',
        )
        .trim(),
    )
    .filter(Boolean)
    .slice(0, 12);
}

export function lowConfidenceRouteMessage(
  route: FeedbackRoute,
  threshold: number,
): string {
  const percent = (threshold * 100).toFixed(0);
  return route === 'route2'
    ? `低于 ${percent}% 阈值：fG 仅作诊断，Route2 仍按十模型输出更新；只有 Route1 会拒绝。`
    : `低于 ${percent}% 阈值：Route1 拒绝整句且不更新；Route2 不由 fG 门控。`;
}

export function toResearchPrediction(
  phrase: string,
  prediction: FeedbackFormPrediction,
): PhraseResearchPrediction {
  return {
    phrase,
    label: LABEL_MAP[prediction.label],
    confidence: prediction.confidence,
    probabilities: {
      Evaluative: prediction.probabilities.evaluative,
      Imperative: prediction.probabilities.imperative,
      Descriptive: prediction.probabilities.descriptive,
    },
    abstained: prediction.abstained,
  };
}

export function aggregateProbabilities(
  predictions: FeedbackFormPrediction[],
): FeedbackProbabilities {
  const count = Math.max(1, predictions.length);
  return {
    Evaluative:
      predictions.reduce((sum, item) => sum + item.probabilities.evaluative, 0) / count,
    Imperative:
      predictions.reduce((sum, item) => sum + item.probabilities.imperative, 0) / count,
    Descriptive:
      predictions.reduce((sum, item) => sum + item.probabilities.descriptive, 0) / count,
  };
}

export function topResearchLabel(probabilities: FeedbackProbabilities): FeedbackLabel {
  return (Object.entries(probabilities) as Array<[FeedbackLabel, number]>).sort(
    (left, right) => right[1] - left[1],
  )[0][0];
}

export function inferValence(text: string): number {
  const normalized = text.toLowerCase();
  const negative = /\b(?:bad|wrong|worse|awful|hate|dont|don't|stop|not)\b|差|错|不要|别|不对/u;
  const positive = /\b(?:good|great|nice|better|thanks|correct|please|should)\b|好|棒|对|请|应该/u;
  if (negative.test(normalized)) return -30;
  if (positive.test(normalized)) return 30;
  return 10;
}

export function gameFeatureCounts(state: GameState): Record<string, number> {
  return {
    ingredient_tomato:
      Number(state.player.held === 'tomato') + Number(state.partner.held === 'tomato'),
    ingredient_onion:
      Number(state.player.held === 'onion') + Number(state.partner.held === 'onion'),
    pick_tomato: Number(state.lastAction.includes('拿了番茄')),
    pick_onion: Number(state.lastAction.includes('拿了洋葱')),
    pick_dish: Number(state.lastAction.includes('拿了盘子')),
    pot_empty: Number(state.pot.stage === 'empty'),
    pot_cooking: Number(state.pot.stage === 'cooking'),
    pot_has_one_tomato: Number(state.pot.tomatoes === 1),
    pot_has_two_tomatoes: Number(state.pot.tomatoes === 2 && state.pot.onions === 0),
    pot_has_two_tomatoes_one_onion: Number(
      state.pot.tomatoes === 2 && state.pot.onions === 1,
    ),
    soup_ready: Number(state.pot.stage === 'ready'),
    serve_ready_soup: Number(state.lastAction.includes('完成一份')),
    supports_serving: Number(state.partner.held === 'soup'),
    completes_recipe: Number(state.ordersCompleted > 0),
    time_cost: (90 - state.secondsLeft) / 90,
    distance_cost: 0,
  };
}

export function actionFeaturesFromText(text: string): Record<string, number> {
  const normalized = text.toLowerCase();
  const output: Record<string, number> = {};
  if (/dish|plate|盘|碟/u.test(normalized)) output.pick_dish = 1;
  if (/tomato|番茄/u.test(normalized)) output.pick_tomato = 1;
  if (/onion|洋葱/u.test(normalized)) output.pick_onion = 1;
  if (/serve|deliver|上菜|出餐|端汤/u.test(normalized)) output.serve_ready_soup = 1;
  if (/move|go|走|移动|过去/u.test(normalized)) output.moves_toward_needed_object = 1;
  return output;
}

export function namedFeaturesFromText(text: string): Record<string, number> {
  const normalized = text.toLowerCase();
  const output: Record<string, number> = {};
  if (/pot|stove|锅|灶/u.test(normalized)) output.near_pot = 1;
  if (/tomato|番茄/u.test(normalized)) output.near_tomato_dispenser = 1;
  if (/onion|洋葱/u.test(normalized)) output.near_onion_dispenser = 1;
  if (/dish|plate|盘|碟/u.test(normalized)) output.near_dish_dispenser = 1;
  if (/serve|window|出餐|上菜/u.test(normalized)) output.near_serving_counter = 1;
  return output;
}
