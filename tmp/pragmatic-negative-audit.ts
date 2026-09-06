import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createFullGaussianPrior } from '../web/lib/browser-models';
import { createGameState, chooseAiDecision, stepGame, type GameState } from '../web/lib/game';
import { REWARD_FEATURES, type RewardWeights } from '../web/lib/subgoal-policy';
import { groundRoute1Feedback } from '../web/lib/route1-grounding';
import { applyGroundedPragmaticFeedback, applyPragmaticRoute1Update } from '../web/lib/pragmatic-route1';
import { scoreVaderSentiment } from '../web/lib/vader-sentiment';
import { recordTrajectoryStep, featurizeTrajectorySteps } from '../web/lib/trajectory-featurizer';

const initial = createGameState('running');
const pickupBefore: GameState = { ...initial, partner: { ...initial.partner, x: 4, y: 4, facing: 'down' } };
const pickupAfter = stepGame(pickupBefore, 'interact', 'stay');
if (pickupAfter.partner.held !== 'onion') throw new Error('Real onion pickup failed');
const trajectoryFeatures = featurizeTrajectorySteps([recordTrajectoryStep(pickupBefore, pickupAfter, 'interact', 'stay')]);
const prior = () => createFullGaussianPrior([...REWARD_FEATURES]);
const weights = (mean: number[]): RewardWeights => Object.fromEntries(REWARD_FEATURES.map((feature, index) => [feature, mean[index]]));
const decision = (state: GameState, mean?: number[]) => {
  const result = chooseAiDecision(state, mean ? weights(mean) : undefined);
  const actual = stepGame(state, result.action, 'stay');
  const scores = Object.fromEntries(result.ranking.map(x => [x.subgoal, x.score]));
  return { chosenSubgoal: result.chosenSubgoal, action: result.action, scores,
    onionMinusTomato: 'GET_ONION' in scores && 'GET_TOMATO' in scores ? scores.GET_ONION - scores.GET_TOMATO : null,
    actualPartnerAfter: actual.partner, actualEvent: actual.lastEventCode, ranking: result.ranking };
};
const snapshots: [string, GameState][] = [
  ['empty_pot_tomato_station', {...initial, partner:{...initial.partner,x:1,y:4,facing:'left'}}],
  ['empty_pot_onion_station', pickupBefore],
  ['empty_pot_initial_position', initial],
  ['cooking_next_order_prefetch', {...initial, partner:{...initial.partner,x:1,y:4,facing:'left'}, pot:{stage:'cooking',tomatoes:2,onions:1,secondsRemaining:15}}],
];
const inputs = [
  {text:'That onion pickup was good.',label:'trajectory' as const},
  {text:'That onion pickup was bad.',label:'trajectory' as const},
  {text:'That onion pickup was terrible.',label:'trajectory' as const},
  {text:"Don't pick onions.",label:'action' as const},
];
const priors = [
  { id: 'zero_prior', make: prior },
  { id: 'synthetic_existing_progress_reward', make: () => { const p=prior(); p.mean[REWARD_FEATURES.indexOf('moves_toward_needed_object')]=30; return p; } },
];
const records = priors.flatMap(({id:priorId,make}) => snapshots.flatMap(([snapshotId,state]) => inputs.map(({text,label}) => {
  const grounding = groundRoute1Feedback({text,grounding:{label,confidence:0.99,modelHash:'fixed-correct-reference-for-updater-isolation'},state,trajectoryFeatures});
  const sentiment = scoreVaderSentiment(text);
  const result = applyGroundedPragmaticFeedback(make(),{grounding,sentiment:sentiment.compound});
  const legacy = applyPragmaticRoute1Update(make(),{feedbackForm:label==='action'?'imperative':'evaluative',feedbackFormConfidence:0.99,trajectoryFeatures:grounding.targetFeatures,actionFeatures:grounding.targetFeatures,sentiment:sentiment.compound});
  return {priorId,snapshotId,text,label,sentiment,grounding,status:result.status,effectiveValence:result.effectiveValence,valenceSource:result.valenceSource,
    before:decision(state,make().mean), literalOnly:decision(state,result.literal.state.mean), after:decision(state,result.state.mean),
    legacyGlobalComplement:decision(state,legacy.state.mean), delta:result.delta};
})));
const sourceFiles=['web/lib/route1-grounding.ts','web/lib/pragmatic-route1.ts','web/lib/browser-models.ts','web/lib/game.ts','web/lib/subgoal-policy.ts','web/lib/trajectory-featurizer.ts','web/lib/vader-sentiment.ts','web/lib/vendor/vader-data.json'];
const report={schema:'pragmatic-negative-behavior-audit-v1',dataScope:'Synthetic only. Actual onion pickup transition replayed, then same frozen observed features paired with constructed feasible decision snapshots. Grounding label is fixed correct to isolate updater; this is not classifier evaluation. Existing-progress prior is an explicit synthetic Gaussian fixture, not a claimed trained posterior. legacyGlobalComplement uses original low-level API, so its prohibition row intentionally does not include the separate live prohibition interpretation.',priorDefinitions:Object.fromEntries(priors.map(p=>[p.id,p.make()])),sourceHashes:Object.fromEntries(sourceFiles.map(p=>[p,createHash('sha256').update(readFileSync(p)).digest('hex')])),pickupBefore:pickupBefore.partner,pickupAfter:pickupAfter.partner,trajectoryFeatures,records};
mkdirSync('artifacts/pragmatic-negative-audit-20260906',{recursive:true});
writeFileSync('artifacts/pragmatic-negative-audit-20260906/report-after-full53.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({trajectoryFeatures,results:records.filter(r=>r.snapshotId==='cooking_next_order_prefetch').map(r=>({prior:r.priorId,snapshot:r.snapshotId,text:r.text,status:r.status,valence:r.effectiveValence,before:r.before.chosenSubgoal,literal:r.literalOnly.chosenSubgoal,after:r.after.chosenSubgoal,legacy:r.legacyGlobalComplement.chosenSubgoal,beforeMargin:r.before.onionMinusTomato,literalMargin:r.literalOnly.onionMinusTomato,afterMargin:r.after.onionMinusTomato,legacyMargin:r.legacyGlobalComplement.onionMinusTomato,action:r.after.action}))},null,2));
