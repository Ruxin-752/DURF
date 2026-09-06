# Synthetic acceptance protocol

All sentences and game states in this directory are AI-authored synthetic material. The acceptance author did not read candidate training/development texts, data files, or training generators. Gold labels have not been independently checked by human annotators.

Each round has 150 single sentences (50 per speech act), 20 mixed messages with 42 oracle clauses, and 20 uncertain/out-of-domain messages. The six unambiguous speech/grounding combinations cover evaluative/action, evaluative/feature, evaluative/trajectory, imperative/action, descriptive/feature, and descriptive/trajectory. This does not independently validate all nine possible combinations.

The predeclared criteria are macro-F1 >= 0.87 and each-class recall >= 0.85 for both prediction heads, plus mixed-message speech-act component-set accuracy >= 0.85. Raw top-1 scores, rejection coverage, and accepted-only accuracy are reported separately. Raw softmax is not a calibrated correctness probability.

For each candidate, both models and production inference/clause-splitting code are frozen before one inference pass. The predictions file records hashes of every loaded TypeScript source and both model artifacts. The dataset seal is checked before inference; source and model hashes are checked afterward. A model changed after looking at acceptance results needs a new blind dataset. Earlier acceptance sets become development regressions.

Uncertain messages are excluded from forced three-class accuracy. Their saved predictions are passed through the real grounding and Pragmatic update functions in two local synthetic contexts: an initial game with no trajectory and a game with an explicit recent onion pickup. Both actual Gaussian state changes and rejection reasons are recorded. These checks do not execute browser submission, create consent, or write participant records.

The root agent performs a separate training-overlap audit after each final evaluation. An empty source list in the acceptance scorer means `not_run`, not zero overlap. The first audit found two single-sentence overlaps, plus three uncertain/oracle-clause overlaps. Exact non-overlap alone does not prove template independence.

| Round | Frozen set | Single speech | Single grounding | Mixed components | Stateful uncertain updates |
| --- | --- | --- | --- | --- | --- |
| 1 | frozen-v2.json | 119/150; failed | 132/150; passed | 17/20 | old gate: 5/20 with recent history |
| 2 | frozen-next-v2.json | 123/150; failed | 115/150; failed | 18/20 | 0/20 in each of two contexts |
| 3 | frozen-round3.json | 143/150; passed | 146/150; passed | 18/20 | 0/20 in each of two contexts |

Round 3 speech-act recall is 90% evaluative, 100% imperative, and 96% descriptive. Grounding recall is 100% action, 94% feature, and 97.78% trajectory. All five evaluative errors are object/feature-value judgments (15/20 correct in that subgroup); three of these wrong predictions were accepted above the score threshold. This remains a known semantic weakness despite passing the predeclared aggregate gates. The two mixed-message errors lose the descriptive component. The final model has not become infallible or human-validated.

The round-3 root overlap audit found one single-sentence and one oracle-clause exact training match (development: zero). A secondary analysis excludes these records and conservatively excludes the complete mixed message containing the overlapping clause. The clean subset still passes: speech 142/149 (95.30%; macro-F1 0.9528), grounding 145/149 (97.32%; macro-F1 0.9725), mixed components 17/19 (89.47%). The original primary report and predictions are unchanged; `round3-clean-subset-report.json` documents this separate calculation.

After restoring the full 53-feature Pragmatic complement, the original round-3 predictions were replayed through the revised update rule. `round3-uncertain-stateful-full53-report.json` records 0/20 grounded and 0/20 actual updates in each of the two synthetic contexts. This is a post-correction regression, not a new classifier blind evaluation; it records the revised source hashes while retaining the original prediction hash.

Different rounds contain different sentences, so these raw percentages are not a controlled before/after comparison. Passing synthetic acceptance establishes only the tested synthetic contract, not human-player accuracy, probability calibration, or generalization to all language.

`frozen-v2.json` corrects the original pre-inference grounding definition while retaining every sentence and speech label. `frozen-next-v2.json` fixes one ambiguous adjective during authoring QA before any model inference. Both earlier seals remain available with explicit revision reasons.

The administrative database probe is implemented separately in `web/lib/research-diagnostic.ts` and `web/app/api/research/diagnostic/route.ts`. It accepts only an authenticated same-origin POST with an empty JSON object. Its fixed synthetic record is inserted, read, and deleted in one transaction in `research_diagnostics`, without using the consent/session/event/feedback tables. Unit tests use a fake D1 implementation; a live call is needed to establish production database transport. D1 batch transaction behavior is documented by [Cloudflare](https://developers.cloudflare.com/d1/worker-api/d1-database/#batch).
