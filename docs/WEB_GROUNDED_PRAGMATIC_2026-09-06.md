# Independent Web Route1 grounding and VADER

The live API is `groundRoute1Feedback({text, grounding, state, trajectoryFeatures})`
followed by `applyGroundedPragmaticFeedback(prior, {grounding, sentiment})`.
`grounding.label` is independently predicted as `action`, `feature`, or `trajectory`.
The displayed Evaluative / Imperative / Descriptive result is not an input to this API.
The legacy `applyPragmaticRoute1Update` remains available for exact paper-math tests.

## VADER

`web/lib/vader-sentiment.ts` translates the official Python VADER 3.3.2 algorithm.
It uses the full original 7,506-entry sentiment lexicon, 3,570-entry emoji table,
intensifiers, negation rules, case and punctuation rules. It does not replace
raw sentiment with kitchen keywords. `scoreVaderSentiment(text)` returns original
`neg`, `neu`, `pos`, and `compound` components; they are not classifier accuracy.

The older JavaScript npm port was inspected but excluded after it scored
`Not good.` as positive. The current implementation gives the official Python
value `-0.3412`, while `Do not pick onions.` remains raw VADER neutral (`0`).

Tables and 287 differential cases were generated from official Python 3.3.2 by
`web/scripts/export-vader-reference.py`. Cases consist of upstream demonstration
sentences and generated synthetic probes; no participant feedback was used.
Source-code SHA-256 is recorded in both the tables and fixture. The official wheel
SHA-256 is `3bf1d243b98b1afad575b9f22bc2cb1e212b94ff89ca74f8a23a588d024ea311`.
The MIT license is preserved with the vendored tables and under public licenses.

Source: [official VADER implementation](https://github.com/cjhutto/vaderSentiment).

## Kitchen adaptations

- Action references select exactly one currently executable subgoal. Its feature
  vector comes from the real policy ranking, including held object, recipe,
  navigation feasibility, and human-path geometry. A command to put a held onion
  in the pot does not reward picking another onion.
- Trajectory references use only the supplied recent trajectory. Named feature
  preferences bind to reward attributes. Pure location/state descriptions do not
  imply action rewards and are rejected when they have no actionable preference.
- Unsupported directions, multiple action targets, double negations, impossible
  actions, low-confidence grounding, and empty references do not change the prior.
  This is a conservative finite kitchen grounding grammar, not unrestricted
  semantic understanding. English is the supported language.
- Pragmatic negative evidence uses every unreferenced dimension of the full
  53-feature reward space before L1 normalization, matching the original released
  observation implementation (`paper-full-feature-complement-v1`). An earlier
  feasible-only restriction was removed after the behavioral audit below.
- Explicit `Don't` / `Do not` / `Avoid` / `Stop` prohibitions have a separate
  `directivePolarity`. Their cited action receives `-30` and alternatives receive
  no negative update. `rawSentiment` stays unchanged and `valenceSource` records
  `explicit_prohibition`; the override is not presented as VADER sentiment.
- Other accepted observations retain paper neutral `+15`, raw sentiment times
  `30`, and observation precision `2`.

## Verification

Targeted verification: 319 tests passed (12 original pragmatic-math cases,
288 VADER tests, 19 grounding/policy cases); TypeScript and targeted lint passed.
The live-policy test changes `Pick a tomato.` to `Pick an onion.`, which changes
the chosen subgoal and atomic action. Executing the respective action through
`stepGame` picks a tomato in the first case and moves toward onions in the second.
Other tests cover prohibitions, held-ingredient insertion, soup pickup/serving,
cross-state feasibility, and the complete unreferenced feature complement.

These engineering tests establish the checked behavior and Python algorithm
parity. They do not establish independent human language accuracy or public
deployment status; those are reported separately by the release workflow.

## Follow-up integration and applicability audit

The component now calls `applyGroundedPragmaticUtterance`: candidate phrase
results commit only when every reference succeeds. A rejected utterance retains
the previous posterior and recent trajectory for a retry. Each research phrase
records its text, candidate status, final `committed` flag, referenced features,
raw sentiment, effective valence, and pragmatic contrast features.

An exposed synthetic development audit found that a recent onion pickup allowed
five unrelated or ambiguous utterances to update a trajectory-classified prior.
`kitchen-reference-applicability-v1` adds an independent applicability check:
questions, off-game framing, and bare feature references do not authorize learning;
trajectory references require a complete explicit appraisal or a kitchen-specific
past action/appraisal construction. Object-specific appraisals require observed
features for that object. Clear general appraisals such as `Great job!` and
`Well done.` remain valid with recent action evidence. Clearly targeted polite
action requests remain eligible for the action binder. These checks leave the
classifier's three-way prediction untouched.

Twenty-five added applicability tests cover all five exposed failures, six pairs
of minimal counterfactual sentences, clear appraisals, empty and mismatched
trajectory context, polite requests versus questions, and each independent
grounding class. The related 61 tests, TypeScript, and targeted lint passed.
No unseen `frozen-next` or `round2` cases were opened for this repair.
The applicability-only revision of `route1-grounding.ts` had SHA-256:
`18f1225c287130337ef4e9209dbd74e215b2a4b9cb1931c76835706373980396`.

## Full-complement correction and negative-feedback behavior

A separate synthetic audit replayed a real onion pickup with `stepGame`, then
reused its observed features (`ingredient_onion`, `pick_onion`, `pot_empty`) in
four constructed decision snapshots and two valid Gaussian priors. These are
controlled policy experiments, not classifier-accuracy examples or human data.
The second prior explicitly sets an existing progress reward of 30 and otherwise
zero means, with covariance 25I; it is a synthetic fixture, not a claimed trained
posterior.

While soup cooks, the earlier feasible-only complement concentrated -30 into
four unreferenced active dimensions. With that progress prior, `That onion pickup
was bad.` changed the real decision from GET_TOMATO / interact to GET_ONION /
right. Its onion-minus-tomato score became +17.5457. The same grounded trajectory,
VADER score, prior and state with the paper's full 50-dimension complement keeps
GET_TOMATO / interact and gives -9.0321. `terrible` behaved similarly (+18.7834
versus -7.7943). This demonstrated a kitchen adaptation error; the original full
complement did not show this inversion in the checked cases.

The live binder now restores the full original complement. The explicit
prohibition exception and applicability rejections remain intact. Positive
`good` and neutral `You picked an onion.` still select GET_ONION; `Don't pick
onions.` retains raw VADER, target valence -30 and no complement, then selects
GET_TOMATO with this prior. Six regression tests reproduce the earlier reversal,
compare full posterior state with the original API, and execute the resulting
actions through `stepGame`.

This correction also restores the original method's unmentioned-feature updates:
future coordination, safety, context and unsupported dimensions can receive
negative evidence. It does not establish universal monotonicity of criticism or
behavior across every prior and game state. The retained prohibition branch is
still an explicit kitchen interpretation rule, not unchanged paper behavior.

Pre-correction evidence is preserved in
`artifacts/pragmatic-negative-audit-20260906/report-before-full53.json`.
The same 32 controlled cases after correction are in `report-after-full53.json`.
All 67 related tests, TypeScript, and targeted lint passed after this correction.
Current `route1-grounding.ts` SHA-256 is
`496eaf093d4009ede49b80a2a2ddc66106865cbcb19832601b9864d1b0285c4d`.

The retained Route2 artifact was separately traced to
`paper_aligned_v5_seed137_selected/ensemble_manifest.json` (SHA-256
`b863aea76336a7c200dee950a6874032793505ea0b54a898750bb57ac7fd168a`).
Its corpus hash exactly matches `route2_teacher_feedback.paper_v5.synthetic.json`
(`336d92f2c9b575ee15c73520e7f0c35d3f351ed7b86ecacc62b673fd686c2d04`):
all 83,592 records have `source=template`, the synthetic v5 generator version,
and synthetic author IDs. Loading this prior synthetic artifact does not introduce
human training rows; it is not a Route2 retraining in the current release.
