# Reference: worked numbers, pitfalls, and gains

Concrete results from applying this skill to the adapted Overcooked Baseline B
corpus (90 decision contexts, 368 rule-teacher feedback intents, DeepSeek
`deepseek-chat`, 4 paraphrases per intent).

## Before / after

| Corpus | Examples | Unique texts | Diversity | Valence (pos/neg) |
|---|---|---|---|---|
| Template-only | 1468 | 135 | 9.2% | 810 / 658 |
| Template + LLM (validated) | 2421 | 1014 | **41.9%** | 1296 / 1125 |

LLM augmentation kept 953 of 1017 generated paraphrases (94%) after validation,
roughly 4.5x the unique-text count while keeping valence balanced.

## Pitfalls learned (each one silently corrupts the corpus)

| Pitfall | Symptom | Fix |
|---|---|---|
| LLM as co-player, not critic | Utterances are "I'll grab a tomato" (self-intent); no valence about the agent | Reframe role to observer/critic of the agent's move |
| Free "paraphrase this" prompt | Paraphrases drift neutral or flip valence | Anchor prompt with stance + reason + noun + speech-act definition |
| Exact-match sentiment gate | Good hard negatives dropped as "sentiment mismatch" | Gate on *non-contradiction* only (raw compound vs ±0.5) |
| Gating on "modified VADER" (+0.5 neutral default) | Every subtle negative looks positive -> mass false drops | Use raw compound score for the gate |
| Contradiction threshold too tight (0.3) | ~29 valid negatives with compound ≈ +0.36 dropped | Raise to 0.5 |
| Grounding vocab misses yielding/idle words | "you're just standing there" (valid WAIT critique) dropped as ungrounded | Add `wait/idle/standing/hang/nothing/move/aside/...` to theme words; don't stopword the WAIT noun |
| Caching empty on network error | Intent permanently loses paraphrases | Don't cache empties; retry on rerun |
| Unhandled `IncompleteRead` / network exception | One flaky response aborts the whole batch | Catch broad exceptions around the LLM call |

## Example: kept vs dropped LLM utterances

Kept (natural, on-intent, correct valence):
- `[+ evaluative | GET_ONION]` "Good call grabbing that onion, I'll get the tomatoes."
- `[+ imperative | GET_ONION]` "Good, grab that onion while I get the tomato."
- `[- descriptive | WAIT]` "You're just standing there while I scramble for ingredients." (kept as a hard negative)

Dropped (correctly):
- Genuinely contradictory: negative-labeled text whose surface tone is clearly
  positive (compound ≥ 0.5).
- Ungrounded: mentions neither the resource nor any coordination-theme word.

## Honesty boundary (state this when reporting)

Synthetic feedback validates **language generalization** (free text -> correct
reward direction under a known rule), NOT the correctness of the rule `w*`
itself. LLM augmentation diversifies wording, not preferences. Real human data
is still required to discover/verify what humans actually value; keep a small
real held-out set that is never used for training.
