---
name: llm-feedback-corpus
description: >-
  Prompt, deduplicate, and validate LLM-generated linguistic-feedback data for
  reward learning (Baseline B / "Learning Rewards from Linguistic Feedback"
  style). Use when synthesizing natural-language feedback with an LLM
  (e.g. DeepSeek), when an LLM "human" should critique an agent's action rather
  than narrate its own plan, or when deciding which generated utterances are
  useful enough to add to a training corpus.
---

# Synthesizing LLM feedback data for reward learning

Goal: turn a cheap LLM into a source of *reward feedback* language that a
reward learner can train on, without collecting a human corpus. The signal that
matters is `(state, utterance) -> reward direction`, not fluent chit-chat.

## The one principle that matters most

**The LLM must be an OBSERVER/CRITIC of the agent's move, not a co-player
narrating its own intent.** Reward learning needs feedback *about the agent's
behavior* with a valence ("that blocks me" / "good, that splits the work"). A
co-player who says "I'll grab a tomato" produces self-intent narration — fluent,
human-like, and **useless as reward signal** (no valence attached to the
agent's action). If your utterances start with "I'll ...", you are collecting
the wrong thing.

## Decouple label from language

- **Label (reward direction + grounding) comes from a rule teacher**, never the
  LLM. You must already know, in each state, which action is good/bad and which
  features it touches (e.g. a gold weight vector `w*` + a featurizer). This is
  the `attributed_sentiment_score` / `target_features` on each example.
- **Language comes from the LLM.** It only rephrases a known intent naturally.

This makes generation cheap, honest, and safe: training uses the rule's label,
so a mis-toned LLM sentence cannot flip the reward direction. It also bounds
what you can claim — you validate *language generalization*, not the rule.

## Prompt recipe (anchored, per feedback intent)

Give the model concrete anchors, not a free "paraphrase this". Build the prompt
from the rule teacher's intent `(feedback_type, polarity, subgoal, referenced
features)`:

```
System: You are a person playing <task> with an AI teammate. You give the AI
quick, natural, spoken feedback about the move it is making right now. Always
first person, casual, one short sentence each, no quotes, no emojis.

User:
Situation: <compact state description>
The AI is <gerund phrase of the subgoal> (<short noun phrase>).
Your honest view: this is <GOOD|BAD> because <plain-language reason from the
    referenced features, e.g. "you're blocking my way" / "that splits the work">.
<speech-act instruction>            # see below
Write N DIFFERENT natural one-sentence reactions you'd say to the AI. Hard rules:
- Every sentence must stay clearly <GOOD|BAD>; never sound <opposite>.
- Every sentence must clearly be about <noun> / <gerund>.
- Vary the wording; do not just reorder the same words.
Respond with a JSON array of N strings and nothing else.
```

Speech-act instruction by feedback type:
- **evaluative**: "Judge that move as good / the right call" (or bad / wrong).
- **imperative**: "Give a COMMAND telling the AI to do that now" (or to stop /
  not do it).
- **descriptive**: "Describe, as a plain fact, what that move does for us" (or
  what it does TO YOU / how it gets in your way).

Why anchors: the stance + reason + noun keep paraphrases on-intent; without them
the LLM drifts to neutral or flips valence.

## Dedup + diversity

- Deduplicate by `(scenario_id, lowercased_text)`; keep templates and LLM
  paraphrases in one pool.
- Always keep a **deterministic template floor** (a few hand-written variants
  per intent) so the corpus is never empty and never depends on a network call.
  The LLM adds diversity on top.
- Track **diversity ratio = unique_texts / total**. Template-only is very low
  (~9%). Good LLM augmentation lifts it to ~40%+. If it stays low, the LLM is
  echoing the template — strengthen the "vary the wording" rule or raise N.

## Validation recipe: keep only useful utterances

"Useful" = reading the **text alone**, the reward learner's own tooling recovers
the intended signal. Gate LLM samples; always keep templates (trusted floor).

1. **Sentiment = non-contradiction, NOT exact match.** Use the *raw* sentiment
   score (e.g. VADER compound). Reject only when it **clearly contradicts** the
   gold label:
   - positive label rejected if compound <= -0.5
   - negative label rejected if compound >= +0.5
   Do **not** require the text to read strongly negative/positive. Implicit
   criticism ("I needed that, you just took it") reads neutral yet is a valid
   **hard negative** — and hard negatives are the scarcest, most valuable data.
   Pitfall: a "modified VADER" that returns a +0.5 default for neutral text makes
   every subtle negative look positive; never gate on that variant.
2. **Grounding.** The text must mention the behavior's resource OR a
   coordination-theme word. Make the theme vocabulary cover yielding/idle/motion
   (`wait, idle, standing, hang, nothing, move, aside, block, path, ...`) or you
   will wrongly drop good "you're just standing there" criticism.
3. **Feedback-type agreement**: report it, but do **not** gate on it — keyword
   type classifiers are crude and type is fuzzier than valence.

## Metrics that decide "is this data useful?"

Report these; the corpus is healthy only if all hold:
- **Valence balance**: has BOTH positive and negative (a corpus with ~0
  negatives cannot teach what to avoid).
- **Speech-act coverage**: evaluative + imperative + descriptive all present.
- **Diversity ratio** meaningfully above the template floor.
- **Non-contradiction rate** high (~99%): almost no sample whose surface tone
  opposes its label.
- **Scenario coverage**: intents drawn from many distinct decision contexts.

## Resilience (batch LLM generation)

- **Cache by a semantic prompt hash** (role + subgoal + theme + feedback_type +
  polarity + N), so reruns are free and reproducible.
- On a transient network error, **do NOT cache the empty result** — return empty
  for that intent and let a rerun retry it (otherwise you permanently lose it).
- Catch broad exceptions around the LLM call so one hiccup (e.g. an
  `IncompleteRead`) cannot abort the whole batch.

## Workflow (runnable)

```
- [ ] 1. Rule teacher labels each decision context -> feedback intents
        (feedback_type, polarity, subgoal, referenced features).
- [ ] 2. Render a deterministic template floor per intent (never empty).
- [ ] 3. LLM augment with the anchored prompt above (cache by semantic hash;
        don't cache empties on network errors).
- [ ] 4. Dedup by (scenario_id, lowercased text).
- [ ] 5. Validate + gate, keeping only useful LLM samples:
          python scripts/validate_feedback.py corpus.json --output corpus.validated.json
- [ ] 6. Read the metrics; the corpus is usable only if valence is balanced,
        all speech-act types appear, and diversity beats the template floor.
```

`scripts/validate_feedback.py` is standalone (only optional NLTK VADER) and runs
on any corpus of `{text, attributed_sentiment_score}` records — adapt
`SUBGOAL_KEYWORDS` / `THEME_WORDS` to your task, or add a `keywords` field per
example.

## Reference implementation in this repo

Under `baselines/baseline_b_linguistic_feedback/adapted_overcooked/`:
- `src/subgoal_teacher.py` — rule teacher: state -> feedback intents (label).
- `src/feedback_templates.py` — template floor + feature-to-plain-language themes.
- `scripts/generate_synthetic_feedback.py` — anchored LLM prompt + caching.
- `scripts/validate_feedback_corpus.py` — the non-contradiction + grounding gate.

See [reference.md](reference.md) for the worked numbers, the pitfalls table, and
the observed before/after diversity and valence gains.
```
