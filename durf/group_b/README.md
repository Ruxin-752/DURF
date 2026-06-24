# Group B: LLM-Assisted Feedback Attribution

Reserved for the new natural-language feedback attribution work.

Planned scope:

1. Convert human language feedback into structured attribution records.
2. Use programmatic trajectory facts and candidate event detectors before
   invoking the LLM.
3. Use the LLM only for human-semantic interpretation that deterministic rules
   cannot resolve reliably.
4. Add clarification questions when attribution is ambiguous.
5. Add a dynamic schema dictionary for new attribution fields.
6. Build training samples for `H_u`, then combine `H_u` with the PPO baseline.

Implementation should start from the current pygame human-AI baseline in
`durf/group_a/play_with_baseline.py`.
