# RllibRingHalfTaskStableTopLeft

Frozen half-task baseline for human-AI interaction testing.

- Source agent: `RllibRingFinalOnionHeldTargetTopLeftCandidate`
- Intended layout: `ring_tomato_onion_10x6_curriculum_final_onion_held_target_top_left`
- Purpose: stable "second half" cooking baseline while full complex-map curriculum is still being trained.
- Fixed evaluation: `outputs/baseline_eval_final_onion_held_target_top_left_candidate.json`
- Result: 10/10 successful episodes, mean reward 20.0.

This agent can complete the late-stage chain from holding onion near the upper-left corridor:

1. Move to pot and place onion.
2. Pick up dish.
3. Pick up ready soup.
4. Deliver soup.

It is not yet a full-complex-map baseline from the original starting state.
