# DURF Research Code

This directory is the home of project-specific research code.

```text
durf/
|-- baseline/   RLlib PPO agent evaluation and visualization
|-- group_a/    Human-PPO pygame play and session logging
`-- group_b/    LLM-assisted feedback attribution (planned)
```

The Overcooked-AI environment remains in `src/overcooked_ai_py/`. Historical
trained RLlib agents live under `models/rllib_agents/`; the default pygame
baseline uses `RllibCrampedRoomSP` on `cramped_room`.

Legacy standalone scalar-feedback tools were moved to
`archive/legacy_scalar_feedback/`. New research work should start from the
pygame human-AI baseline and build the feedback-attribution package on top of
the recorded session trajectories.
