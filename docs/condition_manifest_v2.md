# Condition Manifest (protocol-v2)

> Generated from `durf/feedback_attribution/condition_features.py:MODEL_CONDITION_KEYS`
> Frozen: 2026-08-11

## 28 Model Conditions

Each condition is a boolean feature for Hu training. Encoding: `True → 1`, `False → -1`, `None/unknown → 0`.

### Task & Coordination Conditions

All 28 conditions are available for task head. Coordination head uses subset `COORDINATION_CONDITION_KEYS` (14 conditions).

| # | Key | Type | Domain | Description | Offline Recoverable |
|---|---|---|---|---|---|
| 1 | `pot_empty` | boolean | task | All pots are empty | Yes |
| 2 | `pot_partially_filled` | boolean | task | At least one pot has ingredients but not cooking/ready | Yes |
| 3 | `pot_cooking_or_ready` | boolean | task/coord | At least one pot is cooking or ready to serve | Yes |
| 4 | `human_has_dish` | boolean | task/coord | Human is holding a dish | Yes |
| 5 | `ai_has_dish` | boolean | task/coord | AI is holding a dish | Yes |
| 6 | `human_has_tomato` | boolean | task/coord | Human is holding a tomato | Yes |
| 7 | `human_has_onion` | boolean | task/coord | Human is holding an onion | Yes |
| 8 | `human_has_soup` | boolean | task/coord | Human is holding soup | Yes |
| 9 | `ai_has_tomato` | boolean | task/coord | AI is holding a tomato | Yes |
| 10 | `ai_has_onion` | boolean | task/coord | AI is holding an onion | Yes |
| 11 | `ai_has_soup` | boolean | task/coord | AI is holding soup | Yes |
| 12 | `ai_empty_handed` | boolean | task/coord | AI is not holding any object | Yes |
| 13 | `recipe_needs_tomato` | boolean | task | Recipe still requires tomato | Yes |
| 14 | `recipe_needs_onion` | boolean | task | Recipe still requires onion | Yes |
| 15 | `human_trying_to_pass` | boolean | task/coord | Human is actively moving through AI's path | Online only; offline = `ai_on_human_path` |
| 16 | `narrow_corridor` | boolean | task/coord | Path is constrained (corridor or tight space) | Yes |
| 17 | `ai_on_human_path` | boolean | task/coord | AI occupies tile(s) on human's movement line | Yes |
| 18 | `useful_object_adjacent` | boolean | task | A useful counter object is adjacent to AI | Yes |
| 19 | `human_waiting_near_pot` | boolean | task | Human is stationary near a pot | Yes |
| 20 | `human_holding_last_needed_ingredient` | boolean | task | Human holds the final ingredient recipe needs | Yes |
| 21 | `human_closer_to_dish` | boolean | task | Human is closer to dish dispenser than AI | Yes |
| 22 | `human_closer_to_pot` | boolean | task | Human is closer to pot than AI | Yes |
| 23 | `ai_closer_to_ingredient` | boolean | task | AI is closer to needed ingredient than human | Yes |
| 24 | `useful_counter_object_available` | boolean | task | A useful counter object exists on the map | Yes |
| 25 | `useful_counter_object_closer_than_dispenser` | boolean | task | Counter object is closer than matching dispenser | Yes |
| 26 | `useful_counter_object_closer_to_pot_than_dispenser` | boolean | task | Counter object is closer to pot than dispenser is | Yes |
| 27 | `useful_counter_object_lower_task_cost_than_dispenser` | boolean | task | Counter object has lower total task cost than dispenser | Yes |
| 28 | `ai_adjacent_to_current_subgoal_target` | boolean | task/coord | AI is 1 step from current task subgoal target | Yes |

### Coordination-Only Subset (14 conditions)

From `COORDINATION_CONDITION_KEYS`: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 28.

## 17 Context Conditions

Context keys are for provenance/review only. They do NOT enter the Hu model.

| # | Key | Type | Description |
|---|---|---|---|
| 1 | `needed_ingredient` | string | Which ingredient recipe still needs |
| 2 | `human_inferred_subgoal` | string | Program's best guess at human's task |
| 3-13 | `*_distance_*` | int | Various distance measurements |
| 14-17 | `useful_counter_*` / `matching_dispenser_*` | int/string | Counter object vs dispenser cost comparisons |
