# Historical RLlib Agents

These agents were restored from commit `b1e6c627` by Ruxin. They are kept
outside the removed browser demo path so pygame code can load them directly.

Use each RLlib agent only with the layout it was trained on:

```text
RllibCrampedRoomSP              -> cramped_room
RllibCoordinationRingSP         -> coordination_ring
RllibForcedCoordinationSP       -> forced_coordination
RllibAsymmetricAdvantagesSP     -> asymmetric_advantages
RllibCounterCircuit1OrderSP     -> counter_circuit_o_1order
```

The pygame baseline currently defaults to `RllibCrampedRoomSP`.
