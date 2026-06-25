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
RllibRingTomatoOnion10x6SP      -> ring_tomato_onion_10x6  # train locally first
```

The pygame baseline currently defaults to `RllibCrampedRoomSP`.

To train the new `ring_tomato_onion_10x6` agent locally:

```powershell
conda activate durf310
cd "C:\Users\my185\Desktop\研究\durf\DURF"
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.train_rllib_agent `
  --layout ring_tomato_onion_10x6 `
  --agent-name RllibRingTomatoOnion10x6SP `
  --iterations 2 `
  --num-workers 0 `
  --ray-local-mode `
  --overwrite
```

The command above is a smoke run. Increase `--iterations` for a usable policy.
