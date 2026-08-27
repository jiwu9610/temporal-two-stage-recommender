# Wave-3 v2 noise band — provenance

Band = |best deep_cross(seed 42) − best deep_cross(seed 43)| per category, model_selection R@100.
The seed-43 arm is the report whose grid contains NO seq entries (run with --seed 43; later reports record `seed` explicitly).

- **Books** (A3_ap): seed42 0.06592 (best across `Books_stageB_A3_ap_seq_causal_pos_delta.json` runs) vs seed43 0.06585 (`Books_stageB_A3_ap.json`) → band ±0.00007 (0.11%)
- **Electronics** (A0_ap): seed42 0.10328 (best across `Electronics_stageB_A0_ap_seq_mh_pool_pos_delta_L_2.json` runs) vs seed43 0.10303 (`Electronics_stageB_A0_ap.json`) → band ±0.00025 (0.24%)
- **Video_Games** (A5_ap): seed42 0.16119 (best across `Video_Games_stageB_A5_ap_seq_causal_pos_delta.json` runs) vs seed43 0.16256 (`Video_Games_stageB_A5_ap.json`) → band ±0.00137 (0.85%)
