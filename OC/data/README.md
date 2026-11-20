# Versioned datasets

- `calibration_dataset_v1.json`: the profiling measurements behind the published calibration
  numbers (gamma_compute 0.688, held-out MAPE 17.52% global / 4.65% per model). Frozen: do not
  edit. Add a new version (`calibration_dataset_v2.json`, ...) for new measurements and pass it
  with `reproduce_evaluation.py --profile`.

Runtime files are written per machine and are not tracked by git (see `.gitignore`):
`OC/hardware_profile.json`, `OC/hardware_profile_local.json`, `OC/calibration_profile.json`,
`OC/held_out_evaluation_results.json`, `OC/cluster_nodes_config.json`.
