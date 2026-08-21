# Autonomous Control of CVD Graphene Growth on Liquid Metal Catalysts

Code for a closed-loop control system (DINO-WM) that drives the LMCat reactor at
ID10-SURF (ESRF) toward a target graphene morphology, using in-situ optical microscopy
images and the methane flow rate as the control action. A DINOv2 encoder compresses
frames into a 384-d latent space, a trained MLP ensemble (the "transition model") predicts
how that latent state evolves under a given CH₄ flow, and an action planner searches flow
sequences that minimize the latent distance to a target image.

For the physics, methodology, and discussion of results/limitations, see the accompanying
thesis: *"Towards Autonomous Control of CVD Graphene Synthesis on Liquid Metal Catalysts: A
Deep-Learning-Based Computer Vision Approach"*. This README covers only
the code: how it's organised and how to run it.

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Installation](#2-installation)
3. [`config.yaml` reference](#3-configyaml-reference)
4. [Procedure 1 — Add data for training and validation](#4-procedure-1--add-data-for-training-and-validation)
5. [Procedure 2 — Train a model](#5-procedure-2--train-a-model)
6. [Procedure 3 — Validate a model](#6-procedure-3--validate-a-model)
7. [Procedure 4 — Select the model for execution](#7-procedure-4--select-the-model-for-execution)
8. [Procedure 5 — Execute the model on the reactor](#8-procedure-5--execute-the-model-on-the-reactor)
9. [Hardware prerequisites (BLISS)](#9-hardware-prerequisites-bliss)
10. [Extending the code](#10-extending-the-code)

---

## 1. Repository layout

```
.
├── config.yaml
├── requirements.txt
├── scripts/
│   ├── data_prep/
│   │   └── build_and_partition_data.py     # DINOv2 inference + train/val split (Procedure 1)
│   ├── training/
│   │   ├── train_ensemble_model.py         # canonical training entry point (Procedure 2)
│   ├── evaluation/
│   │   ├── compare_frames_for_transition.py    # transition analysis, thesis §4.1 (Procedure 3)
│   │   ├── evaluate_hyperpars_on_trajectory.py # hyperparameter sweep over hist/step_size
│   │   ├── generate_evaluation_videos.py       # replays a validation recording with model overlay
│   │   ├── generate_video_from_log.py          # replays an execution log (autonomous/equilibrium run)
│   │   ├── generate_image_from_log.py          # single-image summary of an execution log
│   │   ├── predict_next_action.py              # single-prediction sanity check
│   │   └── visualize_dino_features_pca.py, visualize_single_frame.py  # inspection utilities
│   └── execution/
│       ├── run_autonomous_growth.py        # target-seeking control loop (Procedure 5)
│       ├── hold_equilibrium_in_reactor.py  # holds the current state as target
│       ├── run_reactor_active_learning.py  # uncertainty-maximising exploration mode
│       └── functions_online_testing.py     # shared loop bodies used by the scripts above
└── src/
    ├── models/
    │   ├── dinov2_encoder.py   # DinoEncoder — inference only
    │   ├── transition.py       # TransitionModel, EnsembleTransitionModel
    │   └── trainer.py          # Trainer — bagging, checkpoint saving
    ├── data_handling/
    │   ├── hdf5_processor.py     # HDF5Processor — reads .h5, runs DINO, applies crop_index
    │   └── transition_loader.py # TransitionDataLoader — builds (z, a, y) windows from .npy files
    ├── controllers/
    │   └── cem_planner.py   # action planner, is not CEM — wraps the ensemble's action search (see §10 note)
    ├── environment/
    │   ├── environment.py            # ReactorEnv — observe()/act()
    │   └── LMCat_control/
    │       ├── controller.py  # Controller — blissclient RPC calls (set_flow_CH4, etc.)
    │       └── observer.py    # Observer — blissdata stream reads (Image, CH4, ...)
    └── utils/
        ├── evaluation.py  # Evaluator — offline metrics, trajectory rollouts, transition plots
        ├── logger.py      # CSV logging of executed decisions + video/image generation from logs
        ├── plotting.py    # all matplotlib figure code
        └── misc.py        # load_yaml_config, load_model_from_yaml_config, video compilation
```

Directories not tracked in git (created at runtime, see `.gitignore`): `data_arrays/`,
`data_processing/`, `models/`, `logs/`, `plots/`, `videos/`.

Run everything as a module from the repository root so `src/` and `config.yaml` resolve:

```bash
python -m scripts.data_prep.build_and_partition_data
python -m scripts.training.train_ensemble_model
python -m scripts.evaluation.compare_frames_for_transition
python -m scripts.execution.run_autonomous_growth
```

---

## 2. Installation

```bash
git clone <repo-url>
cd automated_graphene_growth
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

* GPU recommended — the DINOv2 inference is the expensive step, training the MLP ensemble afterwards is cheap. It is recommended to   run all scripts from a visa instance from ESRF. One can clone the repository as well as the environment in a directory that is accessible to everyone. That way the installation doesn't have to be repeated everytime a visa instance is created
* `DinoEncoder` pulls `dinov2_vits14_reg` from `facebookresearch/dinov2` via `torch.hub` on
  first use. Needs internet (or a pre-populated `torch.hub` cache) the first time it runs.
* `blissclient` / `blissdata` are only needed for [Procedure 5](#8-procedure-5--execute-the-model-on-the-reactor)
  — they connect to `lid10lmcatctrl`, i.e. they only work on ESRF network (including visa instances). I think the same happens with Data prep, as it needs to access the raw data from the experiments. Training, and offline evaluation run anywhere.

---

## 3. `config.yaml` reference

### `execution.active_model`

```yaml
execution:
  model_directory: "/data/lmcat/Computer_vision/automated_graphene_growth/src/models/saved_transition_models/20260616"
  active_model:
    num_models: 5
    latent_dim: 384       # fixed — DINOv2 ViT-S/14 output dim
    action_dim: 1          # fixed — CH4 flow is the only action
    activation: "relu"
    normalization: "layer"
    history_size: 4
    step_size: 1
    hidden_dimension: 256
    num_hidden_layers: 2
```

`src/utils/misc.py::load_model_from_yaml_config` is the single place that reads this block.
It does two things with it:

1. Builds an `EnsembleTransitionModel` with exactly these hyperparameters.
2. Reconstructs the checkpoint prefix as
   `{model_directory}/mlp_activation_{activation}_norm_{normalization}_hist{history_size}_step{step_size}_hiddim{hidden_dimension}`
   and loads `{prefix}_transition_model_{i}.pth` for `i in range(num_models)`.

So **the filenames are derived from these fields, not stored explicitly** — get any one of
`activation`, `normalization`, `history_size`, `step_size`, `hidden_dimension` wrong and the
loader looks for a file that doesn't exist (or, worse, finds an unrelated one with a
matching name).

`history_size` and `step_size` are in **frames of the training data**, not seconds — see
Procedure 2.

### `data_files`

One entry per recording (or per scan within a recording — the same `.h5` can be listed
several times with different `scan_number`).

| Field | Type | Meaning |
|---|---|---|
| `path` | string | Directory containing the file |
| `file_name` | string | Exact `.h5` file name |
| `scan_number` | int | Scan identifier within the file (used to build the HDF5 group path `{scan_number}.1/measurement/...`) |
| `crop_index` | int / `[start, end]` / `null` | `int` keeps frames `[0, crop_index]`; `[start, end]` keeps that interval; `null` keeps everything. Applied inside `HDF5Processor.slice_data` before any other slicing. |
| `training_ranges` | list of `[start, end]` | Frame intervals (after cropping) used for training. `-1` = to the end. |
| `validation_ranges` | list of `[start, end]` | Frame intervals used for validation. Should not overlap `training_ranges`. |

`build_and_partition_data.py` also branches on the literal filename
`"Gr_4_080426_camera_0001.h5"` to set `sleep_time_basler=1` (2× downsampling) instead of the
default `2`; every other file gets `sleep_time_basler=2`. This is currently hardcoded in the
script, not config-driven — check `HDF5Processor.slice_data` before adding a file with a
different acquisition interval.

---

## 4. Procedure 1 — Add data for training and validation

1. Append an entry to `data_files` in `config.yaml` (see table above).
2. Run:

   ```bash
   python -m scripts.data_prep.build_and_partition_data
   ```

   For each entry this:
   - reads `{path}/{file_name}`, group `{scan_number}.1/measurement/basler` for frames and
     `.../measurement/CH4` for the flow;
   - if `data_arrays/{file_name stem}_scan{scan_number}_embeddings.npy` and
     `..._CH4.npy` already exist, loads them directly (no re-encoding);
   - otherwise runs `DinoEncoder.encode_numpy_array` over every frame and caches the result
     to `data_arrays/`;
   - slices the (cached or freshly computed) embeddings/flow by each interval in
     `training_ranges` / `validation_ranges` and writes them to
     `data_processing/training_data/train_seq_{id}_chunk{n}.npy` /
     `train_CH4_{id}_chunk{n}.npy` (and `eval_*` under `data_processing/validation_data/`
     for the validation ranges).

   **This makes the cache keyed only on `file_name` + `scan_number`, not on the ranges.**
   Editing `training_ranges` / `validation_ranges` for an already-processed file re-slices
   from the cached `.npy` in `data_arrays/` and skips DINO inference entirely. Editing
   `crop_index` does **not** invalidate the cache automatically — delete the corresponding
   files in `data_arrays/` by hand if you change it.

3. Sanity-check the printed shapes (`Embeddings: (...) | CH4: (...)`) and the list of saved
   chunk files against what you expect.

`TransitionDataLoader` (used by training/evaluation) pairs every `train_seq_*.npy` with the
`train_CH4_*.npy` that has the same suffix, and will raise if the file counts or identifiers
don't line up — so a partial write (e.g. the run was interrupted) surfaces immediately at
training time.

---

## 5. Procedure 2 — Train a model

Open `scripts/training/train_ensemble_model.py` and set the hyperparameters at the top —
they are plain Python variables in the script, not read from `config.yaml`:

```python
activation = "leaky_relu"
hidden_dimension = 2048
hist = 1          # history_size, in frames
step_size = 45    # stride between paired frames, in frames
normalization = "layer"
```

`hist` / `step_size` are in units of **frames of the source recording**, so the physical
Δt they represent depends on that recording's acquisition interval (2 s normally, 1 s for
the one file flagged in `build_and_partition_data.py`). `step_size = 45` at a 2 s interval
is the thesis's 90 s prediction step.

Then run:

```bash
python -m scripts.training.train_ensemble_model
```

This will:

1. Build `EnsembleTransitionModel(num_models=5, latent_dim=384, action_dim=1, ...)` with
   the hyperparameters above.
2. Load all chunks from `data_processing/training_data/` via `TransitionDataLoader`.
3. Train each ensemble member on an independent bootstrap resample of the full training set
   (`Trainer.train_ensemble_with_bagging`, Adam, lr `1e-3`, batch `64`, `10` epochs — set in
   the `Trainer(...)` call in the same script).
4. Save each member to
   `models/transition/mlp_activation_{activation}_norm_{normalization}_hist{hist}_step{step_size}_hiddim{hidden_dimension}_transition_model_{i}.pth`.
5. Plot the training loss and print the validation loss / cosine similarity on
   `data_processing/validation_data/`.

**The saved filename must match what `load_model_from_yaml_config` will reconstruct from
`config.yaml`** — that's the link to Procedure 4. Keep the hyperparameters in this script
and in `config.yaml`'s `active_model` block in sync, or write them down somewhere before you
train a batch of variants.

`ensemble_model_main.py` in the same folder is an earlier/alternate version of this script
with a different (non-config-compatible) checkpoint naming scheme (`save_prefix="bagging"`)
— useful as a quick sandbox but its output won't be picked up by the execution scripts as-is.

---

## 6. Procedure 3 — Validate a model

The thesis's offline "transition analysis" (§4.1) is `compare_frames_for_transition.py`.
Open it and set, near the top of `main()`:

```python
hist, step_size, hidden_dimension, normalization, activation = ...  # must match a trained checkpoint
model_name_prefix = PROJECT_ROOT / "models" / "transition" / f"mlp_activation_{activation}_norm_{normalization}_hist{hist}_step{step_size}_hiddim{hidden_dimension}"

horizon = 2
movie_num = 7          # index into config.yaml's data_files list
initial_frame_idx = 180
```

Run:

```bash
python -m scripts.evaluation.compare_frames_for_transition
```

It loads the two frames (`initial_frame_idx` and `initial_frame_idx + step_size*horizon`)
from `data_files[movie_num]`, evaluates every candidate flow through
`EnsembleTransitionModel.predict_action_losses`, and calls
`Evaluator.analyze_and_plot_transition` to save a figure to
`plots/transition_comparison_movie{n}_frame{i}_to_frame{j}.png` showing the two frames plus
the mean planning loss (with ensemble std) per candidate flow — the same plot format as
thesis Figures 4.1–4.3.

Other evaluation entry points, all similarly parameterised by editing constants at the top
of the file rather than by CLI flags (only the `generate_*` scripts take `argparse` args):

| Script | Purpose |
|---|---|
| `evaluate_hyperpars_on_trajectory.py` | Sweeps `hist`/`step_size` grids, trains + evaluates each combination on a full trajectory. |
| `generate_evaluation_videos.py` | `--frame_rate`; replays a validation recording (`movie_num` list hardcoded) with model predictions overlaid, output to `videos/`. |
| `generate_video_from_log.py` / `generate_image_from_log.py` | `--log_name --movie_num [--frame_rate]`; replays a CSV log written by Procedure 5 (see `src/utils/logger.py`), for reviewing an actual reactor run. |
| `predict_next_action.py` | Loads one saved model and prints its predicted next action for one validation window — quickest smoke test that a checkpoint loads and runs. |
| `visualize_dino_features_pca.py`, `visualize_single_frame.py` | Inspect what DINOv2 features look like on given frames; no transition model involved. |

`Evaluator` (`src/utils/evaluation.py`) also exposes `evaluate_ensemble_transition_model`
(aggregate MSE/cosine similarity over a dataset) and `evaluate_ensemble_on_trajectory`
(rollout error over a full sequence) if you need a numeric regression check instead of a
per-transition plot.

---

## 7. Procedure 4 — Select the model for execution

1. Copy the five `*_transition_model_{0..4}.pth` files produced in Procedure 2 into
   `execution.model_directory` (or point `model_directory` at wherever they already are).
2. Edit `execution.active_model` in `config.yaml` so every field matches the checkpoint's
   training hyperparameters exactly: `num_models`, `activation`, `normalization`,
   `history_size`, `step_size`, `hidden_dimension`, `num_hidden_layers`. `latent_dim: 384`
   and `action_dim: 1` should not change.
3. Sanity-check the load, e.g.:

   ```bash
   python -m scripts.evaluation.predict_next_action
   ```

   or by calling `load_model_from_yaml_config(config_path)` directly — a missing/misnamed
   file should directly raise an error.

There is no validation that the config's architecture matches what's actually inside the
`.pth` file; a shape mismatch surfaces as a `RuntimeError` from `load_state_dict`, but a
shape-compatible-but-wrong config will load and run silently.

---

## 8. Procedure 5 — Execute the model on the reactor

Preconditions: reactor at the operating point, BLISS session on `opid10` has the flow
macros loaded (§9), and the machine running the script can reach
`redis://lid10lmcatctrl:25002` (data) and `http://lid10lmcatctrl:8080` (control) — both
hardcoded in `ReactorEnv` / `Controller`.

Three execution modes, all built on the same `CEMPlanner` (despite the name, it's the
ensemble's discrete constant-flow search — see §10) plus `ReactorEnv`:

```bash
python -m scripts.execution.run_autonomous_growth       # target-seeking
python -m scripts.execution.hold_equilibrium_in_reactor  # hold current state
python -m scripts.execution.run_reactor_active_learning  # uncertainty-maximising exploration
```

* **`run_autonomous_growth.py`** — the target frame is set by hardcoded `movie_num` /
  `initial_frame_idx` at the top of `main()`, pulled from an existing recording in
  `data_files` and encoded with `DinoEncoder`. To target a different morphology, change
  those two lines (or adapt the script to load an arbitrary external image). Runs
  `growth_loop_with_target` from `functions_online_testing.py`, then falls back to
  `hold_equilibrium_loop` once the target is judged reached
  (L2 distance below a threshold and cosine similarity > 0.85 — see that file for the exact
  stopping rule and the 3-consecutive-agreeing-predictions actuation filter).
* **`hold_equilibrium_in_reactor.py`** — target is whatever the reactor looks like *right
  at start-up*; the planner is asked to hold it there (`action_space="closer_7"`,
  i.e. flow candidates near the current flow rather than the full 0–9 sccm grid).
* **`run_reactor_active_learning.py`** — no target; calls
  `CEMPlanner.get_highest_variance_action` each step, applying the flow that maximises
  ensemble-prediction disagreement (`horizon=2`).

All three log every decision via `log_model_decision` to
`logs/<mode>_log_<timestamp>.csv`, which is what `generate_video_from_log.py` /
`generate_image_from_log.py` (Procedure 3 table) replay afterwards.

Loop cadence in all three: observe → encode → plan → (maybe) act → `time.sleep(5)`
(`run_reactor_active_learning.py` sleeps `step_size*2` instead — check it matches your
intended cadence before a long unattended run).

---

## 9. Hardware prerequisites (BLISS)

These macros must be loaded in the BLISS session on `opid10` before running any execution
script — `Controller` calls them by name over RPC and the call fails if they're missing.

```python
def set_flow_CH4(F):
    Flow.CH4 = F

def set_flow_Ar(F): #Optional, not necessary
    Flow.Ar = F

def set_flow_H2(F): #Optional, not necessary
    Flow.H2 = F

def set_reactor_pressure(P): #Optional, not necessary
    Flow.Pressure = P
```

Data in: `blissdata.DataStore` (`Observer`, session `lmcat_camera`, streams `basler:image`,
`CH4:CH4`, `H2:H2`, `Ar:Ar`, `Pressure:Pressure`, `ArAux:ArAux`,
`nanodac_thermocouple_T:nanodac_thermocouple_T`). Setpoints out: `blissclient.BlissClient`
(`Controller`, session `lmcat_ctrl`).

---

## 10. Extending the code

* **`src/models/transition.py`** — architecture and planning logic both live here.
  `EnsembleTransitionModel.predict_action_losses` is the constant-flow discrete search;
  it's what both `CEMPlanner` and `compare_frames_for_transition.py` call. If you implement
  variable-length action sequences (thesis §5.2), this is the method to replace — note it's
  currently duplicated almost verbatim inside `CEMPlanner` itself
  (`src/controllers/cem_planner.py`), so update both or refactor one to call the other.
* **`CEMPlanner` name is a misnomer as of this codebase** — it does not implement the
  cross-entropy method (no iterative resampling/refitting of a search distribution); it
  wraps the same discrete grid search as `EnsembleTransitionModel.predict_action_losses`. A
  real CEM planner, or any planner over variable action sequences, would replace this class
  without needing changes to `ReactorEnv` or the execution scripts, since they only depend
  on `get_best_action(current_z, current_a, target_z, action_space)`.
* **`src/data_handling/transition_loader.py`** — where the pairing window (`step_size`,
  `hist_length`) and the loss-weighting-relevant `(z, a, y)` triples are built. Changing how
  training pairs are constructed (e.g. an EMA-based history instead of a flat stack, per
  thesis §5.2) starts here.
* **`src/environment/`** — the only code that talks to hardware. `ReactorEnv` is a thin
  facade; `Controller`/`Observer` are the BLISS-specific pieces. Swapping in a simulator or
  a different reactor's control system means implementing these two with the same
  `observe()` / `act()` interface.
* **`src/utils/evaluation.py::Evaluator`** is the shared home for offline metrics
  (`evaluate_ensemble_transition_model`, `evaluate_ensemble_on_trajectory`) and plotting
  glue (`analyze_and_plot_transition`, `generate_video_frames_for_validation`) — new
  offline diagnostics belong here rather than duplicated in a `scripts/evaluation/*.py`.
