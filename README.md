Here is the updated, structurally sound `README.md`.

The training instructions have been completely rewritten to reflect the new centralized configuration and the fused, idempotent extraction pipeline. I also explicitly noted that DINOv2 is strictly used for inference during the extraction step, ensuring there is no confusion about which model is actually being trained.

```markdown
# Towards Autonomous Control of CVD Graphene Synthesis 

This repository contains the codebase for implementing a closed-loop control system for Chemical Vapor Deposition (CVD) graphene growth on Liquid Metal Catalysts (LMCat). 

Using a deep-learning-based computer vision approach, this framework employs a self-supervised Vision Transformer (DINOv2) to encode in-situ optical microscope images into a latent space. A DINO-World-Model (DINO-WM) is then trained to predict the system's evolution and output optimized methane ($CH_4$) gas flow sequences to reach target graphene morphologies autonomously.

## Repository Structure
* `data_processing/`: Scripts for initial data cleaning and preprocessing.
* `scripts/`: Executable scripts for building arrays, partitioning data, training the model, and running the autonomous growth sequences.
* `src/`: Core module containing the controllers (CEM planner), data handling (HDF5 processing), environment observations, and the DINO-WM architecture.

## Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt

```

## Training the Ensemble Model

Follow these steps to extract visual embeddings (via pre-trained DINOv2 inference), partition your data, and train the predictive world model.

**1. Configure the Pipeline (`config.yaml`)**
All data files, hardware acquisition settings, and partitioning ranges are now controlled centrally.

* Open `config.yaml` located at the root of the repository.
* Add your raw `.h5` recording files to the `data_files` list.
* Explicitly define the `training_ranges` and `validation_ranges` (as index intervals) for each file.
* Ensure the `sleep_time` parameter is set correctly according to the camera acquisition settings used during that specific experiment.

**2. Build and Partition Data Arrays**
Convert all movies and $CH_4$ flow measurements into numpy arrays and partition them automatically based on your configuration.

* Run the following fused script:

```bash
python -m scripts.data_prep.build_and_partition_data

```

* **Note:** This script is idempotent and acts as a computational cache. It saves intermediate `.npy` arrays to `data_arrays/`. If you only update a partition range in `config.yaml`, the script will automatically bypass the computationally heavy DINO inference step and instantly repackage the arrays from the local disk cache.

**3. Train the Model**
With the training and validation chunks saved directly into `data_processing/`, the system is ready for training.

* Open `scripts/training/train_ensemble_model.py` and verify that the model hyperparameters are set to your desired values.
* Execute the training script:

```bash
python -m scripts.training.train_ensemble_model

```

---

# Hardware Prerequisites (BLISS Server Setup)

The following macros must be manually loaded into the lab computer (`opid10`) BLISS session before running any AI scripts.

```python
# Methane Flow Control
def set_flow_CH4(F):
    Flow.CH4 = F

# Background Gases
def set_flow_Ar(F):
    Flow.Ar = F

def set_flow_H2(F):
    Flow.H2 = F

# Reactor Pressure
def set_reactor_pressure(P):
    Flow.Pressure = P

```

> **Note:** The `ReactorEnv` environment will fail to step if these functions are missing from the server's global namespace.

---

# System Workflow

1. **Observe**
Grabs the latest Basler camera frame and sensor telemetry via Redis.
2. **Encode**
Compresses the image into a 384-dimensional latent embedding using DINOv2.
3. **Plan**
The CEM Planner simulates hundreds of future flow trajectories using the Ensemble Transition Model and calculates the path with the lowest target loss.
4. **Act**
Applies the most consistent $CH_4$ flow rate in the reactor.
5. **Wait**
Allows gas travel and physical stabilization before the next cycle.