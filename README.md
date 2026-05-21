# Deep Learning Approach for Autonomous CVD Graphene Synthesis on Liquid Catalysts

Two-dimensional materials (2DMs) have been intensively studied in recent years for future applications in many technological areas. However, the lack of effective mass production techniques restricts their demand for scientific and industrial use. The Liquid Metal Catalysis (LMCat) project achieves the high quality growth of 2DMs by utilizing chemical vapor deposition (CVD) on liquid substrates. The liquid catalyst provides a smooth and homogeneous surface that minimizes defect nucleation and allows for rapid, self-aligning flake growth.

In this project, graphene is synthesized using $\text{CH}_4$ as the carbon precursor, $\text{H}_2$ to control the equilibrium reaction, and Ar as an inert carrier gas. The partial pressure ratio of methane to hydrogen ($P_{\text{CH}_4}/P_{\text{H}_2}$) dictates the kinetics of the system determining both the growth rate and the morphological shape of the graphene flakes.

Manually controlling this parameter to reach a target state of the graphene flakes can be complex and time-consuming. Therefore, this work proposes an autonomous control system for graphene synthesis in the LMCat reactor using computer vision and deep learning.

The methodology employs a pre-trained feature extractor (DINO) to compress *in-situ* optical microscope images of the growing flakes into a set of vector representations. A DINO-World-Model (DINO-WM) is then trained on these vectors alongside sequential gas flows to learn the system's dynamics, specifically how current states evolve under different applied actions.

Ultimately, given a target flake morphology and size, the system outputs the optimal sequence of gas pressure adjustments.

---

# Project Architecture

The repository is modularized to separate the physical hardware environment, the AI world models, and the decision-making planner.

```text
├── notebooks/             # Exploratory data analysis, messy plotting, and model prototyping
│
├── src/                   # Core Python Package
│   ├── data_handling/     # HDF5DataLoader classes and data processing pipelines
│   ├── environment/       # ReactorEnv: Unified interface connecting hardware and observers
│   ├── models/            # DINOv2 Encoder and Ensemble Latent Transition Models
│   ├── controllers/       # CEM Planner, LMCat RPC Controller, and Observers
│   └── utils/             # Math, plotting, and metric evaluation functions
│
├── scripts/               # Main execution files (e.g., run_automated_growth.py, make_video.py)
│
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

# 🔌 Hardware Prerequisites (BLISS Server Setup)

The following macros must be manually loaded into the lab computer (`opid10`) BLISS session before running any AI scripts.

```python
# Methane Flow Control
def set_flowCH4(F):
    Flow.CH4 = F

# Background Gases
def set_flowAr(F):
    Flow.Ar = F

def set_flowH2(F):
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
   Applies the most consistent CH4 flow rate in the reactor.

5. **Wait**  
   Allows gas travel and physical stabilization before the next cycle.