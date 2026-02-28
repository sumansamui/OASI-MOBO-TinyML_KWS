# OASI: Objective-Aware Surrogate Initialization for Multi-Objective Bayesian Optimization in TinyML Keyword Spotting

This repository contains the experimental pipeline for optimizing Keyword Spotting (KWS) models for edge deployment. The framework utilises a Depthwise Separable Convolutional Neural Network (DSCNN) architecture and employs **OASI: Objective-Aware Surrogate Initialisation for Multi-Objective Bayesian Optimisation** to find the optimal trade-off between Validation Accuracy and Model Size (MB). The optimization pipeline leverages an Objective-Aware Surrogate Initialization (OASI) combined with Multi-Objective Bayesian Optimization (MOBO), culminating in Hardware-in-the-Loop (HIL) validation.

---

## Repository Structure

| File | Description |
|------|-------------|
| `oasi_init.py` | Executes the Chaos Simulated Annealing search to generate the initial Pareto front (`OASI_init.xlsx`) |
| `oasi_mobo.py` | Runs the MOBO pipeline sequentially across Random, LHS, Sobol, and OASI initializations |
| `result.py` | Computes Multi-Objective performance metrics (Hypervolume and Generational Distance) and generates comparative visualizations |

---

## Environment Setup

This project is built and tested on **Python 3.11.8**.

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## Execution Pipeline

The search and evaluation process is divided into three distinct stages. Execute the scripts in the following order:

### Step 1: Generate Initial Population

Run the initialization script to generate the foundational Pareto archive using OASI. This script should be set to run for **30 iterations** to build a robust starting population for the Bayesian search.

```bash
python oasi_init.py
```

### Step 2: Multi-Objective Bayesian Optimization (MOBO)

Once the initial population is generated, execute the MOBO pipeline. This will use Gaussian Process Regressors to explore the hyperparameter space and refine the models to the true Pareto front.

```bash
python oasi_mobo.py
```

### Step 3: Result Evaluation

After the MOBO iterations are complete, evaluate the mathematical performance metrics (Hypervolume and Generational Distance) and generate the comparative plots for your final models.

```bash
python evaluation.py
```

> Plots and CSV logs will be saved automatically to their respective output directories.

---

## Hardware-in-the-Loop (HIL) Testing

Validating on-device latency, RAM usage, and inference time is critical. Once the Pareto-optimal KWS models are extracted from `result.py`, they are validated on actual edge microcontrollers using the [ST Edge AI Developer Cloud](https://stedgeai-dc.st.com/home).

1. Export the selected Keras/TensorFlow `.h5` or `.tflite` models.
2. Upload the models to the ST Edge AI Developer Cloud platform.
3. Select the target STM32 board to benchmark C-code generation, flash memory utilization, and real-world inference latency.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{garai2025oasi,
  title={OASI: Objective-Aware Surrogate Initialization for Multi-Objective Bayesian Optimization in TinyML Keyword Spotting},
  author={Garai, Soumen and Pau, Danilo and Samui, Suman},
  journal={IEEE Embedded Systems Letters},
  year={2025}
}
```
