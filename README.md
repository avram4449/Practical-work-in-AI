# Practical Work in AI – Active Learning on Tox21

This repository contains the code and report for the **Practical Work in AI** project:  
**Evaluating BALD with Monte Carlo Dropout versus Random Acquisition in Active Learning on the Tox21 Dataset**.

---

## Project Overview

Active Learning (AL) reduces labeling costs by selectively querying the most informative samples.  
In this project, we compare two strategies:

- **BALD with Monte Carlo Dropout** (uncertainty-based acquisition)  
- **Random acquisition** (uninformed baseline)

on the **Tox21 molecular dataset**, which consists of ~12,000 compounds annotated with 12 toxicity endpoints.  
We evaluate three labeling regimes: **420, 500, and 5500 samples**.

**Key findings:**
- At **420 samples**, BALD underperforms Random (cold-start issue).  
- At **500 samples**, BALD clearly outperforms Random.  
- At **5500 samples**, both converge, but BALD maintains a modest edge.

See the full details in the report.

---

## Requirements & Environment Setup

### Dependencies

All dependencies are listed in `requirements.txt`.  
Main libraries:
- Python 3.9+
- PyTorch 1.12.0 (see requirements.txt for tested version)
- PyTorch Lightning
- RDKit (install via `conda install -c conda-forge rdkit`)
- scikit-learn
- NumPy, SciPy
- Matplotlib, Seaborn

### Setup

```bash
git clone https://github.com/<avram4449>/<Practical-work-in-AI>.git
cd <Practical-work-in-AI>
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
.\venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

---

## Data

We use the **Tox21** dataset.  
- Preprocessed file: `data/tox21_compoundData_wSmiles.csv`  
- Molecules are converted into **2048-bit ECFP6 fingerprints** using RDKit.  
- Labels cover 12 binary toxicity endpoints.  

**Dataset path configuration:**  
Set the path to your dataset in the config file:  
`src/config/data/tox21.yaml`
```yaml
csv_path: C:/FILES/jku/Semester 6/Practical work/realistic-al-open-source/src/data/tox21_compoundData_wSmiles.csv
```
Alternatively, you can override the path via command line or in `config.yaml`.

---

## Running Experiments

### Train & Run Active Learning

```bash
python main.py
```

This will:
1. Load the Tox21 dataset.
2. Initialize the MLP model.
3. Run the AL loop with the chosen acquisition strategy (BALD or Random).
4. Save logs to `experiments/test/`.

### Experiment settings

- **420 samples:** small fixed budget.
- **500 samples:** small budget with minimal reliability.
- **5500 samples:** 20 iterations × 250 acquisitions.

Seeds used: `12345, 23456, 34567`.

**To change acquisition strategy or other settings:**  
Edit the config files in `src/config/` or override via Hydra command line arguments.

---

## Analysis & Visualization

Analysis scripts and notebooks are in the `analysis/` folder.  
For example, to plot per-class AUC curves:

```bash
python analysis/reportplot21.py experiments/test/
```

This will generate grid plots comparing BALD and Random across all Tox21 targets.

---

## Project Structure

```
├── analysis/        # Jupyter notebooks and analysis scripts
├── docs/            # Documentation and report
├── experiments/     # Experiment configs and logs
├── launchers/       # Scripts to launch experiments
├── src/             # Main source code (models, query strategies, training)
├── ssl/             # Additional resources
├── visuals/         # Plots and figures
├── requirements.txt # Python dependencies
├── pyproject.toml   # Build metadata
├── LICENSE          # License
├── README.md        # This file
```

---

## Reproducibility

All necessary information is provided to repeat the experiments.  
- Dataset preprocessing and path configuration are described above.
- Model architecture and training procedure are defined in the source code and config files.
- Experiment settings and random seeds are specified.
- Results and plots can be reproduced using the provided scripts.

---
