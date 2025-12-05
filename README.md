# AI-Driven Production Scheduling Lab

A hands-on mini lab that combines **Industrial Engineering** and **Machine Learning** on a classic problem:

> Single-machine scheduling with **Total Weighted Tardiness (TWT)** minimization.

The goal of this repository is to provide a **clean, reproducible and educational** codebase where you can:

- Model a realistic production scheduling problem,
- Compare classical dispatching rules (FIFO, SPT, EDD),
- Train a simple **AI meta-policy** that learns when to use which rule,
- Use it as a **portfolio project** or **teaching material** for AI-driven Industrial Engineering.

---

## 🎯 Project Overview

We consider a single machine processing a set of jobs.  
Each job has:

- a **processing time** (p_j),
- a **due date** (d_j),
- a **weight / priority** (w_j).

All jobs are available at time 0 and preemption is not allowed.  
The objective is to minimize the **Total Weighted Tardiness**:

min Σ_j w_j max(0, C_j − d_j)

This problem is NP-hard in general, so we rely on **heuristics** and then add an **AI layer** on top of them.

For a detailed mathematical description, see:

➡️ `docs/problem_description.md`

---

## 🧠 What Makes This “AI-Driven”?

Instead of directly solving the optimization problem, we:

1. Implement classical **dispatching rules**:
   - FIFO (First In First Out)
   - SPT (Shortest Processing Time)
   - EDD (Earliest Due Date)

2. Generate many synthetic scheduling **instances** (job sets).

3. For each instance:
   - Run all three rules,
   - Compute their **Total Weighted Tardiness**,
   - Label the instance with the **best rule** (the one with minimum TWT).

4. Summarize each instance with simple, interpretable **features** (number of jobs, processing time statistics, due-date tightness, etc.).

5. Train a **Random Forest classifier** that learns to predict  
   **“Which rule should I use for this system?”**

This gives us an **AI meta-policy** that:

- doesn’t replace operations research,
- but learns to **choose between heuristics adaptively**,  
- and often gets **very close to oracle performance**.

---

## 🧩 Main Features

- 📦 **Single-machine scheduling environment**
  - `Job` dataclass (processing time, due date, weight)
  - `SingleMachineEnv` with step-by-step simulation

- ⚙️ **Classical dispatching rules**
  - FIFO, SPT, EDD implemented as simple Python functions

- 📊 **Performance metrics**
  - Total Weighted Tardiness (TWT)
  - Total Tardiness
  - Average Flow Time
  - Makespan

- 🧬 **Scenario & dataset generation**
  - Synthetic job set generator
  - Meta-dataset generator for ML (features + best policy label)

- 🧠 **Machine Learning meta-policy**
  - Random Forest classifier
  - Jupyter notebook with full training & evaluation pipeline

- 📚 **Documentation**
  - Problem formulation in `docs/problem_description.md`
  - Notebooks showing experiments end-to-end

---

## 🏭 Industrial Engineering Context

This project sits at the intersection of:

- **Production Planning & Scheduling**
- **Operations Research**
- **Data Science & Machine Learning**
- **Industry 4.0 / 5.0**

It is designed to be:

- A **portfolio-ready project** for an Industrial Engineering / Operations student,
- A small **teaching example** for AI in manufacturing,
- A playground to explore **AI for decision-making** in production systems.

---

## 📂 Repository Structure

```text
.
├── data/                      # (optional) synthetic datasets / saved runs
├── docs/
│   └── problem_description.md # mathematical problem description
├── notebooks/
│   ├── 02_heuristics_baseline.ipynb      # classical rules comparison
│   └── 03_ml_policy_experiments.ipynb    # Random Forest meta-policy
├── src/
│   ├── env_single_machine.py  # Job & SingleMachineEnv
│   ├── heuristics.py          # FIFO, SPT, EDD rules
│   ├── metrics.py             # TWT, flow time, makespan, etc.
│   └── simulation.py          # generators & meta-dataset creation
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/cetinkayafatih/ai-production-scheduling-lab.git
cd ai-production-scheduling-lab

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` typically includes:

- numpy
- pandas
- matplotlib
- scikit-learn
- jupyter

---

## 🚀 How to Use

### 1) Run the heuristics baseline notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

- `notebooks/02_heuristics_baseline.ipynb`

In this notebook you can:

- Generate random job sets,
- Run FIFO / SPT / EDD on the same instance,
- Compare:
  - Total Weighted Tardiness
  - Average Flow Time
  - Makespan
- Plot bar charts to visually compare the policies.

---

### 2) Run the ML meta-policy experiments

Open:

- `notebooks/03_ml_policy_experiments.ipynb`

This notebook will:

1. Use `generate_meta_dataset(...)` to create many scheduling instances.
2. Summarize each instance into features.
3. Label each instance with the best rule (FIFO / SPT / EDD).
4. Split into train/test sets.
5. Train a **Random Forest classifier**.
6. Evaluate:
   - Classification accuracy (how often we pick the same rule as oracle),
   - Average TWT for:
     - Always FIFO
     - Always SPT
     - Always EDD
     - ML meta-policy
     - Oracle (best-of-three per instance)
7. Plot a comparison chart of average TWT across these policies.

---

## 📊 Example Results (Illustrative)

On a sample run with **600 synthetic instances**, we observed:

- Static FIFO policy average TWT: ≈ 2300+
- Static SPT policy average TWT: ≈ 1500+
- Static EDD policy average TWT: ≈ 2100+
- **ML meta-policy average TWT:** very close to the best static rule and near-oracle performance
- Oracle policy (best of the three per instance): lowest achievable TWT

The key insight:

> A simple Random Forest meta-policy can **learn when to use which heuristic**, and approach oracle performance **without solving the scheduling problem exactly**.

You can reproduce and update these numbers from the ML notebook and then adjust this section with your actual outputs.

---

## 🛣️ Roadmap & Ideas

Potential extensions for future work:

- Add more dispatching rules:
  - WSPT, ATC, CR, etc.
- Learn a **custom priority index** instead of choosing from a fixed set.
- Implement a **Reinforcement Learning agent** that directly chooses the next job.
- Introduce:
  - Stochastic processing times,
  - Release dates,
  - Machine breakdowns,
  - Multi-machine systems.
- Export trained meta-policy for use in a web app or production simulator.

---

## 🤝 How to Use / Cite

If you use or extend this project in your work, you can reference it as:

*AI-Driven Production Scheduling Lab – Single-machine scheduling with an ML-based meta-policy selecting between classical dispatching rules (FIFO, SPT, EDD).*

Feel free to fork, experiment, and adapt it to your own scenarios.

---

## 👤 Author

**Fatih Çetinkaya**  
Industrial Engineering student exploring **AI-driven decision making** in production systems.

You can reach out for:

- Collaboration ideas,
- Extensions of this project,
- Using this lab as a teaching/demo tool.

---

## 📄 License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.
