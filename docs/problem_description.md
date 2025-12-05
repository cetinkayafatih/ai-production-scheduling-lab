# Single-Machine Scheduling with Total Weighted Tardiness

## 1. Problem Definition

We consider a **single-machine scheduling** problem with a finite set of jobs:

- Set of jobs: \( J = \{1, 2, \dots, n\} \)
- For each job \( j \in J \):
  - Processing time: \( p_j \)
  - Due date: \( d_j \)
  - Weight (priority): \( w_j \)

All jobs are available at time 0, and preemption is not allowed.  
The machine can process at most one job at a time.

A **schedule** is a permutation of the jobs, which induces start times \( S_j \) and completion times \( C_j \) on the single machine.

The **tardiness** of job \( j \) is defined as:
\[
T_j = \max(0, C_j - d_j)
\]

The **weighted tardiness** of job \( j \) is:
\[
w_j T_j
\]

The objective is to find a job sequence \(\pi\) that minimizes the **total weighted tardiness**:
\[
\min \sum_{j \in J} w_j T_j
\]

This problem is known to be **NP-hard** in general.

---

## 2. Dispatching Rules

Instead of solving the optimization problem exactly (which is expensive), we focus on **dispatching rules** that select the next job to process based on simple priority indices:

- **FIFO (First In First Out)**  
  Jobs are processed in the order they arrive / are indexed.

- **SPT (Shortest Processing Time)**  
  At each decision point, select the job with the smallest processing time \( p_j \).

- **EDD (Earliest Due Date)**  
  At each decision point, select the job with the earliest due date \( d_j \).

These rules are simple, interpretable, and widely used in industrial settings.

---

## 3. AI-Driven Meta-Policy

In this project, we build an **AI-driven meta-policy** that, for a given instance (set of jobs), learns to **select the most promising dispatching rule** (FIFO, SPT, or EDD).

### 3.1 Features

Each instance is summarized with system-level features, for example:

- Number of jobs \( n \)
- Total processing time \( \sum_j p_j \)
- Mean and standard deviation of processing times
- Mean and standard deviation of due dates
- Mean and standard deviation of weights
- A simple due-date tightness indicator

### 3.2 Labels

For each instance we:

1. Run all three dispatching rules (FIFO, SPT, EDD).
2. Compute the total weighted tardiness for each rule.
3. Label the instance with the **best rule**, i.e. the one with the smallest total weighted tardiness.

This yields a dataset of \((\text{features}, \text{best\_policy})\) pairs.

### 3.3 Model

We use a **Random Forest classifier** to map from features to the best policy label.  
On synthetic data, this meta-policy achieves high accuracy and total weighted tardiness close to the oracle "best of three" policy.

---

## 4. Extensions and Future Work

Possible extensions include:

- Adding more dispatching rules (e.g. WSPT, ATC).
- Predicting a **custom priority index** instead of selecting from a fixed set of rules.
- Training a **Reinforcement Learning agent** that directly selects the next job at each decision point.
- Incorporating stochastic processing times or release dates into the environment.