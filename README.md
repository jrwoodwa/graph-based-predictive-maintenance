# Predictive Maintenance: Forecasting Machine State Changes from Sensor Graphs

## ⚠️ The Problem

Modern machines have dozens of sensors, producing complex and noisy data — but what operators really need is simple:

> **“Tell me *when* the machine is about to change states — before it does.”**

Whether it’s shifting from operational to recovery or heading toward a failure, timing matters. Acting too late means downtime. Acting too early wastes effort.

Unfortunately, most systems rely on anomaly scores or rule-based alarms that are reactive, vague, or just plain noisy.

---

## 🧠 The Trap: Deep Learning Isn't Always the Answer

Neural networks — especially deep graph models — are powerful but often overkill:

* **Hard to interpret** — Why did it predict failure? No one knows.  
* **Data-hungry** — Struggle when you don’t have millions of labeled examples.  
* **Black box behavior** — Not ideal in safety-critical or regulated environments.  
* **Difficult to debug** — You don’t know if it failed because of the graph structure, the features, or randomness.  

Many teams start with neural networks… only to realize they need something more transparent, reliable, and lightweight.

---

## 🌳 The Solution: Tree-Based Graph Learning (TREE-G)

We use **TREE-G**, a non-neural method that applies decision tree logic to graph-structured data. It’s:

* **Interpretable** — Shows *why* a prediction was made  
* **Efficient** — Handles small and medium datasets without overfitting  
* **Robust** — Works with missing sensor data (no imputation needed)  
* **Precise** — Predicts *how much time* until the next machine state change  

Each machine cycle is represented as a **graph** of sensors, with edges based on correlation and node and edge features derived from moving averages, clustering, and other methods.

The model learns to **regress the time remaining until a state change**, offering clear lead time for intervention.

---

## 🛠️ What’s in the Project

* Graph construction from time series sensor data  
* Feature engineering on nodes, edges, and clusters  
* Multiple clustering methods (Louvain, spectral, k-means)  
* Regression modeling with TREE-G  
* Hyperparameter tuning and cross-validation  
* Output: plain, interpretable forecasts of time-to-transition  

---

## 💡 Why This Matters

This project bridges practical maintenance needs with real structure in data:

* Forecasts *when* — not just *if* — a machine will change states  
* Leverages graph dynamics, not just raw sensor values  
* Gives teams insight they can trust and act on  

Instead of “AI” that no one understands, this is **machine learning that works like an engineer thinks**.
