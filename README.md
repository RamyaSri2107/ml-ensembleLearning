# Titanic Survival Prediction using Ensemble Learning (From Scratch)

This project is a small machine learning assignment where I implemented and tested a few classic **ensemble learning algorithms** on the Kaggle Titanic dataset.

Instead of using scikit-learn models directly, the main goal here was to understand how these algorithms work internally by coding them myself:
- Decision Tree (entropy / information gain)
- Random Forest (bootstrap aggregation + voting)
- AdaBoost (boosting weak learners)

The models are evaluated on Titanic survival prediction using precision/recall/F1 metrics.

---

## Files in this repo

- `decision_tree.py`  
  My implementation of a Decision Tree classifier using entropy and information gain.

- `random_forest.py`  
  Random Forest built using my Decision Tree. Trains multiple trees on bootstrapped samples and predicts using majority vote.

- `adaboost.py`  
  AdaBoost implementation using a shallow Decision Tree (stump) as the weak learner and combining learners using weighted voting.

- `Evaluation.py`  
  End-to-end script that:
  1) loads Titanic `train.csv`
  2) preprocesses the data
  3) trains Decision Tree, Random Forest, and AdaBoost
  4) prints classification reports

- `ML_Assignment3_Report.docx`  
  Short write-up explaining the approach and results.

---

## Dataset

This project expects the Titanic dataset file:

- `train.csv` (from Kaggle Titanic dataset)

Place `train.csv` in the same folder as `Evaluation.py`.

---

## How to run

### 1) Install dependencies

```bash
pip install numpy pandas scikit-learn scipy
