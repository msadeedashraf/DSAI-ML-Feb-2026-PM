# 03 — Predicting Categories with Classification

## What Problem This Solves (Why It Matters in Real Life)

Many real-world problems require predicting categories instead of numbers.

Examples:

- Spam vs Not Spam emails
- Pass vs Fail students
- Fraud vs Legitimate transactions
- Sick vs Healthy patients
- Approved vs Rejected loans

Classification solves problems where we want to predict a category.

---

## Simple Explanation (Plain English First)

Classification means predicting a category from data.

Example:

Hours → Result

1 → Fail
2 → Fail
3 → Pass
4 → Pass
5 → Pass

Prediction:

6 → Pass

---

## Technical Explanation (Accurate but Digestible)

Classification is used when:

Target variable is categorical.

Examples:

Yes / No
Spam / Not Spam
Pass / Fail
Fraud / Normal

Model learns:

y = f(X)

X = Features
y = Category

---

## Binary Classification

Two categories:

Yes / No
0 / 1
True / False

Example:

Spam = 1
Not Spam = 0

---

## Multi-Class Classification

More than two categories:

Low
Medium
High

Example:

Grade A
Grade B
Grade C

---

## Real-World Analogy

Doctor diagnosing patients.

Symptoms = Features
Diagnosis = Target
Doctor = Model

---

## Step-by-Step Example

Goal:

Predict Pass or Fail from study hours.

Hours → Result

1 → 0
2 → 0
3 → 1
4 → 1
5 → 1

Prediction:

6 → 1

---

## Code Example

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    "hours":[1,2,3,4,5],
    "result":[0,0,1,1,1]
})

X = data[["hours"]]
y = data["result"]

model = LogisticRegression()

model.fit(X,y)

print(model.predict([[6]]))
```

---

## Probability Prediction

```python
model.predict_proba([[6]])
```

Example output:

[[0.10 , 0.90]]

Meaning:

10% Fail
90% Pass

---

## Common Mistakes Students Make

Confusing regression with classification.

Regression predicts numbers.

Classification predicts categories.

Using:

X = data["hours"]

Instead of:

X = data[["hours"]]

---

## How This Appears in Interviews / Industry

Classification is used for:

- Fraud detection
- Spam detection
- Customer churn
- Disease detection

---

## Quick Visual Mental Model

Input → Model → Category

Hours → Model → Pass

---

## Student Challenges

Concept Questions:

1. What is classification?
2. What is binary classification?
3. What is multi-class classification?

Coding Task:

Create dataset:

Age → Buy Product

Train model and predict 40.

Debugging:

Fix:

model.fit(data["hours"],data["result"])

---

## Extension Challenge

Add feature:

Attendance

Train model using:

Hours + Attendance

---

## Summary

Classification predicts categories using patterns learned from data.
