# 08 — Logistic Regression (Predicting Probability)

## What Problem This Solves (Why It Matters in Real Life)

Many real-world decisions are based on probability.

Examples:

- Probability a customer will buy
- Probability an email is spam
- Probability a student will pass
- Probability a patient has a disease
- Probability a loan will default

Logistic Regression predicts probability and converts it into categories.

---

## Simple Explanation (Plain English First)

Logistic Regression predicts probability.

Example:

Probability of passing:

0.90

Meaning:

90% chance of passing.

Decision rule:

If probability > 0.5 → Pass
Else → Fail

---

## Technical Explanation (Accurate but Digestible)

Logistic Regression is a classification algorithm.

It predicts probability values between:

0 and 1

Example:

0.10 → Low chance
0.50 → Medium chance
0.90 → High chance

Model:

y = f(X)

X = Features
y = Probability

Decision rule:

Probability > 0.5 → Class 1
Probability ≤ 0.5 → Class 0

---

## Why It Is Called Regression

Logistic Regression uses a regression-style equation to calculate probability.

Category prediction comes after probability.

---

## Real-World Analogy

Weather forecast.

70% chance of rain.

Decision:

Take umbrella.

Probability first.
Decision second.

---

## Step-by-Step Example

Goal:

Predict Pass/Fail from study hours.

Dataset:

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

X = data[[ "hours" ]]
y = data["result"]

model = LogisticRegression()

model.fit(X,y)

print(model.predict([[6]]))
```

---

## Probability Example

```python
model.predict_proba([[6]])
```

Example output:

[[0.05 , 0.95]]

Meaning:

5% Fail
95% Pass

---

## Common Mistakes Students Make

Thinking Logistic Regression predicts numbers.

It predicts categories.

Not understanding probability.

Forgetting double brackets:

X = data[[ "hours" ]]

---

## How This Appears in Interviews / Industry

Logistic Regression predicts probabilities and categories.

Used in:

- Fraud detection
- Customer churn
- Medical diagnosis

---

## Quick Visual Mental Model

Input → Probability → Decision

Hours → 0.9 → Pass

---

## Student Challenges

Concept Questions:

1. What does Logistic Regression predict first?
2. What range are probabilities?
3. What threshold is commonly used?

Coding Task:

Create dataset:

Age → Buy Product

Train Logistic Regression.

Predict age 40.

Debugging:

Fix:

model.fit(data["hours"],data["result"])

---

## Extension Challenge

Try thresholds:

0.3
0.5
0.7

Compare predictions.

---

## Summary

Logistic Regression predicts probabilities and converts them into categories.
