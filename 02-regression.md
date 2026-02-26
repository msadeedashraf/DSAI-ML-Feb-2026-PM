# 02 — Predicting Numbers with Regression

## What Problem This Solves (Why It Matters in Real Life)

Many real-world problems require predicting numbers:

- House prices
- Salaries
- Exam scores
- Sales revenue
- Temperature

Regression solves problems where we want to predict a continuous number.

---

## Simple Explanation (Plain English First)

Regression means predicting a number from data.

Example:

Experience → Salary

1 → 40000
2 → 50000
3 → 60000
4 → 70000
5 → 80000

The model learns:

More experience → Higher salary

Prediction:

6 → ≈ 90000

---

## Technical Explanation (Accurate but Digestible)

Regression is used when:

Target variable is numeric.

Examples:

Price
Score
Income
Temperature

Model learns:

y = f(X)

X = Features
y = Numeric Target

Example:

Salary = f(Experience)

---

## Real-World Analogy

Imagine drawing a line through points on a graph.

The line helps estimate new values.

Regression automatically finds the best line.

---

## Step-by-Step Example

Goal:

Predict exam score using hours studied.

Hours → Score

1 → 50
2 → 55
3 → 65
4 → 70
5 → 80

Prediction:

6 → ≈ 90

---

## Code Example

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    "hours":[1,2,3,4,5],
    "score":[50,55,65,70,80]
})

X = data[["hours"]]
y = data["score"]

model = LinearRegression()
model.fit(X,y)

print(model.predict([[6]]))
```

---

## Visualization

```python
import matplotlib.pyplot as plt

plt.scatter(X,y)
plt.plot(X,model.predict(X))

plt.xlabel("Hours")
plt.ylabel("Score")

plt.show()
```

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

Regression is used to predict:

- Prices
- Revenue
- Demand
- Growth

---

## Quick Visual Mental Model

Input → Model → Number

Hours → Model → Score

Data → Best Fit Line → Prediction

---

## Student Challenges

Concept Questions:

1. What is regression?
2. Give 3 regression examples.
3. What type of output does regression produce?

Coding Task:

Create dataset:

Temperature → Ice Cream Sales

Train model and predict 30 degrees.

Debugging:

Fix:

model.fit(data["hours"],data["score"])

---

## Extension Challenge

Add feature:

Sleep Hours

Train model using:

Hours + Sleep

---

## Summary

Regression predicts numeric values using patterns learned from past data.
