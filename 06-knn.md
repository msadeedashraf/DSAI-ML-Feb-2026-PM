# 06 — k-Nearest Neighbors (KNN)

## What Problem This Solves (Why It Matters in Real Life)

Many decisions in real life are based on similar past cases.

Examples:

- Recommending movies based on similar users
- Predicting house prices based on nearby houses
- Suggesting products based on similar customers
- Diagnosing patients with similar symptoms

KNN solves problems where similar inputs produce similar outputs.

---

## Simple Explanation (Plain English First)

KNN means:

Make a prediction by looking at the most similar examples.

Example:

Hours → Result

1 → Fail
2 → Fail
3 → Pass
4 → Pass
5 → Pass

Prediction:

6 → Pass

The closest examples vote.

---

## Technical Explanation (Accurate but Digestible)

KNN finds the closest data points.

k = number of neighbors.

Example:

k = 3

Model finds the 3 closest points.

Prediction is based on majority vote.

For regression, prediction is the average.

---

## Real-World Analogy

Choosing a restaurant.

Ask 5 friends.

If most recommend it, you go.

Friends = Neighbors
Decision = Prediction

---

## Step-by-Step Example

Goal:

Predict Pass/Fail from study hours.

Dataset:

Hours → Result

1 → 0
2 → 0
3 → 1
4 → 1
5 → 1

Choose:

k = 3

Predict:

6 → 1

---

## Code Example (Classification)

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = pd.DataFrame({
    "hours":[1,2,3,4,5],
    "result":[0,0,1,1,1]
})

X = data[["hours"]]
y = data["result"]

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X,y)

print(model.predict([[6]]))
```

---

## Code Example (Regression)

```python
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

data = pd.DataFrame({
    "hours":[1,2,3,4,5],
    "score":[50,55,65,70,80]
})

X = data[["hours"]]
y = data["score"]

model = KNeighborsRegressor(n_neighbors=3)

model.fit(X,y)

print(model.predict([[6]]))
```

---

## Common Mistakes Students Make

Thinking KNN learns formulas.

KNN stores data instead.

Choosing k too large.

Choosing k = 1.

Forgetting double brackets:

X = data[["hours"]]

---

## How This Appears in Interviews / Industry

KNN predicts values using closest examples.

Used in:

- Recommendation systems
- Pattern recognition
- Similarity search

---

## Quick Visual Mental Model

New Point → Closest Points → Vote → Prediction

Example:

6 Hours → (5,4,3) → Pass

---

## Student Challenges

Concept Questions:

1. What does KNN stand for?
2. What is k?
3. What happens if k is too large?

Coding Task:

Create dataset:

Age → Buy Product

Train KNN.

Predict age 40.

Debugging:

Fix:

KNeighborsClassifier(3)

---

## Extension Challenge

Try:

k = 1
k = 3
k = 5

Compare predictions.

---

## Summary

KNN predicts values by looking at the most similar examples in the dataset.
