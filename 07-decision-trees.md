# 07 — Decision Trees (Learning Using If–Else Rules)

## What Problem This Solves (Why It Matters in Real Life)

Many real-life decisions follow step-by-step rules.

Examples:

- Loan approval decisions
- Medical diagnosis
- Hiring decisions
- Credit scoring
- Customer targeting

Decision Trees allow computers to learn these rules automatically.

---

## Simple Explanation (Plain English First)

Decision Trees work like a flowchart.

Example:

If Hours > 2.5 → Pass
Else → Fail

Prediction:

1 hour → Fail
4 hours → Pass

Decision Trees learn rules from data.

---

## Technical Explanation (Accurate but Digestible)

Decision Trees split data into smaller groups.

Each split is based on a condition.

Example:

Hours ≤ 2.5
Hours > 2.5

Each branch leads to a prediction.

---

## Tree Structure

Root Node:

First decision.

Branches:

Possible paths.

Leaf Nodes:

Final predictions.

---

## Classification Trees

Predict categories.

Examples:

Pass / Fail
Spam / Not Spam

---

## Regression Trees

Predict numbers.

Examples:

House Price
Score
Salary

---

## Real-World Analogy

Doctor diagnosis.

Fever?

Yes → Check cough
No → Healthy

Each question reduces uncertainty.

---

## Step-by-Step Example

Goal:

Predict Pass/Fail using study hours.

Dataset:

1 → 0
2 → 0
3 → 1
4 → 1
5 → 1

Tree learns:

Hours > 2.5 → Pass
Hours ≤ 2.5 → Fail

Prediction:

6 → Pass

---

## Code Example (Classification)

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.DataFrame({
    "hours":[1,2,3,4,5],
    "result":[0,0,1,1,1]
})

X = data[["hours"]]
y = data["result"]

model = DecisionTreeClassifier()

model.fit(X,y)

print(model.predict([[6]]))
```

---

## Visualizing the Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plot_tree(model)

plt.show()
```

---

## Code Example (Regression)

```python
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

data = pd.DataFrame({
    "hours":[1,2,3,4,5],
    "score":[50,55,65,70,80]
})

X = data[["hours"]]
y = data["score"]

model = DecisionTreeRegressor()

model.fit(X,y)

print(model.predict([[6]]))
```

---

## Common Mistakes Students Make

Trees can become too large.

Deep trees memorize data.

Better:

max_depth = 3

Forgetting double brackets:

X = data[["hours"]]

---

## How This Appears in Interviews / Industry

Decision Trees use learned if–else rules.

Used in:

- Credit approval
- Medical diagnosis
- Risk prediction

---

## Quick Visual Mental Model

Hours > 2.5 ?

Yes → Pass
No → Fail

---

## Student Challenges

Concept Questions:

1. What is a Decision Tree?
2. What is a root node?
3. What is a leaf node?

Coding Task:

Create dataset:

Age → Buy Product

Train Decision Tree.

Predict age 40.

Debugging:

Fix:

DecisionTreeClassifier.fit(X,y)

---

## Extension Challenge

Try:

max_depth = 1
max_depth = 3
max_depth = 5

Compare predictions.

---

## Summary

Decision Trees make predictions using learned if–else rules.
