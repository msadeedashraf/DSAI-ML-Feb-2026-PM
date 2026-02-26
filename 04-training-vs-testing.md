# 04 — Training vs Testing (How Models Really Learn)

## What Problem This Solves (Why It Matters in Real Life)

A Machine Learning model can appear perfect during training but fail in the real world.

Example:

Practice test score = 100%
Real test score = 50%

This happens when a model memorizes instead of learning.

Training vs Testing helps determine whether a model works on new data.

---

## Simple Explanation (Plain English First)

Machine Learning uses two types of data.

Training Data:
Used to teach the model.

Testing Data:
Used to evaluate the model.

Key idea:

Never test on the same data used for training.

---

## Technical Explanation (Accurate but Digestible)

Dataset is divided into:

Training Set
Testing Set

Typical split:

80% Training
20% Testing

Training:

model.fit(X_train,y_train)

Testing:

model.predict(X_test)

---

## Real-World Analogy

Learning to drive.

Training:
Practice in parking lot.

Testing:
Driving on real roads.

Testing proves real skill.

---

## Step-by-Step Example

Goal:

Predict exam score from hours studied.

Split dataset into:

Training data
Testing data

Train model on training data.

Test model on testing data.

---

## Code Example

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    "hours":[1,2,3,4,5,6,7,8],
    "score":[50,55,65,70,80,85,90,95]
})

X = data[["hours"]]
y = data["score"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=1
)

model = LinearRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print(predictions)
```

---

## Common Mistakes Students Make

Training and testing on same data:

model.fit(X,y)
model.predict(X)

This produces misleading results.

Forgetting random_state:

random_state makes results consistent.

Using very small test sets.

---

## How This Appears in Interviews / Industry

Typical question:

Why split data?

Answer:

To evaluate performance on unseen data.

---

## Quick Visual Mental Model

Full Dataset → Split → Train + Test

Training → Learning

Testing → Checking

---

## Student Challenges

Concept Questions:

1. What is training data?
2. What is testing data?
3. Why not test on training data?

Coding Task:

Create dataset:

Hours → Score

Split 70/30

Train model.

Debugging:

Fix:

model.fit(X,y)
model.predict(X)

---

## Extension Challenge

Try test_size values:

0.1
0.3
0.5

Compare results.

---

## Summary

Training teaches the model. Testing checks if the model learned patterns that work on new data.
