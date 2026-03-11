# 16 — Overfitting and Underfitting (Why Models Fail)

## What Problem This Solves (Why It Matters in Real Life)

Machine Learning models can fail in two main ways:

- Model too simple → Poor predictions
- Model too complex → Memorizes data

Example:

Training Accuracy = 99%
Test Accuracy = 60%

Model memorized training data.

This is called overfitting.

Understanding overfitting and underfitting helps build better models.

---

## Simple Explanation (Plain English First)

### Underfitting

Model is too simple.

Cannot learn patterns.

Poor predictions.

---

### Overfitting

Model is too complex.

Memorizes training data.

Fails on new data.

---

### Good Model

Balanced model.

Learns patterns without memorizing.

---

## Technical Explanation (Accurate but Digestible)

### Underfitting

Low training accuracy.

Low test accuracy.

Model too simple.

---

### Overfitting

High training accuracy.

Low test accuracy.

Model memorizes training data.

---

### Good Fit

Good training accuracy.

Good test accuracy.

Model generalizes well.

---

## Real-World Analogy

Student studying for exam.

Underfitting:

Studies very little → Fails exam

Overfitting:

Memorizes answers → Different exam → Fails

Good Fit:

Understands concepts → Passes exams

---

## Step-by-Step Example

Compare models:

Simple model → Underfit

Balanced model → Good fit

Complex model → Overfit

---

## Code Example

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.DataFrame({
    "hours":[1,2,3,4,5,6,7,8],
    "result":[0,0,1,1,1,1,1,1]
})

X = data[["hours"]]
y = data["result"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=1
)

model1 = DecisionTreeClassifier(max_depth=1)

model1.fit(X_train,y_train)

print("Simple Train:",model1.score(X_train,y_train))
print("Simple Test:",model1.score(X_test,y_test))

model2 = DecisionTreeClassifier()

model2.fit(X_train,y_train)

print("Complex Train:",model2.score(X_train,y_train))
print("Complex Test:",model2.score(X_test,y_test))
```

---

## Common Mistakes Students Make

Trusting training accuracy.

Making models too complex.

Ignoring test accuracy.

---

## How This Appears in Interviews / Industry

Overfitting happens when a model performs well on training data but poorly on new data.

---

## Quick Visual Mental Model

Underfit:

Train 60%
Test 55%

Good Fit:

Train 90%
Test 88%

Overfit:

Train 99%
Test 60%

---

## Student Challenges

Concept Questions:

1. What is underfitting?
2. What is overfitting?
3. How detect overfitting?

Coding Task:

Train:

DecisionTreeClassifier(max_depth=1)

DecisionTreeClassifier(max_depth=5)

Compare results.

Debugging:

Fix:

model.score(X_train,y_test)

---

## Extension Challenge

Use cross validation to detect overfitting.

---

## Summary

Overfitting and underfitting describe common Machine Learning model failures.
