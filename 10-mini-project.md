# 10 — Mini Project (Building Your First Machine Learning System)

## What Problem This Solves (Why It Matters in Real Life)

Real Machine Learning projects combine multiple steps:

- Loading data
- Preparing data
- Training models
- Evaluating models
- Making predictions

This project demonstrates a complete Machine Learning workflow.

---

## Simple Explanation (Plain English First)

This project predicts:

Student Pass or Fail

Based on:

Hours studied
Attendance

Steps:

1. Create dataset
2. Prepare data
3. Train model
4. Test model
5. Make predictions

---

## Technical Explanation (Accurate but Digestible)

Machine Learning pipeline:

Data → Features → Split → Model → Evaluation → Prediction

Steps:

1 Load dataset
2 Select features and target
3 Train-test split
4 Train model
5 Evaluate model
6 Make predictions

---

## Real-World Analogy

Building a house:

Design
Materials
Construction
Inspection
Move in

Machine Learning:

Data
Training
Testing
Evaluation
Prediction

---

## Step-by-Step Example

Goal:

Predict student results.

Dataset:

Hours | Attendance | Result

1 | 50 | 0
2 | 60 | 0
3 | 70 | 1
4 | 80 | 1
5 | 90 | 1

Features:

Hours
Attendance

Target:

Result

---

## Full Code Example

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.DataFrame({

"hours":[1,2,3,4,5,6,7,8],

"attendance":[50,60,70,80,90,85,95,90],

"result":[0,0,1,1,1,1,1,1]

})

X = data[["hours","attendance"]]
y = data["result"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=1
)

model = LogisticRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,predictions))

print(model.predict([[6,85]]))
```

---

## Common Mistakes Students Make

Skipping the train/test split.

Mixing features and target.

Forgetting evaluation.

Wrong column names.

---

## How This Appears in Interviews / Industry

Typical task:

Load data → Train model → Evaluate → Predict

---

## Quick Visual Mental Model

Dataset → Features → Split → Model → Evaluate → Predict

---

## Student Challenges

Concept Questions:

1. What is the ML pipeline?
2. Why split data?
3. Why evaluate models?

Coding Task:

Add feature:

Sleep Hours

Train model again.

Debugging:

Fix:

model.fit(data)

---

## Extension Challenge

Train:

Logistic Regression
KNN
Decision Tree

Compare accuracy.

---

## Summary

A Machine Learning project involves preparing data, training models, evaluating performance, and making predictions.
