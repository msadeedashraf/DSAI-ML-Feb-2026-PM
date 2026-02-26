# 05 — Model Evaluation (How Good Is Your Model?)

## What Problem This Solves (Why It Matters in Real Life)

After building a Machine Learning model we must determine:

Is the model good or bad?

A model that makes predictions is not necessarily useful.

Examples:

- House price predictions may be far off
- Fraud models may miss criminals
- Medical models may misdiagnose patients

Model evaluation tells us how accurate predictions are.

---

## Simple Explanation (Plain English First)

After training a model we compare:

Predicted Values vs Real Values

Example:

Real → Predicted

65 → 60
80 → 78
90 → 92

Difference = Error

Small error = Good model
Large error = Bad model

---

## Technical Explanation (Accurate but Digestible)

Different metrics exist for different problems.

Regression Metrics:

MAE — Mean Absolute Error
MSE — Mean Squared Error

Classification Metric:

Accuracy

---

### MAE

Average difference between predicted and real values.

Lower MAE is better.

---

### MSE

Average squared difference between predicted and real values.

Large errors count more.

Lower MSE is better.

---

### Accuracy

Percentage of correct predictions.

Example:

8 correct out of 10

Accuracy = 80%

Higher accuracy is better.

---

## Real-World Analogy

Weather prediction.

Prediction:

25°C

Actual:

27°C

Error:

2°C

Small error means good prediction.

---

## Step-by-Step Example

Goal:

Evaluate exam score predictions.

Train model.

Make predictions.

Compare predicted vs real values.

Calculate error.

---

## Code Example (Regression)

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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

print("MAE:",mean_absolute_error(y_test,predictions))
print("MSE:",mean_squared_error(y_test,predictions))
```

---

## Code Example (Classification)

```python
from sklearn.metrics import accuracy_score

real = [1,0,1,1,0]
predicted = [1,0,1,0,0]

print(accuracy_score(real,predicted))
```

---

## Common Mistakes Students Make

Thinking predictions automatically mean success.

Looking at only one prediction.

Expecting 100% accuracy.

Confusing MAE and MSE.

---

## How This Appears in Interviews / Industry

Regression is evaluated using:

MAE
MSE

Classification is evaluated using:

Accuracy

---

## Quick Visual Mental Model

Real Values → Predictions → Compare → Error

Prediction − Reality = Error

---

## Student Challenges

Concept Questions:

1. What is model evaluation?
2. What is MAE?
3. What is MSE?
4. What is accuracy?

Coding Task:

Create dataset:

Temperature → Ice Cream Sales

Train model and calculate MAE and MSE.

Debugging:

Fix:

mean_absolute_error(predictions,y_test)

---

## Extension Challenge

Try different test_size values.

Compare MAE results.

---

## Summary

Model evaluation measures how well a machine learning model performs on real data.
