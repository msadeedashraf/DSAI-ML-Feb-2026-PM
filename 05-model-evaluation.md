# 05 --- Model Evaluation (How Good Is Your Model?)

## Why This Chapter Exists

In the previous chapter we trained a machine learning model and used it
to make predictions.

Example:

``` python
predictions = model.predict(X_test)
```

The model gives us answers.

But an important question remains:

**Are those answers good?**

A machine learning model that produces predictions is **not
automatically useful**.

Examples:

-   A house price model may predict **\$800,000** when the real price is
    **\$500,000**
-   A fraud detection model may **miss criminals**
-   A medical model may **misdiagnose patients**

So after training a model, we must measure **how accurate the
predictions are**.

This process is called:

**Model Evaluation**

------------------------------------------------------------------------

# The Core Idea

Machine learning evaluation always compares two things:

**Predicted Values vs Real Values**
Example:

| Real Score | Predicted Score |
| ---------- | --------------- |
| 65         | 60              |
| 80         | 78              |
| 90         | 92              |


Example:

Difference between them is called **error**.

Small error → Good model\
Large error → Poor model

------------------------------------------------------------------------

# The Machine Learning Pipeline So Far

At this point students should see the full workflow:

    Collect Data
          ↓
    Split Data (Train / Test)
          ↓
    Train Model
          ↓
    Make Predictions
          ↓
    Evaluate Predictions

Chapter 5 focuses on the **last step**.

------------------------------------------------------------------------

# Two Types of Machine Learning Problems

Before choosing evaluation metrics, we must understand **what type of
problem we are solving**.

Machine learning problems usually fall into two categories.

------------------------------------------------------------------------

## Regression

Predicting a **number**.

Examples:

-   House price
-   Temperature
-   Sales revenue
-   Exam score

Example prediction:

    Real: 85
    Predicted: 82

------------------------------------------------------------------------

## Classification

Predicting a **category**.

Examples:

-   Fraud / Not Fraud
-   Pass / Fail
-   Spam / Not Spam
-   Disease / No Disease

Example prediction:

    Real: Fraud
    Predicted: Not Fraud

Because these problems are different, **their evaluation metrics are
also different**.

------------------------------------------------------------------------

# Regression Evaluation

Regression models predict **numbers**, so evaluation focuses on **how
far predictions are from real values**.

Two common metrics are used.

------------------------------------------------------------------------

## MAE --- Mean Absolute Error

MAE calculates the **average difference** between predictions and real
values.

Example errors:

    Real: 70   Predicted: 65   Error: 5
    Real: 80   Predicted: 78   Error: 2
    Real: 90   Predicted: 88   Error: 2

Average error:

    MAE = (5 + 2 + 2) / 3 = 3

Meaning the model is off by **about 3 points on average**.

Lower MAE is better.

------------------------------------------------------------------------

## MSE --- Mean Squared Error

MSE squares the errors before averaging.

Why?

Because **large mistakes should be punished more**.

Example:

    Errors: 5, 2, 2

    Squared errors:
    25, 4, 4

    MSE = (25 + 4 + 4) / 3

Large errors become **much more noticeable**.

Lower MSE is better.

------------------------------------------------------------------------

# Regression Code Example

``` python
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

------------------------------------------------------------------------

# Classification Evaluation

Classification models predict **categories**, not numbers.

So instead of measuring distance between numbers, we measure:

**How many predictions were correct?**

------------------------------------------------------------------------

## Accuracy

Accuracy measures the percentage of correct predictions.

Example:

    Real:      [1,0,1,1,0]
    Predicted: [1,0,1,0,0]

Correct predictions:

    4 out of 5

Accuracy:

    4 / 5 = 0.8 = 80%

Higher accuracy is better.

------------------------------------------------------------------------

# Classification Code Example

``` python
from sklearn.metrics import accuracy_score

real = [1,0,1,1,0]
predicted = [1,0,1,0,0]

print("Accuracy:", accuracy_score(real,predicted))
```

------------------------------------------------------------------------

# Real-World Analogy

Think about **weather forecasting**.

Prediction:

    25°C

Actual temperature:

    27°C

Error:

    2°C

If forecasts are always within **1--2°C**, the model is good.

If forecasts are often **10°C off**, the model is poor.

Machine learning evaluation works the same way.

------------------------------------------------------------------------

# Common Mistakes Students Make

-   Thinking predictions automatically mean success
-   Looking at only one prediction
-   Expecting 100% accuracy
-   Confusing regression metrics with classification metrics

------------------------------------------------------------------------

# Student Challenges

## Concept Questions

1.  What is model evaluation?
2.  Why do we need evaluation metrics?
3.  What is MAE?
4.  What is MSE?
5.  What is accuracy?

------------------------------------------------------------------------

## Coding Task

Create a dataset:

Temperature → Ice Cream Sales

Train a regression model and calculate:

-   MAE
-   MSE

------------------------------------------------------------------------

## Debugging Task

Fix the error:

``` python
mean_absolute_error(predictions,y_test)
```

------------------------------------------------------------------------

# Extension Challenge

Try different values of:

    test_size

Observe how the **evaluation results change**.

------------------------------------------------------------------------

# Chapter Summary

Model evaluation tells us **how well a machine learning model
performs**.

Regression models use:

-   MAE
-   MSE

Classification models use:

-   Accuracy

Evaluation helps determine whether a model is **useful, reliable, or
needs improvement**.
