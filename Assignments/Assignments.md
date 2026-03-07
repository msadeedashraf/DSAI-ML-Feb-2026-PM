# Machine Learning Case Study Assignments

## Applying ML Concepts to Real-World Problems

These assignments are designed to help you **apply machine learning concepts in real-world scenarios**.
You will build models, analyze results, and explain the reasoning behind your choices using proper **machine learning terminology**.

Across these assignments you will practice and demonstrate understanding of:

* Features
* Target variables
* Training vs Testing data
* Model selection
* Hyperparameters
* Overfitting and Underfitting
* Scaling
* Model evaluation (accuracy, confusion matrix, error)
* Classification vs Regression problems

---

# Case Study 1 — Netflix Viewing Behavior

## Scenario

You are part of the **Netflix Data Science team**.

Netflix wants to predict:

1. **How many minutes a user will watch**
2. **Whether a user will binge-watch a show**

Dataset generator:

```python
(X_minutes, y_minutes), (X_binge, y_binge) = make_netflix()
```

Features include:

* `clicks`
* `is_weekend`

Targets:

* `minutes` (total watch time)
* `binge` (0 = not binge watching, 1 = binge watching)

---

## Task 1 — Identify the Problem Type

Explain the following:

1. Is predicting **minutes watched** a **regression or classification** problem?
2. Is predicting **binge watching** a **regression or classification** problem?
3. Explain your reasoning.

---

## Task 2 — Features and Target

Identify the following:

Features (X)

```
?
```

Target variables (y)

```
?
```

Explain the difference between **features and targets**.

---

## Task 3 — Train/Test Split

Split the data using:

```python
train_test_split()
```

Explain:

* Why we split data into **training and testing sets**
* What problem this helps prevent

---

## Task 4 — Build Two Models

Train two models for predicting **minutes watched**:

1. Linear Regression
2. KNN Regressor

Explain:

* Which model performed better
* Why the results might differ

---

## Task 5 — KNN Hyperparameter Experiment

Train KNN with different values:

```
k = 1
k = 5
k = 10
```

Explain:

* How changing **k** affects predictions
* Which value worked best
* How this relates to **overfitting and underfitting**

---

## Task 6 — Make a Prediction

Predict watch time for a user with:

```
clicks = 9
is_weekend = 1
```

Explain the meaning of the prediction.

---

## Reflection

Why might **KNN work well for recommendation systems** like Netflix?

---

# Case Study 2 — Hospital Risk Prediction

## Scenario

You are working with a **hospital analytics team**.

The goal is to identify patients who are at **high medical risk**.

Dataset:

```python
X, y = make_healthcare()
```

Features:

* `age`
* `fever`
* `oxygen`

Target:

```
risk (0 = low risk, 1 = high risk)
```

---

## Task 1 — Identify the ML Problem

Explain:

Is this problem **classification or regression**?

Why?

---

## Task 2 — Features and Target

Explain:

* What are **features**?
* What is the **target variable**?

Provide examples from this dataset.

---

## Task 3 — Data Scaling

Before using KNN, scale the dataset.

Explain:

Why scaling is important for **distance-based models**.

---

## Task 4 — Model Comparison

Train two models:

```
Logistic Regression
KNN Classifier
```

Compare:

* Accuracy
* Predictions

Explain why one model may perform better.

---

## Task 5 — Confusion Matrix

Print the confusion matrix.

Explain the meaning of:

```
True Positive
True Negative
False Positive
False Negative
```

Write explanations in **plain English**.

Example:

```
False Negative = sick patient predicted as healthy
```

---

## Task 6 — Real-World Impact

Which mistake is more dangerous in healthcare?

```
False Positive
False Negative
```

Explain your reasoning.

---

# Case Study 3 — Credit Card Fraud Detection

## Scenario

You are working with a **bank fraud detection team**.

The goal is to predict whether a transaction is fraudulent.

Dataset:

```python
X, y = make_fraud()
```

Features:

* `amount`
* `is_international`
* `night`

Target:

```
fraud (0 = legitimate transaction, 1 = fraud)
```

---

## Task 1 — Class Imbalance

Run:

```python
y.value_counts()
```

Explain:

* What class imbalance is
* Why fraud detection datasets often have imbalanced data

---

## Task 2 — Model Training

Train two models:

```
Logistic Regression
KNN Classifier
```

Print:

* Accuracy
* Confusion Matrix

---

## Task 3 — Evaluate the Models

Explain:

Why **accuracy alone may not be enough** for evaluating fraud detection.

Example:

```
A model could predict all transactions as legitimate
and still achieve high accuracy.
```

---

## Task 4 — Experiment with KNN

Train KNN with different values of k:

```
k = 3
k = 7
k = 15
```

Explain:

* How predictions change
* Which value performs best

---

## Task 5 — How KNN Detects Similar Transactions

Explain how KNN determines similarity between transactions.

Example explanation:

```
Transactions with similar amounts,
international status, and time of day
may behave similarly.
```

---

## Task 6 — Real-World Consequences

Explain the impact of the following mistakes:

```
False Positive → legitimate transaction flagged as fraud
False Negative → fraudulent transaction allowed
```

Which mistake is worse for a bank?

Explain your reasoning.

---

# Assignment Deliverables

Submit the following:

1. **Jupyter Notebook**

   * Code
   * Model training
   * Predictions
   * Evaluation metrics

2. **Short Report (1–2 pages)**

Explain:

* Problem type
* Features vs target
* Model choice
* Hyperparameters
* Model performance
* Real-world implications

---

# Evaluation Criteria

Your submission will be graded based on:

| Criteria                      | Weight |
| ----------------------------- | ------ |
| Concept understanding         | 30%    |
| Correct use of ML terminology | 20%    |
| Model implementation          | 25%    |
| Analysis and explanation      | 15%    |
| Clarity of report             | 10%    |

---

# Learning Goals

After completing these assignments you should be able to:

* Identify **classification vs regression problems**
* Select appropriate **machine learning models**
* Understand **training vs testing data**
* Interpret **model evaluation metrics**
* Explain the impact of **model mistakes in real-world systems**

---

# Important Note

Do not simply copy code patterns.

Your explanations should demonstrate that you understand:

```
how machine learning models work
why they are used
and what their predictions mean in real-world systems.
```
