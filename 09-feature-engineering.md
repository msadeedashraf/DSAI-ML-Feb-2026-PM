# 09 — Feature Engineering (Preparing Data for Machine Learning)

## What Problem This Solves (Why It Matters in Real Life)

Real-world data is messy and cannot be used directly in Machine Learning.

Common problems:

- Missing values
- Text instead of numbers
- Different scales
- Irrelevant columns
- Incorrect values

Feature Engineering prepares data so models can learn correctly.

---

## Simple Explanation (Plain English First)

Feature Engineering means preparing data so Machine Learning can understand it.

This includes:

- Cleaning data
- Fixing missing values
- Converting text to numbers
- Selecting useful columns
- Scaling numbers

Better features produce better predictions.

---

## Technical Explanation (Accurate but Digestible)

Feature Engineering includes:

Handling Missing Values
Encoding Categories
Scaling Features
Feature Selection

---

### Handling Missing Values

Example:

Income = ?

Solutions:

Replace with average
Remove row
Replace with median

---

### Encoding Categories

Machine Learning requires numbers.

Example:

Toronto
Montreal
Vancouver

Convert to numbers.

---

### Feature Scaling

Features may have different ranges.

Example:

Age = 25
Income = 70000

Scaling balances features.

---

### Feature Selection

Some columns are useless.

Example:

Student ID

Remove irrelevant columns.

---

## Real-World Analogy

Cooking a meal.

Ingredients must be prepared before cooking.

Raw data must be prepared before Machine Learning.

---

## Step-by-Step Example

Goal:

Prepare dataset.

Dataset:

Age | Income | City

Fix missing values.

Encode City.

Train model.

---

## Code Example (Missing Values)

```python
import pandas as pd

data = pd.DataFrame({
    "age":[25,30,35],
    "income":[40000,None,70000]
})

data["income"].fillna(
    data["income"].mean(),
    inplace=True
)

print(data)
```

---

## Code Example (Encoding)

```python
data = pd.DataFrame({
    "city":["Toronto","Montreal","Vancouver"]
})

data["city_code"] = data["city"].astype("category").cat.codes

print(data)
```

---

## Code Example (Scaling)

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.DataFrame({
    "age":[25,30,35],
    "income":[40000,50000,70000]
})

scaler = StandardScaler()

scaled = scaler.fit_transform(data)

print(scaled)
```

---

## Common Mistakes Students Make

Training on raw data.

Encoding categories incorrectly.

Forgetting scaling.

Keeping irrelevant features.

---

## How This Appears in Interviews / Industry

Feature Engineering prepares data for Machine Learning.

Real ML work:

70–80% data preparation.

---

## Quick Visual Mental Model

Raw Data → Clean Data → Model

Messy Data → Clean Data → Machine Learning

---

## Student Challenges

Concept Questions:

1. What is feature engineering?
2. Why clean data?
3. Why convert text to numbers?

Coding Task:

Create dataset:

Age
Income
City

Add missing value.

Fix dataset.

Debugging:

Fix:

data["income"].fillna(mean)

---

## Extension Challenge

Try scaling methods:

StandardScaler
MinMaxScaler

Compare results.

---

## Summary

Feature Engineering prepares raw data for Machine Learning.
