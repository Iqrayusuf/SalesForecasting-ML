# SalesForecasting-ML
Sales forecasting project using Python and machine learning in Google Colab. Built a model to predict future sales using enriched features and evaluated performance with SMAPE.
# 📊 Sales Forecasting ML Project

A machine learning project built in **Google Colab** to forecast product sales using enriched data features. This project uses **Upgini** for automated feature enrichment and **CatBoost** for model training. Forecast accuracy is evaluated using **SMAPE** for real-world performance insight.

---

## 📚 Overview

This project aims to predict future product sales using a combination of enriched features and advanced machine learning techniques. Instead of manually crafting all features, it leverages **Upgini** to automatically enhance the dataset with external data, then uses **CatBoost** to build an efficient regression model.

---

## 🧰 Technologies & Libraries

- 🐍 **Python**
- ☁️ **Google Colab**
- 🧠 **CatBoost** – Gradient boosting on decision trees
- 🔍 **Upgini** – Automated feature enrichment
- 📈 **SMAPE** – Symmetric Mean Absolute Percentage Error for evaluation

---

## 🔄 Workflow Steps

1. 📥 Load the original dataset (5 years of sales data)
2. 🔍 Enrich features automatically using **Upgini**
3. 🧪 Split time-series data for training and testing
4. 🧠 Train regression model using **CatBoost**
5. 📊 Evaluate performance using **SMAPE**

---

## 🔎 Sample Code Snippet

```python
# Train model on enriched training data
model.fit(enriched_train_features, sampled_train_target)

# Predict on enriched test set
enriched_preds = model.predict(enriched_test_features)

# Evaluate using SMAPE
eval_metric(sampled_test_target.values, enriched_preds, "SMAPE")
