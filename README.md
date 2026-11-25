# Fraud Transaction Detection

Detects fraudulent credit card transactions with XGBoost and ensemble models, with a Streamlit app for trying predictions.

## What is in this repo

| File | Purpose |
|---|---|
| `fraud_detection_using_xgboost.ipynb` | Data preparation, training and evaluation |
| `streamlit_application.py` | Web app for predicting a single transaction |
| `xgb_classifier_model.pkl` | Trained XGBoost model, loaded by the app |
| `adb_classifier_model.pkl`, `voting_classifier_model.pkl` | Other trained models saved from the experiments |
| `requirements.txt` | Dependencies |

## Notebook

- Reads `fraudTrain.csv` and `fraudTest.csv` and combines them.
- Drops identifier columns and balances the classes with `RandomUnderSampler`.
- Builds features including the distance between customer and merchant (haversine formula, in km).
- Tunes `XGBClassifier` with `GridSearchCV` (for example `n_estimators` of 100, 150 or 200) and also builds a soft `VotingClassifier`.
- Evaluates with classification reports and ROC AUC, and saves the model with `joblib`.

The CSV data files are not in this repo.

## Streamlit app

The app asks for a transaction's amount, gender, city population, distance from the merchant, age, time of day, month and category. It predicts with `xgb_classifier_model.pkl` and shows whether the transaction looks fraudulent.

```bash
pip install -r requirements.txt
streamlit run streamlit_application.py
```
