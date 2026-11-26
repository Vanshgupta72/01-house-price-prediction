# Engineering and Verification Report

## Main Ames House Price Prediction

- Training data: 1,460 labeled rows.
- Holdout validation: 292 rows (20%, random_state=42).
- Features used: 79 (the `Id` identifier is excluded).
- Model: XGBoost regressor in a scikit-learn preprocessing pipeline.
- Target handling: `log1p` during training; `expm1` for predictions and metrics.
- Holdout MAE: $15,856.87
- Holdout RMSE: $26,566.97
- Holdout R²: 0.9080

### End-to-end checks performed

- Training command completed successfully using the supplied `train.csv`.
- Model bundle was saved and reloaded by the prediction command.
- Prediction command generated 1,459 rows from the supplied `test.csv`.
- Submission schema: `Id`, `SalePrice`.
- Submission row count matches the test CSV row count.
- The generated predictions were inspected for numeric validity and positivity.
- Six automated tests cover both project workflows, model pipeline behavior, unseen categories, input validation, and prediction output shape.
- Python source compilation completed without errors.

## Ford Car Price Prediction

- The original source was an unfinished preprocessing draft.
- It has been completed into a train/evaluate/save and batch-predict workflow.
- The original archive did not contain `ford.csv`.
- The new Ford workflow was trained and exercised against generated synthetic test data; it has **not** been trained or evaluated against the real Ford dataset.
- Add a properly licensed dataset at `01_car_price_prediction/data/ford.csv` before real-data training.

## Security and publishing

- The supplied legacy `.pkl` was not loaded. It is retained only under `legacy/` for reference and ignored by Git.
- The newly trained model is included in this ZIP for immediate local prediction, but generated artifacts are ignored by Git; train again after cloning the repository.
- Dataset source/license provenance was not included in the upload. Verify redistribution rights and add attribution before making the CSV files public.
- No repository license was added because the right to relicense the supplied source/data was not established.

## Test summary

- Python compile: 14 source files passed.
- Automated unit/integration tests: 6 passed.
- Main dataset training: passed.
- Main dataset batch prediction: passed (1,459 predictions).
- Ford real-dataset training: blocked only by the missing original `ford.csv`; synthetic-data training/prediction tests passed.
