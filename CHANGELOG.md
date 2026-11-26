# Changelog

## GitHub-ready engineering pass

- Replaced the original monolithic Ames training script with a reusable, configurable XGBoost pipeline.
- Fixed inverse target transformation for interpretable price predictions.
- Added holdout evaluation, final full-data refit, model metadata, and batch submission generation.
- Completed the Ford car-price project with a Random Forest pipeline, training CLI, prediction CLI, and clear dataset requirements.
- Added synthetic-data tests for both projects.
- Preserved original source files and the supplied legacy model artifact.
