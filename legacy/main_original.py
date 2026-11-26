import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

df = pd.read_csv(DATA_PATH)

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())

# pd.set_option('display.max_rows', None)
# print(df.isnull().sum().sort_values(ascending=False))

# print(df['PoolQC'].value_counts(dropna=False))  # to be droped
# print(df["MiscFeature"].value_counts(dropna=False)) to be droped
# print(df["Alley"].value_counts(dropna=False)) # To be dropped
# print(df["Fence"].value_counts(dropna=False)) to be dropped
# print(df["FireplaceQu"].value_counts(dropna=False)) # Fill with none

droper_one = ["PoolQC",
    "MiscFeature",
    "Alley",
    "Fence"]

df = df.drop(columns=droper_one)   # dropping highly sparse data
# print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))

df["FireplaceQu"] = df["FireplaceQu"].fillna("None")  # cosidering it as not present in house instead of dropping it

garage_cat = [
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond"
]

for col in garage_cat:
    df[col] = df[col].fillna("None")    # Same as above 

df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0) # Garage hi ni h to banega kaise 

bsmt_cols = [     # basement hai hi ni 
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2"
]

for col in bsmt_cols:
    df[col] = df[col].fillna("None")

# print(df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False))

# print(df["MasVnrType"].value_counts(dropna=False))

# print(df[df["MasVnrType"].isna()][["MasVnrType", "MasVnrArea"]])  # galti ni h sach me missing h 

df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

# print(df["LotFrontage"].describe())

# plt.figure(figsize=(5,3))
# df["LotFrontage"].plot.box()
# plt.show()
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median()) # outlier effected mean

# p(df["Electrical"].value_counts(dropna=False))

df["Electrical"].value_counts(dropna=False)

# print(df.isnull().sum())

# EDA


# (i) Distribution checking 

# print(df["SalePrice"].describe())

# plt.figure(figsize=(7,4))
# df['SalePrice'].hist(bins=30)
# plt.xlabel('Saleprice')
# plt.ylabel('Count')
# plt.title('Distribution of SalePrice')
# plt.show()     # It is right skewed

# print(df['SalePrice'].skew())
df['SalePrice']= np.log1p(df['SalePrice'])
# print(df['SalePrice'].skew())

numeric_features = df.select_dtypes(include=["number"]).columns

# (ii)  Correalation checking

# plt.figure(figsize=(8,5))
# plt.scatter(df['OverallQual'],df["SalePrice"])
# plt.xlabel('OverallQual')
# plt.ylabel('SalePrice(log transformed)')
# plt.title('Comaprsion btw overallqual and saleprice')
# plt.show()

# corr = df.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
# print(corr.head(10))

# import matplotlib.pyplot as plt

# plt.figure(figsize=(8,5))

# plt.scatter(df["GrLivArea"], df["SalePrice"])

# plt.xlabel("Ground Living Area")
# plt.ylabel("Log SalePrice")
# plt.title("GrLivArea vs SalePrice")

# plt.show()

# print(df[df["GrLivArea"] > 4000][["GrLivArea", "SalePrice"]])

# plt.figure(figsize=(8,5))
# plt.scatter(df['GarageCars'],df['SalePrice'])
# plt.xlabel('GarageCars')
# plt.ylabel('Log SalePrice')
# plt.title('Correlation of  Saleprice and Garagecars:')
# plt.show()

# plt.figure(figsize=(8,5))
# df.boxplot(column='SalePrice',by='GarageCars')
# plt.title('GarageCars vs Saleprice')
# plt.xlabel('GarageCars')
# plt.ylabel('Log Saleprice')
# plt.show()

# plt.figure(figsize=(8,5))

# plt.scatter(df["TotalBsmtSF"], df["SalePrice"])

# plt.xlabel("Total Basement Area (sq ft)")
# plt.ylabel("Log SalePrice")
# plt.title("TotalBsmtSF vs SalePrice")

# plt.show()

# print(df[df["TotalBsmtSF"] > 5000][
#     ["TotalBsmtSF", "SalePrice", "OverallQual", "GrLivArea", "GarageCars"]
# ])

# print(df.loc[1298, [
#     "OverallQual",
#     "OverallCond",
#     "YearBuilt",
#     "Neighborhood",
#     "SaleCondition",
#     "SaleType"
# ]])

# plt.figure(figsize=(10,6))

# df.boxplot(column="SalePrice", by="OverallQual")

# plt.title("OverallQual vs SalePrice")
# plt.suptitle("")
# plt.xlabel("Overall Quality")
# plt.ylabel("Log SalePrice")

# plt.show()

# df.boxplot(column= 'SalePrice', by= 'YearBuilt')
# plt.title('SalePrice vs Yearbuilt')
# plt.suptitle("")
# plt.xlabel('Yearbuilt')
# plt.ylabel('Saleprice')
# plt.show()

X = df.drop('SalePrice',axis=1)
y = df['SalePrice']

numeric_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(include="object").columns


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num",numeric_pipeline,numeric_features),
        ('cat',categorical_pipeline,categorical_features)
    ]
)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# from sklearn.linear_model import LinearRegression

# linear_model = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', LinearRegression())
# ])

# linear_model.fit(X_train,y_train)

# y_pred = linear_model.predict(X_test)

from  sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# mae = mean_absolute_error(y_test,y_pred)

# mse = mean_squared_error(y_test,y_pred)

# rmse = np.sqrt(mse)

# r2 = r2_score(y_test,y_pred)

# print("===== Linear Regression =====")
# print("MAE  :", mae)
# print("RMSE :", rmse)
# print("R2   :", r2)

from sklearn.tree import DecisionTreeRegressor

decision_tree_model = Pipeline([
    ('preprocessor',preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])

# decision_tree_model.fit(X_train, y_train)

# y_pred_dt = decision_tree_model.predict(X_test)

# mae_dt = mean_absolute_error(y_test, y_pred_dt)

# rmse_dt = np.sqrt(
#     mean_squared_error(y_test, y_pred_dt)
# )

# r2_dt = r2_score(y_test, y_pred_dt)

# print("===== Decision Tree =====")
# print("MAE  :", mae_dt)
# print("RMSE :", rmse_dt)
# print("R2   :", r2_dt)

from sklearn.ensemble import GradientBoostingRegressor

gradient_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])



# gradient_model.fit(X_train, y_train)

# y_pred_gb = gradient_model.predict(X_test)

# mae_gb = mean_absolute_error(y_test, y_pred_gb)

# rmse_gb = np.sqrt(
#     mean_squared_error(y_test, y_pred_gb)
# )

# r2_gb = r2_score(y_test, y_pred_gb)

# print("===== Gradient Boosting =====")
# print("MAE  :", mae_gb)
# print("RMSE :", rmse_gb)
# print("R2   :", r2_gb)

from xgboost import XGBRegressor

# xgb_model = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', XGBRegressor(
#         n_estimators=1000,
#         learning_rate=0.03,
#         max_depth=3,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective='reg:squarederror',
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# xgb_model.fit(X_train, y_train)

# y_pred_xgb = xgb_model.predict(X_test)

# mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# rmse_xgb = np.sqrt(
#     mean_squared_error(y_test, y_pred_xgb)
# )

# r2_xgb = r2_score(y_test, y_pred_xgb)

# print("===== XGBoost =====")
# print("MAE  :", mae_xgb)
# print("RMSE :", rmse_xgb)
# print("R2   :", r2_xgb)

# actual_prices = np.expm1(y_test)
# predicted_prices = np.expm1(y_pred_xgb)

# mae_original = mean_absolute_error(
#     actual_prices,
#     predicted_prices
# )

# rmse_original = np.sqrt(
#     mean_squared_error(
#         actual_prices,
#         predicted_prices
#     )
# )

# r2_original = r2_score(
#     actual_prices,
#     predicted_prices
# )

# print("===== XGBoost | Original Price Scale =====")
# print("MAE  :", mae_original)
# print("RMSE :", rmse_original)
# print("R2   :", r2_original)

from sklearn.model_selection import GridSearchCV

# dt_params = {
#     "model__max_depth": [3, 5, 7, 10, None],
#     "model__min_samples_split": [2, 5, 10],
#     "model__min_samples_leaf": [1, 2, 4]
# }

# dt_grid = GridSearchCV(
#     estimator=decision_tree_model,
#     param_grid=dt_params,
#     cv=5,
#     scoring="r2",
#     n_jobs=-1,
#     verbose=1
# )

# dt_grid.fit(X_train, y_train)

# print("Best Parameters:")
# print(dt_grid.best_params_)

# print("Best CV r2:")
# print(-dt_grid.best_score_)

# best_dt = dt_grid.best_estimator_

# y_pred_dt_tuned = best_dt.predict(X_test)

# mae_dt_tuned = mean_absolute_error(y_test, y_pred_dt_tuned)

# rmse_dt_tuned = np.sqrt(
#     mean_squared_error(y_test, y_pred_dt_tuned)
# )

# r2_dt_tuned = r2_score(
#     y_test,
#     y_pred_dt_tuned
# )

# print("===== Tuned Decision Tree =====")
# print("MAE :", mae_dt_tuned)
# print("RMSE:", rmse_dt_tuned)
# print("R2  :", r2_dt_tuned)


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV

# rf_params = {
#     "model__n_estimators": [200],
#     "model__max_depth": [None, 10, 20],
#     "model__min_samples_split": [2, 5],
#     "model__min_samples_leaf": [1, 2],
#     "model__max_features": ["sqrt", 0.8]
# }

# rf_pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("model", RandomForestRegressor(
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# rf_grid = GridSearchCV(
#     estimator= rf_pipeline,
#     param_grid=rf_params,
#     cv=5,
#     scoring="r2",
#     n_jobs=-1,
#     verbose=1
# )

# # rf_grid.fit(X_train, y_train)

# # print("Best Parameters:")
# # print(rf_grid.best_params_)

# # print("Best CV R2:")
# # print(rf_grid.best_score_)

# rf_grid.fit(X_train, y_train)

# print("Best Parameters:")
# print(rf_grid.best_params_)

# print("Best CV R2:")
# print(rf_grid.best_score_)

# best_rf = rf_grid.best_estimator_

# best_rf = rf_grid.best_estimator_

# y_pred_rf_tuned = best_rf.predict(X_test)

# mae_rf_tuned = mean_absolute_error(
#     y_test,
#     y_pred_rf_tuned
# )

# rmse_rf_tuned = np.sqrt(
#     mean_squared_error(
#         y_test,
#         y_pred_rf_tuned
#     )
# )

# r2_rf_tuned = r2_score(
#     y_test,
#     y_pred_rf_tuned
# )

# print("===== Tuned Random Forest =====")
# print("MAE :", mae_rf_tuned)
# print("RMSE:", rmse_rf_tuned)
# print("R2  :", r2_rf_tuned)

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ))
])

xgb_params = {
    "model__n_estimators": [500, 1000],
    "model__learning_rate": [0.03, 0.05],
    "model__max_depth": [2, 3],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8]
}

xgb_grid = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=xgb_params,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)

# print("Best Parameters:")
# print(xgb_grid.best_params_)

# print("Best CV R2:")
# print(xgb_grid.best_score_)

best_xgb = xgb_grid.best_estimator_

y_pred_xgb_tuned = best_xgb.predict(X_test)

mae_xgb_tuned = mean_absolute_error(
    y_test,
    y_pred_xgb_tuned
)

rmse_xgb_tuned = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_xgb_tuned
    )
)

r2_xgb_tuned = r2_score(
    y_test,
    y_pred_xgb_tuned
)

# print("===== Tuned XGBoost =====")
# print("MAE :", mae_xgb_tuned)
# print("RMSE:", rmse_xgb_tuned)
# print("R2  :", r2_xgb_tuned)

final_model = xgb_grid.best_estimator_

import joblib

joblib.dump(
    final_model,
    str(PROJECT_ROOT / "house_price_xgboost.pkl")
)

print("Final model saved successfully!")