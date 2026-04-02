import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
from time import gmtime, strftime

# Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression, Ridge , Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold


parser = argparse.ArgumentParser(
    prog="FDIC Bank Failure LGD Predictor",
    description="Training script for Loss Given Default prediction (ML Project 2025/2026)"
)

# File paths
parser.add_argument("--bank_data", type=str, default="Summary_of_Results_3_18_2026.csv", help="Path to the FDIC bank dataset")
parser.add_argument("--unemp_data", type=str, default="US_unemployment_rate.csv", help="Path to the US unemployment rate data")
parser.add_argument("--fed_data", type=str, default="avg Federal Funds Effective Rate.csv", help="Path to the Federal Funds Rate data")
parser.add_argument("--feature_set", type=str, default="enriched", help="Choose 'baseline' or 'enriched'")
parser.add_argument("--task", type=str, default="regression", help="Choose 'regression' or 'classification'")
parser.add_argument("--ml_method", type=str, default="RandomForest", help="Model to use: 'RandomForest', 'LinearRegression', or 'Ridge'")
parser.add_argument("--rf_n_estimators", type=int, default=100, help="Number of trees (if RandomForest is selected)")
parser.add_argument("--cv_nsplits", type=int, default=5, help="Number of splits for cross-validation")
parser.add_argument("--save_dir", type=str, default="outputs", help="Directory where models and logs will be saved")

args = parser.parse_args()

# Create the directory containing the model, the logs, etc.
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
out_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(out_dir, exist_ok=True)

path_model = os.path.join(out_dir, "model.pkl")
path_config = os.path.join(out_dir, "config.json")
path_logs = os.path.join(out_dir, "logs.json")

# Store the configuration
with open(path_config, "w") as f:
    json.dump(vars(args), f, indent=4)


# Loading the dataset
print("--- Loading and preparing data---")

def prepare_data(bank_path, unemp_path, fed_path):
    df = pd.read_csv(bank_path)
    unemp = pd.read_csv(unemp_path)
    fed_rate = pd.read_csv(fed_path)

    # 1. Basic cleaning (Keep only 'FAILURE' and drop missing or zero values)
    df = df[df['RESTYPE'] == 'FAILURE'].copy()
    df = df.dropna(subset=['COST', 'QBFASSET', 'FAILDATE'])
    df = df[(df['COST'] != 0) & (df['QBFASSET'] > 0)]

    # 2. Create the target variable (LGD)
    df['LGD'] = df['COST'] / df['QBFASSET']

    # 3. Process dates and extract State
    df['FAILDATE'] = pd.to_datetime(df['FAILDATE'])
    df['YEAR'] = df['FAILDATE'].dt.year
    df['STATE'] = df['CITYST'].str[-2:]

    # 4. Process and merge macroeconomic data
    unemp['YEAR'] = pd.to_datetime(unemp['observation_date']).dt.year
    annual_unemp = unemp.groupby('YEAR')['UNRATE'].mean().reset_index()
    
    fed_rate['YEAR'] = pd.to_datetime(fed_rate['DATE']).dt.year
    annual_fed = fed_rate.groupby('YEAR')['VALUE'].mean().reset_index().rename(columns={'VALUE': 'FEDFUNDS'})

    df = df.merge(annual_unemp, on='YEAR', how='left')
    df = df.merge(annual_fed, on='YEAR', how='left')

    # 5. Feature Engineering
    df['Deposit_to_Asset_Ratio'] = df['QBFDEP'] / df['QBFASSET']

    # Calculate failures per State over the last 12 months
    df = df.sort_values('FAILDATE')
    df['State_Failures_Last_12M'] = 0
    for i in range(len(df)):
        current_state = df.iloc[i]['STATE']
        current_date = df.iloc[i]['FAILDATE']
        window_start = current_date - pd.Timedelta(days=365)
        count = df[(df['STATE'] == current_state) & 
                   (df['FAILDATE'] < current_date) & 
                   (df['FAILDATE'] >= window_start)].shape[0]
        df.at[df.index[i], 'State_Failures_Last_12M'] = count

    # Drop rows where macro data could not be matched
    return df.dropna(subset=['UNRATE', 'FEDFUNDS', 'LGD'])

# Execute data preparation
df_final = prepare_data(args.bank_data, args.unemp_data, args.fed_data)

# Define Features and Target
if args.feature_set == "baseline":
    print("Using BASELINE features (Original data only)... ")
    #we take only the columns of the base dataset 
    numerical_cols = ['QBFASSET', 'QBFDEP'] 
    categorical_cols = ['CHCLASS1', 'RESTYPE1', 'SAVR', 'STATE']
    
elif args.feature_set == "enriched":
    print("Using enriched features (With Macro and  Engineered features)... ")
    #here we take all the columns added + base columns 
    numerical_cols = ['QBFASSET', 'QBFDEP', 'Deposit_to_Asset_Ratio', 'State_Failures_Last_12M', 'YEAR', 'UNRATE', 'FEDFUNDS']
    categorical_cols = ['CHCLASS1', 'RESTYPE1', 'SAVR', 'STATE']
    
else:
    raise ValueError("The argument --feature_set must be 'baseline' or 'enriched'")

features= numerical_cols + categorical_cols

#target and metrics for models 
if args.task == "regression":
    target_name = 'LGD'
    scoring_metric = 'r2'
    cv_strategy = KFold(n_splits=args.cv_nsplits, shuffle=True, random_state=42) #bc we have sorted the data by date, we need to shuffle the cross validation
    print("Task: REGRESSION (Target: LGD, Metric: R2)")

elif args.task == "classification":
    #Create the binary target 
    df_final['LGD_class'] = (df_final['LGD'] > 0.2).astype(int)
    target_name = 'LGD_class'
    scoring_metric = 'roc_auc'
    cv_strategy = StratifiedKFold(n_splits=args.cv_nsplits, shuffle=True, random_state=42) #shuffle for cross-validation
    print("Task: CLASSIFICATION (Target: LGD > 0.2, Metric: ROC AUC)")
else:
    raise ValueError("The argument --task must be 'regression' or 'classification'")

X = df_final[features]
y = df_final[target_name]

# ColumnTransformer: scaling and one-hot encoding
preprocess = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

#Build the model 

#Regression Models 
if args.task == "regression":
    if args.ml_method == "RandomForest":
        model = RandomForestRegressor(random_state=42)
        param_grid = {'model__n_estimators': [100, 200, 400], 'model__max_depth': [5, 10, 20], 'model__min_samples_split': [2, 5]}
    elif args.ml_method == "Ridge":
        model = Ridge()
        param_grid = {'model__alpha': [0.01, 0.1, 1, 10, 100]}
    elif args.ml_method == "Lasso":
        model = Lasso(max_iter=10000)
        param_grid = {'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
    elif args.ml_method == "LinearRegression":
        model = LinearRegression()
        param_grid = {} 
    else:
        raise ValueError(f"Unknown regression model: {args.ml_method}")
    
#Classification Models:
elif args.task == "classification":
    if args.ml_method == "RandomForest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {'model__n_estimators': [200, 400], 'model__max_depth': [5, 10, 20], 'model__min_samples_split': [2, 5]}
    elif args.ml_method == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1], 'model__max_depth': [3, 5]}
    elif args.ml_method == "LogisticRegression":
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {} 
    else:
        raise ValueError(f"Unknown classification model: {args.ml_method}") 
    

#Build the pipeline
pipeline = Pipeline([
    ("preprocessor", preprocess),
    ("model", model)
])

print(f"-------Training and optimizing model: {args.ml_method.upper()} ------------")

#SearchGrid for the models that have parameters to tune 
if param_grid:
    print(f"Searching for best hyperparameters ({args.cv_nsplits}-fold CV)...")
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv_strategy,scoring=scoring_metric, n_jobs=-1)
    grid_search.fit(X, y)
    
    final_model = grid_search.best_estimator_
    mean_score = grid_search.best_score_
    best_params = grid_search.best_params_
    
    print("\n---------Best Parameters--------")
    for key, value in best_params.items():
        print(f" {key}: {value}")

else:
    print(f"Evaluating model ({args.cv_nsplits}-fold CV)...")
    lst_scores = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring='r2')
    mean_score = np.mean(lst_scores)

    final_model = pipeline.fit(X, y)
    best_params = "None" #no GridSearch for those models 


# Save model
with open(path_model, 'wb') as f:
    pickle.dump(final_model, f)

# Save the metrics and configuration to logs.json
results = {
    "task": args.task,
    "model_used": args.ml_method,
    "feature_set": args.feature_set,
    "scoring_metric": scoring_metric,
    "mean_score": mean_score,
    "best_parameters": best_params
}
#saving
with open(path_model, 'wb') as f:
    pickle.dump(final_model, f)

with open(path_logs, "w") as f:
    json.dump(results, f, indent=4)

GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

#render
print(f"\n{GREEN}--> Success:{RESET} Model and logs saved in: {out_dir}")
print(f"{BLUE}-->{RESET} Final Score ({scoring_metric}): {mean_score:.4f}\n")