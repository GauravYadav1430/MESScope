"""
Mess Food Survey - Complete ML Pipeline
=======================================
Dataset: 210 responses, 21 meals tracked
Features: Rating (1-5), How much eaten (Full/Half/Skipped), Liked? (Yes/Neutral/No)
Target: demand_score (per meal) and wastage_score (per meal)

Problem Formulation:
- demand_score = weighted consumption (Full=1.0, Half=0.5, Skipped=0.0)
- wastage_score = 1 - demand_score (inverse of consumption; food prepared but not eaten)
- We train a RandomForest regressor to predict demand_score for each meal
  given: rating, liked_encoded, eat_encoded
- This enables: "If we serve X, how much will be consumed?"
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'mess_data.csv')

df = pd.read_csv(file_path)
print(f"Dataset shape: {df.shape}")

# List of all 21 meals (normalize naming)
MEALS = [
    'Aloo Pyaz Paratha', 'Rajma', 'Chana Masala', 'Poha', 'Paneer Do Pyaza',
    'Mix Veg + Rasmalai', 'Mix Paratha', 'Kadhi Pakoda', 'Chicken/Paneer Tikka',
    'Idli Sambhar', 'Gobi Aloo', 'Aloo Beans + Milk Cake', 'Pav Bhaji',
    'Black Chana', 'Paneer Bhurji/Egg Curry', 'Dosa Uttapam', 'Chole Paneer',
    'Dal Tadka + Rasgulla', 'Amritsari Naan Chole', 'Aloo Nutri',
    'Chicken/Paneer Chilli'
]

# ─────────────────────────────────────────────────────────────
# STEP 2: PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

# Encoding maps
EAT_MAP    = {'Full': 1.0, 'Half': 0.5, 'Skipped': 0.0}
LIKE_MAP   = {'Yes': 1, 'Neutral': 0, 'No': -1}
VENUE_MAP  = {'Mess': 0, 'Both': 1, 'Outside': 2}

# Encode venue
df['venue_enc'] = df['Where do you usually eat?'].map(VENUE_MAP).fillna(1)

# Build per-meal feature rows for training
rows = []
for meal in MEALS:
    # Find columns (handle Dosa typo in 'How much' column)
    rate_col = f'Rate the following meals [{meal}]'
    
    # 'Dosa Uttapam' has a typo in 'How much' column: 'Dosa Uttpam'
    eat_col_candidates = [
        f'How much did you eat? [{meal}]',
        f'How much did you eat? [Dosa Uttpam]'  # typo variant
    ]
    like_col = f'Did you like the meal? [{meal}]'
    
    eat_col = None
    for c in eat_col_candidates:
        if c in df.columns:
            eat_col = c
            break

    if eat_col is None or rate_col not in df.columns or like_col not in df.columns:
        print(f"  WARNING: Missing columns for {meal}")
        continue

    for _, row in df.iterrows():
        eat_val  = EAT_MAP.get(str(row[eat_col]).strip(), 0.5)
        like_val = LIKE_MAP.get(str(row[like_col]).strip(), 0)
        rate_val = int(row[rate_col]) if pd.notna(row[rate_col]) else 3
        venue    = row['venue_enc']

        # demand_score: weighted consumption (0.0 – 1.0)
        demand_score = eat_val
        # wastage_score: inverse of demand (1.0 = fully wasted, 0.0 = fully consumed)
        wastage_score = 1.0 - demand_score

        rows.append({
            'meal': meal,
            'rating': rate_val,
            'like_enc': like_val,
            'eat_enc': eat_val,
            'venue_enc': venue,
            'demand_score': demand_score,
            'wastage_score': wastage_score,
        })

ml_df = pd.DataFrame(rows)
print(f"ML dataset shape: {ml_df.shape}")
print(ml_df.describe().round(3))

# One-hot encode meal names for model training
meal_dummies = pd.get_dummies(ml_df['meal'], prefix='meal')
feature_cols = ['rating', 'like_enc', 'venue_enc'] + list(meal_dummies.columns)
X = pd.concat([ml_df[['rating', 'like_enc', 'venue_enc']], meal_dummies], axis=1)
y_demand  = ml_df['demand_score']
y_wastage = ml_df['wastage_score']

# ─────────────────────────────────────────────────────────────
# STEP 3: TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, yd_train, yd_test, yw_train, yw_test = train_test_split(
    X, y_demand, y_wastage, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4: MODEL TRAINING — Random Forest
# Chosen because:
#   • Handles non-linear relationships between rating/likes and demand
#   • Robust to small datasets (210×21 expanded rows)
#   • Provides feature importance natively
#   • No strong distributional assumptions needed
# ─────────────────────────────────────────────────────────────
demand_model  = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
wastage_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)

demand_model.fit(X_train, yd_train)
wastage_model.fit(X_train, yw_train)

# ─────────────────────────────────────────────────────────────
# STEP 5: EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, name):
    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    print(f"\n{name} Model:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")
    return {'mae': round(mae,4), 'rmse': round(rmse,4), 'r2': round(r2,4)}

d_metrics = evaluate(demand_model,  X_test, yd_test, "Demand")
w_metrics = evaluate(wastage_model, X_test, yw_test, "Wastage")

# ─────────────────────────────────────────────────────────────
# STEP 6: FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
importances = pd.Series(demand_model.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=False).head(10)
print("\nTop 10 Feature Importances (Demand Model):")
print(top_features.round(4))

# ─────────────────────────────────────────────────────────────
# STEP 7: PER-MEAL AGGREGATE INSIGHTS
# ─────────────────────────────────────────────────────────────
meal_stats = ml_df.groupby('meal').agg(
    avg_rating=('rating', 'mean'),
    avg_demand=('demand_score', 'mean'),
    avg_wastage=('wastage_score', 'mean'),
    like_score=('like_enc', 'mean')
).round(3).reset_index()

meal_stats['recommendation'] = meal_stats['avg_demand'].apply(
    lambda d: 'Increase Production' if d >= 0.7
              else ('Decrease Production' if d <= 0.4 else 'Maintain')
)
meal_stats = meal_stats.sort_values('avg_demand', ascending=False)
print("\nPer-meal Insights:")
print(meal_stats.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 8: SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
os.makedirs('../models', exist_ok=True)

joblib.dump(demand_model,  '../models/demand_model.pkl')
joblib.dump(wastage_model, '../models/wastage_model.pkl')

# Save feature column list so backend can reconstruct input
with open('../models/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

# Save meal list
with open('../models/meals.json', 'w') as f:
    json.dump(MEALS, f)

# Save meal stats for dashboard
meal_stats.to_json('../models/meal_stats.json', orient='records')

# Save encoding maps
meta = {
    'eat_map': EAT_MAP,
    'like_map': LIKE_MAP,
    'venue_map': VENUE_MAP,
    'd_metrics': d_metrics,
    'w_metrics': w_metrics,
    'feature_importances': top_features.to_dict()
}
