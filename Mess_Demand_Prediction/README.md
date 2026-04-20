# MESScope — Mess Intelligence & Demand Forecasting System

## 📁 Folder Structure
```
mess_ml_project/
├── data/
│   └── mess_data.csv          # Original survey dataset (210 rows, 66 cols)
├── models/                    # Auto-generated after training
│   ├── demand_model.pkl
│   ├── wastage_model.pkl
│   ├── feature_cols.json
│   ├── meals.json
│   ├── meal_stats.json
│   └── meta.json
├── analysis/
│   └── ml_pipeline.py         # Full ML: preprocessing → training → evaluation → saving
├── backend/
│   └── app.py                 # Flask REST API
├── frontend/
│   └── dashboard.html         # Complete standalone dashboard (no build step needed)
├── requirements.txt
└── README.md
```

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML model
```bash
cd analysis
python ml_pipeline.py
# Outputs: models/demand_model.pkl, wastage_model.pkl, feature_cols.json, etc.
```

### 3. Start the Flask API
```bash
cd backend
python app.py
# API running at http://localhost:5000
```

### 4. Open the Dashboard
```
Open frontend/dashboard.html in any browser (no server needed — data is embedded).
```

---

## 🔌 API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Health check |
| GET | `/meals` | List of all 21 meals |
| GET | `/data` | Full meal stats + model metrics |
| GET | `/insights` | Feature importance + top/bottom meals |
| GET | `/meal_trends` | All meals sorted by demand with recommendations |
| POST | `/predict` | Predict demand/wastage for a meal |

### POST /predict — Example
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"meal": "Rajma", "rating": 4, "liked": "Yes", "venue": "Mess"}'
```

Response:
```json
{
  "meal": "Rajma",
  "demand_pct": 97.3,
  "wastage_pct": 2.7,
  "recommendation": "Increase Production",
  "historical_demand": 74.5,
  "historical_wastage": 25.5
}
```

---

## 🧠 ML Approach Summary

### Problem
Predict per-meal demand and wastage from student survey data.

### Features (24 total)
- `rating` — student's 1–5 rating of the meal
- `like_enc` — encoded sentiment: Yes=1, Neutral=0, No=−1
- `venue_enc` — where they eat: Mess=0, Both=1, Outside=2
- `meal_*` — 21 one-hot columns for meal identity

### Targets
- `demand_score` — consumption proxy: Full=1.0, Half=0.5, Skipped=0.0
- `wastage_score` — 1 − demand_score

### Model: Random Forest Regressor
Chosen for:
- Non-linear feature interactions
- Built-in feature importance
- Robust on small-to-medium tabular data
- No strong distributional assumptions

### Performance
| Metric | Score |
|--------|-------|
| MAE    | 0.2016 |
| RMSE   | 0.2837 |
| R²     | 0.528  |

### Top Feature Importance
| Feature | Importance |
|---------|-----------|
| Did you like it? | 83.97% |
| Rating | 7.21% |
| Venue | 2.55% |
| Meal identity | ~6.27% (combined) |

---

## 💡 Key Insights
1. **Dosa Uttapam** and **Amritsari Naan Chole** are the top performers (>83% demand)
2. **Aloo Beans + Milk Cake** has the worst wastage (62.9%) → reduce production
3. Satisfaction ("liked?") drives 84% of demand prediction — most critical signal
4. 7% of students eat entirely outside — strong dissatisfaction indicator
5. Common requests: namkeen lassi, better rotis, anda bhurji
