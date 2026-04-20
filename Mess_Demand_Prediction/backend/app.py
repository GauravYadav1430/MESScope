"""
MESScope Operational Backend
=============================
Solves: "Given a meal and expected headcount, how many portions should we prepare?"

All predictions are based ONLY on pre-serving data:
  - Meal name  (known before cooking)
  - Headcount  (known before cooking)

No ratings, no post-consumption signals used anywhere.

Endpoints:
  GET  /health          - sanity check
  GET  /meals           - list all meals + stats
  POST /predict         - core: portions, leftovers, confidence
  GET  /wastage_report  - meals ranked by wastage
"""

import os, json, math
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE  = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, '..', 'models')

with open(os.path.join(MODEL, 'meal_aggregates.json')) as f:
    raw = json.load(f)

MEAL_DATA = {m['meal']: m for m in raw}   # O(1) lookup by name

with open(os.path.join(MODEL, 'operational_meta.json')) as f:
    META = json.load(f)

SAFETY_BUFFER = META['safety_buffer']          # 0.05 = +5%
FALLBACK_AVG  = META['fallback_avg_consumption']
FALLBACK_STD  = META['fallback_std_consumption']


def compute_prediction(meal_name: str, headcount: int) -> dict:
    """
    Formula
    -------
    avg_consumption_factor  = pct_full*1.0 + pct_half*0.5 + pct_skip*0.0
    base_portions           = headcount * avg_consumption_factor
    portions_to_prepare     = ceil(base_portions * 1.05)   (+5% safety buffer)
    expected_leftover       = portions_to_prepare - (headcount * avg_cf)
    """
    is_fallback = meal_name not in MEAL_DATA

    if is_fallback:
        avg_cf   = FALLBACK_AVG
        std_cf   = FALLBACK_STD
        conf     = 'Low'
        pct_full = round(avg_cf, 4)
        pct_half = 0.0
        pct_skip = round(1.0 - avg_cf, 4)
        wastage  = round(1.0 - avg_cf, 4)
    else:
        d        = MEAL_DATA[meal_name]
        avg_cf   = d['avg_consumption']
        std_cf   = d['std_consumption']
        conf     = d['confidence']
        pct_full = d['pct_full']
        pct_half = d['pct_half']
        pct_skip = d['pct_skipped']
        wastage  = d['wastage_rate']

    base_portions       = headcount * avg_cf
    portions_to_prepare = math.ceil(base_portions * (1 + SAFETY_BUFFER))
    expected_consumed   = round(headcount * avg_cf, 1)
    expected_leftover   = round(max(0.0, portions_to_prepare - expected_consumed), 1)
    leftover_pct        = round((expected_leftover / portions_to_prepare) * 100, 1) if portions_to_prepare > 0 else 0

    conf_msg = {
        'High':   f'Consistent eating pattern (std={std_cf:.3f}). Prediction is reliable.',
        'Medium': f'Moderate variance (std={std_cf:.3f}). Prediction is reasonable.',
        'Low':    f'High variance (std={std_cf:.3f}). Add extra caution.',
    }

    return {
        'meal':                  meal_name,
        'headcount':             headcount,
        'portions_to_prepare':   portions_to_prepare,
        'base_portions':         round(base_portions, 1),
        'expected_consumed':     expected_consumed,
        'expected_leftover':     expected_leftover,
        'leftover_pct':          leftover_pct,
        'confidence':            conf,
        'confidence_reason':     conf_msg[conf],
        'safety_buffer_applied': f'{int(SAFETY_BUFFER * 100)}%',
        'avg_consumption_rate':  round(avg_cf * 100, 1),
        'wastage_rate':          round(wastage * 100, 1),
        'eating_breakdown': {
            'expected_full_plates':  round(headcount * pct_full),
            'expected_half_plates':  round(headcount * pct_half),
            'expected_skipped':      round(headcount * pct_skip),
            'pct_full':              round(pct_full * 100, 1),
            'pct_half':              round(pct_half * 100, 1),
            'pct_skipped':           round(pct_skip * 100, 1),
        },
        'is_fallback': is_fallback,
    }


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'meals_loaded': len(MEAL_DATA)})


@app.route('/meals', methods=['GET'])
def get_meals():
    meals_list = sorted([
        {
            'meal':            n,
            'avg_consumption': round(d['avg_consumption'] * 100, 1),
            'wastage_rate':    round(d['wastage_rate']    * 100, 1),
            'confidence':      d['confidence'],
            'pct_full':        round(d['pct_full']        * 100, 1),
            'pct_half':        round(d['pct_half']        * 100, 1),
            'pct_skipped':     round(d['pct_skipped']     * 100, 1),
        }
        for n, d in MEAL_DATA.items()
    ], key=lambda x: x['avg_consumption'], reverse=True)
    return jsonify({'meals': meals_list, 'count': len(meals_list)})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    errors = []

    meal_name = str(data.get('meal_name', '')).strip()
    if not meal_name:
        errors.append("'meal_name' is required.")

    headcount = data.get('expected_headcount')
    try:
        headcount = int(headcount)
        if not (1 <= headcount <= 10000):
            raise ValueError
    except (TypeError, ValueError):
        errors.append("'expected_headcount' must be an integer between 1 and 10000.")

    if errors:
        return jsonify({'error': True, 'messages': errors}), 400

    return jsonify(compute_prediction(meal_name, headcount))


@app.route('/wastage_report', methods=['GET'])
def wastage_report():
    report = sorted([
        {
            'meal':        n,
            'wastage_rate': round(d['wastage_rate']    * 100, 1),
            'consumption':  round(d['avg_consumption'] * 100, 1),
            'pct_skipped':  round(d['pct_skipped']     * 100, 1),
            'confidence':   d['confidence'],
            'action': (
                'Consider removing from menu'  if d['wastage_rate'] >= 0.55 else
                'Reduce production quantity'    if d['wastage_rate'] >= 0.40 else
                'Monitor — moderate wastage'    if d['wastage_rate'] >= 0.25 else
                'Good — keep current quantity'
            )
        }
        for n, d in MEAL_DATA.items()
    ], key=lambda x: x['wastage_rate'], reverse=True)
    return jsonify({'wastage_report': report})


if __name__ == '__main__':
    print(f"MESScope API → http://localhost:5000  ({len(MEAL_DATA)} meals loaded)")
    app.run(debug=True, host='0.0.0.0', port=5000)
