from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('bot_click_model.pkl')

cat_feats = ['ip_address','device','browser','geo_location','referrer']
num_feats = [
    'click_frequency','session_duration','mouse_movements',
    'click_pattern_count','scroll_depth','time_on_page',
    'num_pages_visited','click_interval_std','hour_of_day'
]

# 1) Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df['click_time']  = pd.to_datetime(df['click_time'])
    df['hour_of_day'] = df['click_time'].dt.hour
    X = df[cat_feats + num_feats]
    pred = model.predict(X)[0]
    return jsonify({'is_bot': int(pred)})

# 2) Everything else (/, .css, .js) comes from the same folder
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_frontend(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    # run from this folder
    app.run(host='0.0.0.0', port=5000, debug=True)
