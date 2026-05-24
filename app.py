import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

model = None
X_test = None

def load_resources():
    global model, X_test
    print("Loading model and pre-computed features...")
    if os.path.exists('xgboost_model.pkl'):
        with open('xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
    if os.path.exists('X_test.pkl'):
        X_test = pd.read_pickle('X_test.pkl')
    print("Resources loaded.")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None or X_test is None:
        load_resources()
        if model is None or X_test is None:
            return jsonify({'error': 'Model or feature data not found. Is training finished?'}), 503
            
    data = request.json
    shop_id = data.get('shop_id')
    item_id = data.get('item_id')
    
    if shop_id is None or item_id is None:
        return jsonify({'error': 'Please provide shop_id and item_id'}), 400
        
    try:
        shop_id = int(shop_id)
        item_id = int(item_id)
    except ValueError:
        return jsonify({'error': 'shop_id and item_id must be integers'}), 400
        
    row = X_test[(X_test['shop_id'] == shop_id) & (X_test['item_id'] == item_id)]
    
    if row.empty:
        return jsonify({'error': 'No pre-calculated features found for this Shop/Item combination for Nov 2015. Please try another pair.'}), 404
        
    prediction = model.predict(row)
    prediction_clipped = prediction.clip(0, 20)
    
    return jsonify({
        'shop_id': shop_id,
        'item_id': item_id,
        'predicted_sales': round(float(prediction_clipped[0]), 2)
    })

if __name__ == '__main__':
    load_resources()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
