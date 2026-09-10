import os
from flask import Flask, request, jsonify, send_from_directory
from predictive_engine.manager import predict_disaster_risk

# 1. Server Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_folder = os.path.join(current_dir, 'frontend')
app = Flask(__name__, static_folder=frontend_folder)

# 2. Route to load the UI
@app.route('/')
def load_dashboard():
    return send_from_directory(app.static_folder, 'dashboard.html')

# 3. Route for the ML Engine (Your existing code)
@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    try:
        incoming_data = request.get_json()
        risk_result = predict_disaster_risk(incoming_data)
        
        return jsonify({
            'status': 'success',
            'prediction': risk_result
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# 4. Start the Server
if __name__ == '__main__':
    print("Starting SIH Disaster Management Server...")
    app.run(debug=True, port=5000)
