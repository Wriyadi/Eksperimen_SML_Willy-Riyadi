import time
import requests
import psutil
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest

app = Flask(__name__)

# ==========================================
# 10 METRIKS UNTUK GRAFANA (KRITERIA ADVANCE)
# ==========================================
# 1. Total Requests
REQUEST_TOTAL = Counter('model_requests_total', 'Total inference requests')
# 2. Success Requests
REQUEST_SUCCESS = Counter('model_requests_success', 'Total successful requests')
# 3. Failed Requests
REQUEST_FAILED = Counter('model_requests_failed', 'Total failed requests')
# 4. Latency
LATENCY = Histogram('model_latency_seconds', 'Time taken to process request')
# 5. CPU Usage
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
# 6. Memory Usage
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Memory usage percentage')
# 7. Prediction Result Output
PREDICTION_RESULT = Counter('model_prediction_output', 'Count of predictions', ['stroke_risk'])
# 8. Input Feature: Age
AGE_INPUT = Histogram('input_feature_age', 'Distribution of input age', buckets=[0, 30, 50, 70, 90, 100])
# 9. Input Feature: Avg Glucose Level
GLUCOSE_INPUT = Histogram('input_feature_avg_glucose_level', 'Distribution of glucose levels', buckets=[50, 100, 150, 200, 250])
# 10. Input Feature: BMI
BMI_INPUT = Histogram('input_feature_bmi', 'Distribution of BMI', buckets=[10, 20, 30, 40, 50])

# URL MLflow Docker Container
MLFLOW_MODEL_URL = "http://localhost:5001/invocations"

@app.route('/metrics', methods=['GET'])
def metrics():
    # Update resource metrics
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4'}

@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_TOTAL.inc()
    start_time = time.time()
    
    try:
        data = request.json
        
        # Ekstrak data input untuk metrik (contoh mengambil baris pertama)
        if 'dataframe_records' in data:
            record = data['dataframe_records'][0]
            if 'age' in record: AGE_INPUT.observe(float(record['age']))
            if 'avg_glucose_level' in record: GLUCOSE_INPUT.observe(float(record['avg_glucose_level']))
            if 'bmi' in record: BMI_INPUT.observe(float(record['bmi']))

        # Forward request ke MLflow Model
        response = requests.post(MLFLOW_MODEL_URL, json=data, headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            REQUEST_SUCCESS.inc()
            pred_value = response.json().get('predictions', [0])[0]
            PREDICTION_RESULT.labels(stroke_risk=str(pred_value)).inc()
        else:
            REQUEST_FAILED.inc()

        LATENCY.observe(time.time() - start_time)
        return jsonify(response.json()), response.status_code

    except Exception as e:
        REQUEST_FAILED.inc()
        LATENCY.observe(time.time() - start_time)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)