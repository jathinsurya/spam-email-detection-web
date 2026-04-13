# app.py

from flask import Flask, render_template, request, jsonify
from model import predict_message

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route("/", methods=["GET", "POST"])
def home():
    result_data = None

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        
        if message:
            result_data = predict_message(message)
        else:
            result_data = {"error": "Please enter a message"}

    return render_template("index.html", result=result_data)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for predictions."""
    data = request.get_json()
    message = data.get("message", "").strip()
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    result = predict_message(message)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)