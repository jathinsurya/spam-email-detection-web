# Simplified version to test
from flask import Flask, render_template, request
from model import predict_message

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            try:
                result = predict_message(message)
                print(f"✅ Prediction: {result}")
            except Exception as e:
                print(f"❌ ERROR: {e}")
                result = {"error": str(e)}
        else:
            result = {"error": "Empty message"}
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
