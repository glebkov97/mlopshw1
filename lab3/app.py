from flask import Flask, request, jsonify
import iris_classification as ic

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    prediction = ic.predict(ic.model, data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)