from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load saved model files
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', [])

    # Create input array
    input_data = np.zeros(len(columns))

    for symptom in symptoms:
        if symptom in columns:
            index = columns.index(symptom)
            input_data[index] = 1

    # Predict
    prediction = model.predict([input_data])[0]
    disease = le.inverse_transform([prediction])[0]

    return jsonify({
        'disease': disease,
        'symptoms_used': symptoms,
        'total_symptoms': len(symptoms)
    })

if __name__ == '__main__':
    app.run(debug=True)