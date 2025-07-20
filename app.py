from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("heart_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        input_data = {
            'Age': float(form_data['Age']),
            'RestingBP': float(form_data['RestingBP']),
            'Cholesterol': float(form_data['Cholesterol']),
            'FastingBS': float(form_data['FastingBS']),
            'MaxHR': float(form_data['MaxHR']),
            'Oldpeak': float(form_data['Oldpeak']),
            'Sex_M': 1.0, 'Sex_F': 0.0,
            'ChestPainType_ATA': 1.0, 'ChestPainType_NAP': 0.0, 'ChestPainType_ASY': 0.0, 'ChestPainType_TA': 0.0,
            'RestingECG_Normal': 1.0, 'RestingECG_LVH': 0.0, 'RestingECG_ST': 0.0,
            'ExerciseAngina_N': 1.0, 'ExerciseAngina_Y': 0.0,
            'ST_Slope_Up': 1.0, 'ST_Slope_Flat': 0.0, 'ST_Slope_Down': 0.0
        }

        df = pd.DataFrame([input_data], columns=model_columns)
        prediction = model.predict(df)
        result = 'Positive for Heart Disease' if prediction[0] == 1 else 'Negative for Heart Disease'

        return render_template('index.html', prediction_text=f'Result: {result}')
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    print("Starting Flask App...")
    app.run(debug=True)
