from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('random_forest_regression_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Convert the received JSON data into a DataFrame
    nouvelle_donnee = pd.DataFrame([data])

    # Condition to check if the date is prior to or equal to February 29, 2024
    if (nouvelle_donnee['Annee'][0] < 2024) or (
            nouvelle_donnee['Annee'][0] == 2024 and nouvelle_donnee['Mois'][0] <= 2):
        return jsonify({'error': 'Date invalide : la date doit être postérieure au 29 février 2024.'})

    # Prediction
    prediction = int(round(model.predict(nouvelle_donnee)[0]))

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
