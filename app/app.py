from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Ensure the model path is correct
model_path = os.path.join(os.path.dirname(__file__), '../models/house_price_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Estimated House Price: $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
