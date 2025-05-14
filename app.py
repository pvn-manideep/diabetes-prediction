from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained Random Forest model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form data
            pregnancies = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bloodpressure = int(request.form['bloodpressure'])
            skinthickness = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            diabetespedigree = float(request.form['diabetespedigree'])
            age = int(request.form['age'])

            # Prepare data for prediction
            input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness,
                                    insulin, bmi, diabetespedigree, age]])

            # Make prediction
            result = model.predict(input_data)[0]
            prediction = "likely" if result == 1 else "not likely"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
