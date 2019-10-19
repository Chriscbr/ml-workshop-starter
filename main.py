from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)

# from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
model = None

import pandas as pd
import numpy as np

def load_data():
    pd_dataframe = pd.read_csv('diabetes.txt', sep='\t')
    np_data = pd_dataframe.to_numpy()
    return np_data

def train_model():
    # Load the diabetes dataset
    diabetes = load_data()

    # Pick the first four features of the dataset
    # (age, sex, bmi, blood pressure)
    diabetes_X = diabetes[:, :4]

    # Extract the target values (the last column)
    diabetes_y = diabetes[:, -1]

    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X, diabetes_y)

    return regr
  
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        bmi = float(request.form['bmi'])
        bp = float(request.form['bp'])
        sample = [age, sex, bmi, bp]
        prediction = model.predict([sample])
        return render_template("predict.html", result = prediction)

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    model = train_model()
    app.run(debug = True)
