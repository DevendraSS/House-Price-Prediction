from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from sklearn import linear_model
from joblib import load
app = Flask(__name__)

model = linear_model.LinearRegression()
model = load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method== 'POST':
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])

        data = pd.DataFrame([[area, bedrooms, age]], columns=['area', 'bedrooms', 'age'])
        predicted_price = model.predict(data)

        return render_template('index.html', predicted_price=round(predicted_price[0],2))
    
if __name__ == '__main__':
    app.run(debug=True)

