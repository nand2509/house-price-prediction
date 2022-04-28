from flask import Flask, render_template, request
import pandas as pd
import os
import pickle
import joblib

app = Flask(__name__)
df = pd.read_csv("benguluruhome.csv")
pipe1 = pickle.load(open("RidgeModel.pkl", 'rb'))
pipe = os.path.join(app.root_path, 'pipe1')
@app.route('/')
def index():
    locations = sorted(df['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('locations')
    bhk =  request.form.get('bhk')
    bath =  request.form.get('bath')
    sqft =  request.form.get('total_sqft')

    print(locations, bhk, bath, sqft)
    input = pd.DataFrame([[locations, sqft, bath, bhk]], columns=['locations', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0]

    return str(prediction)
if __name__=="__main__":
    print("starting python flask server f or home price prediction")
    app.run(debug=True, port=5001)