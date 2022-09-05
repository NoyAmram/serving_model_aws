import pickle
import pandas as pd
from flask import Flask, request

MODEL_FILE = "churn_model.pkl"
AWS_PORT = 8080

app = Flask(__name__)

global clf


def read_model(file_name):
    """ :param file_name for path to model file
        :returns the fitted model"""
    with open(file_name, "rb") as f:
        global clf
        clf = pickle.load(f)
    return clf


def predict(data):
    """:param data to predict by fitted classification model that was read from pickle file
        :returns model prediction"""
    return clf.predict(data)


@app.route('/predict_churn')
def get_forecast():
    """receives inputs for a single prediction as parameters
    returns a single prediction as a string."""
    is_male = int(request.args.get('is_male'))
    num_inters = int(request.args.get('num_inters'))
    late_on_payment = int(request.args.get('late_on_payment'))
    age = int(request.args.get('age'))
    years_in_contract = float(request.args.get('years_in_contract'))

    x = pd.DataFrame([[is_male, num_inters, late_on_payment, age, years_in_contract]])
    answer = predict(x)
    return str(answer[0])


def main():
    """ starting function to call above functions and verify prediction results"""
    read_model(MODEL_FILE)


if __name__ == '__main__':
    main()
    app.run(host='0.0.0.0', port=AWS_PORT, debug=True)
