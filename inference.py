import pickle
import numpy as np
import pandas as pd
from flask import Flask, request

MODEL_FILE = "churn_model.pkl"
PREDICT_FILE = "preds.csv"
TEST_FILE = "X_test.csv"

app = Flask(__name__)

global clf


def read_model(file_name):
    """ :param file_name for path to model file
        :returns the fitted model"""
    with open(file_name, "rb") as f:
        global clf
        clf = pickle.load(f)
    return clf


def read_test_data(test_file):
    """ :param test_file for test data as csv
        :returns DataFrame of test data"""
    return pd.read_csv(test_file, encoding="utf-8")


def read_predictions(prediction_file):
    """ :param prediction_file for prediction saved in train module
        :returns numpy array of the predictions """
    return np.loadtxt(prediction_file, delimiter=',')


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
    X_test = read_test_data(TEST_FILE)
    predictions = read_predictions(PREDICT_FILE)
    forcast = predict(X_test)
    assert (forcast == predictions).all()


if __name__ == '__main__':
    main()
    app.run(host='0.0.0.0', port=8081, debug=True)
