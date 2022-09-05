import numpy as np
import pandas as pd
import requests

PREDICT_FILE = "preds.csv"
TEST_FILE = "X_test.csv"
HOME = 'http://127.0.0.1:8080/predict_churn'
NUM_OBSERVATIONS = 5


def read_test_data(test_file):
    """ :param test_file for test data as csv
        :returns DataFrame of test data"""
    return pd.read_csv(test_file, encoding="utf-8")


def read_predictions(prediction_file):
    """ :param prediction_file for prediction saved in train module
        :returns numpy array of the predictions """
    return np.loadtxt(prediction_file, delimiter=',')


def call_service(url, parameters):
    """ function to imitate client requests for server
    return prediction result"""
    response = requests.get(url, params=parameters)
    return response.text


def get_params(data):
    """ Receives one row from test data frame
     Returns dictionary of the values according to required parameters for url"""
    parameters = {'is_male': data['is_male'].values[0],
                  'num_inters': data['num_inters'].values[0],
                  'late_on_payment': data['late_on_payment'].values[0],
                  'age': data['age'].values[0],
                  'years_in_contract': data['years_in_contract'].values[0]
                  }
    return parameters


def main():
    """starting function to define url and generate numbers """
    X_test = read_test_data(TEST_FILE)
    true_values = read_predictions(PREDICT_FILE)
    for n in range(NUM_OBSERVATIONS):
        data = X_test.sample(1)
        index = data.index
        result = call_service(url=HOME, parameters=get_params(data))
        assert int(result) == (int(true_values[index][0]))


if __name__ == '__main__':
    main()
