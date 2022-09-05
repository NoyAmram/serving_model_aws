import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

DATA_FILE = "cellular_churn_greece.csv"
MODEL_FILE = "churn_model.pkl"
PREDICT_FILE = "preds.csv"
TEST_FILE = "X_test.csv"
TARGET = 'churned'


def read_file(file_name):
    """Receives csv file read it with pandas and return as DataFrame """
    df = pd.read_csv(file_name, encoding="utf-8")
    # verify no missing values before return
    assert df.isna().sum().mean() == 0
    return df


def split_df(df, target):
    """Receives df and target name and returns data split to train and test """
    y = df[target]
    X = df.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # verify split is accordingly before return
    assert X_train.shape[0] / X.shape[0] == 0.8
    assert X_test.shape[0] / X.shape[0] == 0.2

    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """Receives train data and create model object
    :returns model fitted on train data"""
    model = RandomForestClassifier()
    model.fit(X_train, np.array(y_train).reshape(-1))
    return model


def save_model(model, output_file):
    """Receives fitted model and save it to a pickle file"""
    with open(output_file, "wb") as f:
        pickle.dump(model, f)


def save_predict(X_test, model, prediction_file):
    """Receives X_test and fitted model
     Returns prediction and save them to a csv file """
    y_pred = np.array(model.predict(X_test))
    np.savetxt(prediction_file, [y_pred], delimiter=',', fmt='%d')
    return y_pred


def print_accuracy(y_test, y_pred):
    """ function receives prediction and true values and prints accuracy score and classification report"""
    print(f"Accuracy of RandomForestClassifier model: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Classification report:\n{classification_report(y_test, y_pred)}")


def main():
    """Starting function to call above functions of the program and save X_test to a file"""
    cellular_df = read_file(DATA_FILE)
    X_train, X_test, y_train, y_test = split_df(cellular_df, TARGET)
    clf_model = train(X_train, y_train)
    save_model(clf_model, MODEL_FILE)
    prediction = save_predict(X_test, clf_model, PREDICT_FILE)
    X_test.to_csv(TEST_FILE, index=False)
    print_accuracy(y_test, prediction)


if __name__ == '__main__':
    main()
