import os
import sys
import pickle
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARRENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARRENT_DIR)
from model import Main  # noqa: E402


def create_pytest_data():
    """Creates test data for unit testing with pytest."""
    testdata = {}
    for i in range(0, 3):
        df = pd.read_csv("model/data/penguins.csv")
        CLASS = df.Species.unique()[i]
        X_VARIABLES = [
            "Culmen Length (mm)",
            "Culmen Depth (mm)",
        ]
        Y_VARIABLE = "Species"
        data = Main.Data(df, CLASS, X_VARIABLES, Y_VARIABLE)
        model = Main.Model(data)
        model_metrics = Main.get_model_metrics(data.y_test, model.y_pred_test)
        testcase = {
            "data": data,
            "model": model,
            "metrics": model_metrics,
            "CLASS": CLASS,
        }
        testdata[i] = testcase
    test_data_path = os.path.join(
        PARRENT_DIR, "tests", "unit", "testdata", "get_model_metrics.pkl"
    )
    pickle.dump(testdata, open(test_data_path, "wb"))


if __name__ == "__main__":
    create_pytest_data()
