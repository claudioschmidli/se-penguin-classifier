"""Create data for test_get_model_metrics()."""
import os
import pickle
import sys

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARRENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARRENT_DIR)
from model import classifier  # noqa: E402


def create_pytest_data():
    """Create test data for unit testing with pytest."""
    testdata = {}
    for i in range(0, 3):
        df = pd.read_csv("model/data/penguins.csv")
        CLASS = df.Species.unique()[i]
        X_variables = [
            "Culmen Length (mm)",
            "Culmen Depth (mm)",
        ]
        y_variables = "Species"
        data = classifier.Data(df, CLASS, X_variables, y_variables)
        model = classifier.Model(data)
        model_metrics = classifier.get_model_metrics(data.y_test, model.y_pred_test)
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
