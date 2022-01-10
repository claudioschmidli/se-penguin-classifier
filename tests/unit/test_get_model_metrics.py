"""Unit test for the function get_model_metrics() in the module."""
# @ Tipi: This is a alternative approach we used for testing. Please also have a look at the other tests.

import os
import pickle
import sys

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
from model import classifier  # noqa: E402


def get_testdata() -> list:
    """Load the test data for from a pickle file.

    Returns:
        list: List of tuples  for each test case
    """
    test_data_path = os.path.join(CURRENT_DIR, "testdata", "get_model_metrics.pkl")
    testdata = pickle.load(open(test_data_path, "rb"))
    # save all test cases in a dictionary and return it
    testdata_parametrized = []
    for i in range(0, 3):
        testdata_parametrized.append(
            (
                testdata[i]["data"].y_test,
                testdata[i]["model"].y_pred_test,
                testdata[i]["metrics"],
            )
        )
    return testdata_parametrized


def test_get_model_metrics(testparameters: list):
    """Test function get_model_metrics(). Raise an AssertionError if test fails.

    Args:
        testparameters (tuple): First item y_test, second item y_pred_test and third item dictionary containing expected output.
    """
    y_test = testparameters[0]
    y_pred_test = testparameters[1]
    expectation = testparameters[2]
    assert classifier.get_model_metrics(y_test, y_pred_test) == expectation


def pytest_generate_tests(metafunc):
    """Load the test data using the function get_testdata() and parametrize the data for the pytest test.

    Args:
        metafunc (Metafunc): Objects passed to the pytest_generate_tests hook, which help to inspect the test function and
        to generate tests according to test configuration or values specified in the class or module where
        the test function is defined.
    """
    if "testparameters" in metafunc.fixturenames:
        testdata = get_testdata()
        metafunc.parametrize("testparameters", testdata)
