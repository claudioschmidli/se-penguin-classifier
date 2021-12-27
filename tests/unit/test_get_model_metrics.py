import os
import sys
import pickle
import pytest

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)
from model import Main  # noqa: E402


@pytest.fixture(scope="module")
def initialize():
    # set up
    yield None
    # tearDown


def get_testdata() -> list:
    """Loads the test data for this pytest test from a pickle file.

    Returns:
        list: List of tuples  for each test case
    """
    test_data_path = os.path.join(currentdir, "testdata", "get_model_metrics.pkl")
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


def test_get_model_metrics(testparameters):
    """Pytest test for the function get_model_metrics. Raisses an AssertionError if test fails.

    Args:
        testparameters (tuple): First item y_tain, second item y_pred_train and third irem dictionary containing expected output.
    """
    y_test = testparameters[0]
    y_pred_test = testparameters[1]
    expectation = testparameters[2]
    assert Main.get_model_metrics(y_test, y_pred_test) == expectation


def pytest_generate_tests(metafunc):
    """Loads the test data using the function get_testdata() and parametrizes the data for the pytest test.

    Args:
        metafunc ([type? Wie kann ich das rausfinden??]):
    """

    if "testparameters" in metafunc.fixturenames:
        testdata = get_testdata()
        metafunc.parametrize("testparameters", testdata)
