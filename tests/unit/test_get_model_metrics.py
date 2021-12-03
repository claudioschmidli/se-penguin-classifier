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
    test_data_path = os.path.join(currentdir, "testdata", "get_model_metrics.pkl")
    testdata = pickle.load(open(test_data_path, "rb"))
    # save all test cases in a dictionary and return it
    testdata_parametrized = []
    for i in range(0, 3):
        testdata_parametrized.append(
            (
                testdata[i]["data"].y_train,
                testdata[i]["model"].y_pred_train,
                testdata[i]["metrics"],
            )
        )
    return testdata_parametrized


def test_get_model_metrics(testparameters):
    y_train = testparameters[0]
    y_pred_train = testparameters[1]
    expectation = testparameters[2]
    assert Main.get_model_metrics(y_train, y_pred_train) == expectation


def pytest_generate_tests(metafunc):
    if "testparameters" in metafunc.fixturenames:
        testdata = get_testdata()
        metafunc.parametrize("testparameters", testdata)


# @pytest.mark.parametrize("a,b,expected", [(1,1,0), ])
# class Template:
#    def subtraction(self, initialize, a, b, expected):
#        diff = a - b
#        assert diff == expected
