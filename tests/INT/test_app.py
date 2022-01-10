"""Functional test for the flask app."""
import os
import sys
from typing import Dict, Optional

import pytest
from flask import render_template

from API.flask.app import app as flask_app
from API.flask.app import render_result

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)


def render_html(prediction_text: str) -> str:
    """Render the expected HTML page to be returned from the app after a prediction.

    Args:
        prediction_text (str): HTML string with results to be inserted at placeholder in the rendered HTML page

    Returns:
        str: Expected HTML page returned from the app after a prediction
    """
    with flask_app.app_context(), flask_app.test_request_context():
        html = render_template(
            os.path.join("inputFeatures.html"), prediction_text=prediction_text
        )
        return html


def mimic_result(
    result_type: str,
    float_features: Optional[list] = [
        None,
    ],
) -> str:
    """Mimic the expected results of the web app based on the input.

    Args:
        result_type (str): Define expected prediction (positive, negative or error)
        float_features (Optional[list], optional): List containing user input features. Defaults to [None, ].

    Returns:
        str: Expected result in form of a HTML string.
    """
    if result_type == "positive":
        return render_result([True], float_features)
    elif result_type == "negative":
        return render_result([False], float_features)
    elif result_type == "error":
        return render_result("error")


def get_status_code(
    client: flask_app.test_client(), test_input: Dict[str, Dict[str, str]]
) -> int:
    """Return expected status code of a API query.

    Args:
        client (flask_app.test_client): Test client of the flask app on which a query will be applied.
        test_input (Dict[str, Dict[str, str]]): Dict[args, Dict[culmen_length, culmen_depth]]

    Returns:
        int: Expected status code
    """
    if test_input["url"] == "/":
        res = client.get(test_input["url"], data=test_input["args"])
        return res.status_code
    else:
        res = client.post(test_input["url"], data=test_input["args"])
        return res.status_code


def get_data(
    client: flask_app.test_client(), test_input: Dict[str, Dict[str, str]]
) -> str:
    """Return expected HTML string finally displayed in in the app.

    Args:
        client (flask_app.test_client): Test client of the flask app on which a query will be applied.
        test_input (Dict[str, Dict[str, str]]): Dict[args, Dict[culmen_length, culmen_depth]]

    Returns:
        str: HTML string containing the final result that will be displayed in the browser
    """
    if test_input["url"] == "/":
        res = client.get(test_input["url"], data=test_input["args"])
        return res.get_data(as_text=True)
    else:
        res = client.post(test_input["url"], data=test_input["args"])
        return res.get_data(as_text=True)


@pytest.fixture
def app() -> flask_app:
    """Create flask object for functional testing.

    Yields:
        Iterator[flask_app]: Flask object
    """
    yield flask_app


@pytest.fixture
def client(app: flask_app) -> flask_app.test_client():
    """Create test client for functional testing.

    Args:
        app (flask_app): Flask object

    Returns:
        flask_app.test_client: Flask test client
    """
    return app.test_client()


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            {"url": "/", "args": None},
            {
                "status_code": 200,
                "data": render_html(""),
            },
        ),
        (
            {
                "url": "/predict",
                "args": {"culmen_length": "8.0", "culmen_depth": "3.0"},
            },
            {
                "status_code": 200,
                "data": render_html(mimic_result("positive", [8.0, 3.0])),
            },
        ),
        (
            {
                "url": "/predict",
                "args": {"culmen_length": "1.0", "culmen_depth": "1.0"},
            },
            {
                "status_code": 200,
                "data": render_html(mimic_result("positive", [1.0, 1.0])),
            },
        ),
        (
            {
                "url": "/predict",
                "args": {"culmen_length": "50.0", "culmen_depth": "1.0"},
            },
            {
                "status_code": 200,
                "data": render_html(mimic_result("negative", [50.0, 1.0])),
            },
        ),
        (
            {"url": "/predict", "args": {"culmen_length": "80", "culmen_depth": "1.5"}},
            {
                "status_code": 200,
                "data": render_html(mimic_result("negative", [80.0, 1.5])),
            },
        ),
        (
            {
                "url": "/predict",
                "args": {"culmen_length": "five", "culmen_depth": "1.0"},
            },
            {
                "status_code": 200,
                "data": render_html(mimic_result("error")),
            },
        ),
    ],
)
def test_app(
    client: flask_app.test_client(),
    test_input: Dict[str, Dict[str, str]],
    expected: Dict[int, mimic_result("str")],
):
    """Test app by functional tests. Raise an AssertionError if test fails.

    Args:
        client (flask_app.test_client): Flask test client
        test_input (Dict[str, Dict[str, str]]): Dict[args, Dict[culmen_length, culmen_depth]]
        expected (Dict[int, mimic_result): Dict[status_code, mimic_result('positive|netagtive|error')
    """
    assert get_status_code(client, test_input) == expected["status_code"]
    assert get_data(client, test_input) == expected["data"]
