"""Functions and classes for creating a linear model for penguin classification."""
import os
import pickle

import numpy as np
import pandas as pd
import sklearn.linear_model
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def plot_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature1: str,
    feature2: str,
) -> sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay:
    """Create a scatter plot of the culmen length and culmen depth for each data point.

    Set the body colors by the predictions and the edge colors by test/train data set.

    Args:
        X_train (np.ndarray): Numpy array containing two features.
        X_test (np.ndarray): Numpy array containing two features.
        y_train (np.ndarray): Numpy array filled with predictions (true and false booleans).
        y_test (np.ndarray): Numpy array filled with predictions (true and false booleans).
        feature1 (str): Label for x axis
        feature2 (str): label for y axis

    Returns:
        sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay: Matplot fig object.
        Use e.g. plt.show() after calling the function to show it.
    """
    local_fig = plt.figure(figsize=(6, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="black")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="red")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    return local_fig


def plot_decision_regions(
    X: np.ndarray,
    y: np.ndarray,
    classifier: sklearn.linear_model._logistic.LogisticRegression,
    resolution: float = 0.02,
) -> sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay:
    """Create a scatter plot of the data set and add the found decision bounndary to it.

    Args:
        X (np.ndarray): Numpy array containing two features.
        y (np.ndarray): Numpy array filled with predictions (true and false booleans).
        classifier (sklearn.linear_model._logistic.LogisticRegression): Liner model from sklearn.
        resolution (float, optional): Resolution of the plotted decision boundary. Defaults to 0.02.

    Returns:
        sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay: Matplot fig object.
        Use e.g. plt.show() after calling the function to show it.
    """
    markers = ("o", "o", "o", "o", "o")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    local_fig, local_ax = plt.subplots(figsize=(6, 6))
    local_ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    local_ax.set_xlim(xx1.min(), xx1.max())
    local_ax.set_ylim(xx2.min(), xx2.max())

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        local_ax.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=[cmap(idx)],
            marker=markers[idx],
            label=cl,
            edgecolor="none",
        )

    _ = local_ax.set_ylabel("predicted (y)")
    _ = local_ax.set_xlabel("input (x)")
    return local_fig, local_ax


def get_model_metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate the model metrics by comparing labeled and predicted data.

    Args:
        y (np.ndarray): Numpy array filled with labels (true and false booleans).
        y_pred (np.ndarray): Numpy array filled with predictions (true and false booleans).

    Returns:
        dict: Dictionary containing different metrics.
    """
    matrix = metrics.confusion_matrix(y, y_pred)
    model_metrics = {}
    model_metrics["tp"] = matrix[1, 1]
    model_metrics["tn"] = matrix[0, 0]
    model_metrics["fn"] = matrix[1, 0]
    model_metrics["fp"] = matrix[0, 1]
    model_metrics["fpr"] = model_metrics["fp"] / (
        model_metrics["fp"] + model_metrics["tn"]
    )  # FPR = False positive rate
    model_metrics["precision score"] = model_metrics["tp"] / (
        model_metrics["tp"] + model_metrics["fp"]
    )
    model_metrics["recall_score"] = model_metrics["tp"] / (
        model_metrics["tp"] + model_metrics["fn"]
    )
    model_metrics["f1 score"] = (
        2
        * (model_metrics["precision score"] * model_metrics["recall_score"])
        / (model_metrics["precision score"] + model_metrics["recall_score"])
    )
    return model_metrics


def show_confusion_matrix(
    X: np.ndarray,
    y: np.ndarray,
    clf: sklearn.linear_model._logistic.LogisticRegression,
) -> sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay:
    """Plot confusion matrix.

    Args:
        X (np.ndarray): Numpy array containing two features.
        y (np.ndarray): Numpy array filled with labels (true and false booleans).
        clf (sklearn.linear_model._logistic.LogisticRegression): Liner model from sklearn.

    Returns:
        sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay: Matplot fig object.
        Use e.g. plt.show() after calling the function to show it.
    """
    matrix = ConfusionMatrixDisplay.from_estimator(clf, X, y)
    matrix.ax_.set_title("Confusion Matrix", color="black")
    plt.xlabel("Predicted Genre", color="black")
    plt.ylabel("True Genre", color="black")
    plt.gcf().axes[0].tick_params(colors="black")
    plt.gcf().axes[1].tick_params(colors="black")
    plt.gcf().set_size_inches(10, 5)
    return matrix


class Data:
    """Class containing all dataset related data."""

    def __init__(
        self,
        df: pd.DataFrame,
        CLASS: str,
        X_variables: list,
        y_variables: str,
    ):
        """Filter the data by CLASS and split the dataset into test and training set.

        Args:
            df (pd.DataFrame): Pandas dataframe.
            CLASS (str): String describing the penguin class.
            X_variables (np.ndarray): Matrix containing selected features to be evaluated.
            y_variables (np.ndarray): Array containing labels (true, false).
        """
        self.df = self.preprocess_df(df, X_variables)
        self.CLASS = CLASS
        self.X_variables = X_variables
        self.y_variables = y_variables
        seed = 1
        self.df_train, self.df_test = train_test_split(self.df, random_state=seed)
        self.X = np.array(self.df[X_variables])
        self.y = np.array(self.df[y_variables] == CLASS)
        self.X_train = np.array(self.df_train[X_variables])
        self.y_train = np.array(self.df_train[y_variables] == CLASS)
        self.X_test = np.array(self.df_test[X_variables])
        self.y_test = np.array(self.df_test[y_variables] == CLASS)

    def preprocess_df(
        self,
        df: pd.DataFrame,
        X_variables: list,
    ) -> pd.DataFrame:
        """Remove rows containing na in the dataset.

        Args:
            df (pd.DataFrame): Pandas dataframe.
            X_variables (np.ndarray): Matrix containing dataset features to be evaluated.
            y_variables (np.ndarray): Array containing dataset labels (true, false).

        Returns:
            pd.DataFrame: Filtered pandas dataframe without na values.
        """
        df = df.dropna(subset=X_variables)
        return df


class Model:
    """Class containing all model related data."""

    def __init__(self, data: Data) -> None:
        """Assign the model data to the class instance.

        Args:
            data (Data): Data object containing the data. See Data class for more information.
        """
        self.logr = self.make_model(data)
        self.y_pred_train = self.logr.predict(data.X_train)
        self.y_pred_test = self.logr.predict(data.X_test)

    def make_model(
        self,
        data: Data,
    ) -> sklearn.linear_model._logistic.LogisticRegression:
        """Create a model from the given data.

        Args:
            data (Data): Data object containing the data. See Data class for more information.

        Returns:
            sklearn.linear_model._logistic.LogisticRegression: Linear model from sklearn.
        """
        self.logr = LogisticRegression()
        self.logr.fit(data.X_train, data.y_train)
        return self.logr


if __name__ == "__main__":
    # select data
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.join(CURRENT_DIR, "data/penguins.csv")
    MODEL_PATH = os.path.join(CURRENT_DIR, "data/model.pkl")
    df = pd.read_csv(DATA_PATH)

    # Define dataset features and filter dataset by penguin class
    CLASS = df.Species.unique()[0]  # 0: Adelie
    X_variables = [
        "Culmen Length (mm)",
        "Culmen Depth (mm)",
    ]  # you can choose here different physical dimensions
    y_variables = "Species"
    data = Data(df, CLASS, X_variables, y_variables)

    # Plot data
    print(
        f"{data.X_train.shape[0]} training samples\n{data.X_test.shape[0]} test samples"
    )
    fig = plot_features(
        data.X_train,
        data.X_test,
        data.y_train,
        data.y_test,
        X_variables[0],
        X_variables[1],
    )

    # Make model
    model = Model(data)

    # Save model to disk
    pickle.dump(model.logr, open(MODEL_PATH, "wb"))

    # Check model
    fig, ax = plot_decision_regions(data.X, data.y, model.logr)

    # Evaluate training data
    print("\nModel metrics train data")
    model_metrics = get_model_metrics(data.y_train, model.y_pred_train)
    for key, value in model_metrics.items():
        print(f"{key}: {value}")
    fig = show_confusion_matrix(data.X_train, data.y_train, model.logr)

    # Evaluate test data
    print("\nModel metrics test data")
    model_metrics = get_model_metrics(data.y_test, model.y_pred_test)
    for key, value in model_metrics.items():
        print(f"{key}: {value}")
    show_confusion_matrix(data.X_test, data.y_test, model.logr)
    # plt.show()
