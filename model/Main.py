# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import pickle


def plot_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature1: str,
    feature2: str,
) -> sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay:
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="black")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="red")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    return fig


def plot_decision_regions(
    X: np.ndarray, y: np.ndarray, classifier: str, resolution: float = 0.02
) -> sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay:

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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=[cmap(idx)],
            marker=markers[idx],
            label=cl,
            edgecolor="none",
        )

    _ = ax.set_ylabel("predicted (y)")
    _ = ax.set_xlabel("input (x)")
    return fig, ax


def get_model_metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
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
    y_pred: np.ndarray,
    clf: sklearn.linear_model._logistic.LogisticRegression,
) -> sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay:
    matrix = ConfusionMatrixDisplay.from_estimator(clf, X, y)
    matrix.ax_.set_title("Confusion Matrix", color="black")
    plt.xlabel("Predicted Genre", color="black")
    plt.ylabel("True Genre", color="black")
    plt.gcf().axes[0].tick_params(colors="black")
    plt.gcf().axes[1].tick_params(colors="black")
    plt.gcf().set_size_inches(10, 5)
    return matrix


class data:
    def __init__(self, df, CLASS, X_VARIABLES, Y_VARIABLE):
        self.df = self.preprocess_df(df, X_VARIABLES, Y_VARIABLE)
        self.CLASS = CLASS
        self.X_VARIABLES = X_VARIABLES
        self.Y_VARIABLE = Y_VARIABLE
        seed = 1
        self.df_train, self.df_test = train_test_split(self.df, random_state=seed)
        self.X = np.array(self.df[X_VARIABLES])
        self.y = np.array(self.df[Y_VARIABLE] == CLASS)
        self.X_train = np.array(self.df_train[X_VARIABLES])
        self.y_train = np.array(self.df_train[Y_VARIABLE] == CLASS)
        self.X_test = np.array(self.df_test[X_VARIABLES])
        self.y_test = np.array(self.df_test[Y_VARIABLE] == CLASS)

    def preprocess_df(self, df, X_VARIABLES, Y_VARIABLE):
        df = df.dropna(subset=X_VARIABLES)
        return df


class model:
    def __init__(self, data):
        self.logr = self.make_model(data)
        self.y_pred_train = self.logr.predict(data.X_train)
        self.y_pred_test = self.logr.predict(data.X_test)

    def make_model(self, data):
        self.logr = LogisticRegression()
        self.logr.fit(data.X_train, data.y_train)
        return self.logr


if __name__ == "__main__":
    # select data
    df = pd.read_csv("model/data/penguins.csv")

    # Define dataset features and filter dataset by class
    CLASS = df.Species.unique()[0]  # 0: Adelie
    X_VARIABLES = [
        "Culmen Length (mm)",
        "Culmen Depth (mm)",
    ]  # Spiele mit unterschiedlichen Variablen
    Y_VARIABLE = "Species"
    data = data(df, CLASS, X_VARIABLES, Y_VARIABLE)

    # Plot data
    print(
        f"{data.X_train.shape[0]} training samples\n{data.X_test.shape[0]} test samples"
    )
    fig = plot_features(
        data.X_train,
        data.X_test,
        data.y_train,
        data.y_test,
        X_VARIABLES[0],
        X_VARIABLES[1],
    )

    # Make model
    model = model(data)

    # Save model to disk
    pickle.dump(model.logr, open("model.pkl", "wb"))

    # Load model
    # model = pickle.load(open('model.pkl', 'rb'))

    # Check model
    fig, ax = plot_decision_regions(data.X, data.y, model.logr)

    # Evaluate training data
    print("\nModel metrics train data")
    model_metrics = get_model_metrics(data.y_train, model.y_pred_train)
    for key, value in model_metrics.items():
        print(f"{key}: {value}")
    fig = show_confusion_matrix(
        data.X_train, data.y_train, model.y_pred_train, model.logr
    )

    # Evaluate test data
    print("\nModel metrics test data")
    model_metrics = get_model_metrics(data.y_test, model.y_pred_test)
    for key, value in model_metrics.items():
        print(f"{key}: {value}")
    show_confusion_matrix(data.X_test, data.y_test, model.y_pred_test, model.logr)
    # plt.show()
