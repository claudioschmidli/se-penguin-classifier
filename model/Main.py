# -*- coding: utf-8 -*-

#Import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#Set constants
plt.style.use('seaborn-muted')

def plot_features(X_train, X_test,  y_train, y_test, feature1, feature2):
    # Anzeige der Trainings (schwarz) - bzw. Testdaten (rot) gemäss den gewählten Prädiktoren und Klassen.
    plt.figure(figsize=(6, 6))
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, edgecolor='k')
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, edgecolor='r')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    
def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    markers = ('o', 'o', 'o', 'o', 'o')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() , X[:, 0].max() 
    x2_min, x2_max = X[:, 1].min() , X[:, 1].max() 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl,
                    edgecolor='none')
        
    _ = ax.set_ylabel('predicted (y)')
    _ = ax.set_xlabel('input (x)')
    
def print_confusion_matrix(y,y_pred):
    matrix = metrics.confusion_matrix(y, y_pred)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    tp = matrix[1,1] 
    tn = matrix[0,0] 
    fn = matrix[1,0] 
    fp = matrix[0,1] 

    # FPR = FP / total negative samples = FP / (FP + TN)
    fpr = fp / (fp + tn)


    # Precision = TP / predicted positive samples = TP / (TP + FP)
    precision = tp / (tp + fp) 
    precision_score = metrics.precision_score(y, y_pred, average=None)
    print('Precision =', precision, precision_score)

    # Recall = TP / actual positive samples = TP / (TP + FN)
    recall = tp / (tp + fn)
    recall_score = metrics.recall_score(y, y_pred, average=None)
    print('Recall = ', recall, recall_score)

    f1_m = 2 * (precision * recall) / (precision + recall)
    f1 = metrics.f1_score(y, y_pred, average=None)


    print('Confusion matrix:\n', matrix)
    print('True positives = ', tp, ', false positives = ', fp)
    print('False negatives = ', fn, ', true negatives = ', tn)
    print('False-Positive-Rate = ', fpr)
    print('F1 score sklearn = ', f1, ' F1 score manuell = ', f1_m)
    
def show_confusion_matrix(X,y,y_pred, clf):

    matrix = confusion_matrix(y,y_pred)
    matrix1 = plot_confusion_matrix(clf, X, y)
    matrix1.ax_.set_title('Confusion Matrix', color='black')
    plt.xlabel('Predicted Genre', color='black')
    plt.ylabel('True Genre', color='black')
    plt.gcf().axes[0].tick_params(colors='black')
    plt.gcf().axes[1].tick_params(colors='black')
    plt.gcf().set_size_inches(10,5)
    plt.show()
  
    
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
        self.logr.fit(data.X_train,data.y_train) 
        return self.logr
    
    
    

if __name__ == "__main__":
    #select data
    df = pd.read_csv('data/penguins.csv')
    CLASS = df.Species.unique()[0] # 0: Adelie
    X_VARIABLES=['Culmen Length (mm)', 'Culmen Depth (mm)'] # Spiele mit unterschiedlichen Variablen
    Y_VARIABLE='Species'
    data=data(df, CLASS, X_VARIABLES, Y_VARIABLE)
    
    #Plot data
    print(f'{data.X_train.shape} training samples; {data.X_test.shape} test samples;')
    plot_features(data.X_train, data.X_test, data.y_train, data.y_test, X_VARIABLES[0], X_VARIABLES[1])
    
    #Make model
    model=model(data)
    
    #Check model
    plot_decision_regions(data.X, data.y, model.logr)
    # evaluate training data
    print('Train')
    print_confusion_matrix(data.y_train, model.y_pred_train)
    show_confusion_matrix(data.X_train, data.y_train, model.y_pred_train, model.logr)
    print('Test')
    # Evaluate test data
    print_confusion_matrix(data.y_test, model.y_pred_test)
    show_confusion_matrix(data.X_test, data.y_test, model.y_pred_test, model.logr)
    
    
    
