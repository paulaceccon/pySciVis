"""
This script presents some visualization for machine learning algorithms. Such visualizations are based on 
the ones presented in the "Applied Machine Learning" (https://www.coursera.org/learn/python-machine-learning)
course
Docs:
    http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_iris.html
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html#sphx-glr-auto-examples-ensemble-plot-voting-decision-regions-py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

def plot_hyperplane(c, color, fitted_model):
    """
    Plot the one-against-all classifiers for the given model.

    Parameters
    --------------

    c : index of the hyperplane to be plot
    color : color to be used when drawing the line
    fitted_model : the fitted model
    """
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    try:
        coef = fitted_model.coef_
        intercept = fitted_model.intercept_
    except:
        return

    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color, zorder=3)


def plot_decision_boundary(X, y, fitted_model, features, targets):
    """
    This function plots a model decision boundary as well as it tries to plot 
    the decision probabilities, if available.
    Requires a model fitted with two features only.

    Parameters
    --------------

    X : the data to learn
    y : the classification labels
    fitted_model : the fitted model
    """
    cmap = plt.get_cmap('Set3')
    prob = cmap
    colors = [cmap(i) for i in np.linspace(0, 1, len(fitted_model.classes_))]

    plt.figure(figsize=(10, 5))
    for i, plot_type in enumerate(['Decision Boundary', 'Decision Probabilities']):
        plt.subplot(1, 2, i+1)

        mesh_step_size = 0.01  # step size in the mesh
        x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))
        # First plot, predicted results using the given model
        if i == 0:
            Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])
            for h, color in zip(fitted_model.classes_, colors):
                plot_hyperplane(h, color, fitted_model) 
        # Second plot, predicted probabilities using the given model
        else:
            prob = 'RdYlBu_r'
            try:
                Z = fitted_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            except:
                plt.text(0.4, 0.5, 'Probabilities Unavailable', horizontalalignment='center', 
                         verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
                plt.axis('off')
                break
        Z = Z.reshape(xx.shape)
        # Display Z
        plt.imshow(Z, interpolation='nearest', cmap=prob, alpha=0.5, 
                   extent=(x_min, x_max, y_min, y_max), origin='lower', zorder=1)
        # Plot the data points
        for k, color in zip(fitted_model.classes_, colors):
            idx = np.where(y == k)
            plt.scatter(X[idx, 0], X[idx, 1], facecolor=color, edgecolor='k', lw=1,
                        label=targets[k], cmap=cmap, alpha=0.8, zorder=2)
        if i == 0:
            plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.3), ncol=len(targets))  
        plt.title(plot_type + '\n' + 
                  str(fitted_model).split('(')[0]+ ' Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 5)))
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.gca().set_aspect('equal')      
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.08, wspace=0.2)
    plt.show()


def plot_scatter_matrix(df, output_path, hue=None):
    """
    This function plots scatter matrix between each pair
    of features in the data.

    Parameters
    --------------

    df : data frame of data features and target
    output_path : saving path
    """
    fig = sns.pairplot(df, hue=hue)
    fig.fig.subplots_adjust(right=0.8)
    fig.savefig(output_path+'.pdf', bbox_inches='tight', dpi=400)


def plot_confusion_matrix(y_test, y_predicted, targets, model=''):
    """
    This function plots a confusion matrix.

    Parameters
    --------------

    y_test : ground truth
    y_predicted : predicted targets
    target : possible targets
    model : name of the fitted model
    """
    confusion_mc = confusion_matrix(y_test, y_predicted)
    df_cm = pd.DataFrame(confusion_mc, index = targets, columns = targets)

    plt.figure(figsize = (5.5,4))
    sns.heatmap(df_cm, annot=True)
    plt.title(model+' \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predicted)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__': 
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    scaler = preprocessing.StandardScaler()
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])
    df = df.apply(pd.to_numeric, args=('coerce',))
    for i, v in enumerate(df['target'].unique()):
        df["target"][df["target"]==v] = iris.target_names[i]

    plot_scatter_matrix(df, 'scatter_matrix', 'target')

    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    clf4 = SGDClassifier(alpha=0.001, n_iter=100).fit(X, y)

    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)
    clf4.fit(X_train, y_train)

    plot_decision_boundary(X, y, clf1, iris.feature_names[:2], iris.target_names)
    plot_decision_boundary(X, y, clf2, iris.feature_names[:2], iris.target_names)
    plot_decision_boundary(X, y, clf3, iris.feature_names[:2], iris.target_names)
    plot_decision_boundary(X, y, clf4, iris.feature_names[:2], iris.target_names)

    y_predicted = clf2.predict(X_test)
    plot_confusion_matrix(y_test, y_predicted, iris.target_names)