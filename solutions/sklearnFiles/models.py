from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier



def logisticRegression(tuning=False, parameters=None):
    if tuning:
        model = LogisticRegression()

        gridSearch = GridSearchCV(
            estimator = model,
            param_grid = [{
                'C' : [1, 2, 3, 4, 5],
                'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] 
            }],
            scoring = 'f1',
            n_jobs = -1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            model = LogisticRegression(**parameters)
        else:
            model = LogisticRegression()
        
        return model


def decisionTree(tuning=False, parameters=None):
    if tuning:
        model = DecisionTreeClassifier()

        gridSearch = GridSearchCV(
            estimator = model,
            param_grid = [{
                'criterion' : ['gini', 'entropy'],
                'splitter' : ['best', 'random'],
                'max_depth': [5, 10, 15, 20, 25],
                'max_features' : [5, 10, 15, 20, 25],
                'max_leaf_nodes' : [1, 2, 3, 4, 5] 
            }],
            scoring = 'f1',
            n_jobs = -1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            model = DecisionTreeClassifier(**parameters)
        else:
            model = DecisionTreeClassifier()
        
        return model


def nearestCentroid(tuning=False, parameters=None):
    if tuning:
        print('Nearest Centroid does not support tuning')
    
    else:
        model = None
        if parameters:
            model = NearestCentroid(**parameters)
        else:
            model = NearestCentroid()
        
        return model


def neuralNetwork(tuning=False, parameters=None):
    if tuning:
        model = MLPClassifier()

        gridSearch = GridSearchCV(
            estimator = model,
            param_grid = [{
                'hidden_layer_sizes' : [(100, )],
                'max_iter' : [100],
                'activation' : ['relu', 'identity', 'tanh', 'relu'],
                'solver' : ['lbfgs', 'sgd', 'adam']
            }],
            scoring = 'f1',
            n_jobs = -1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            model = MLPClassifier(**parameters)
        else:
            model = MLPClassifier()
        
        return model


def randomForestClassifier(tuning=False, parameters=None):
    if tuning:
        model = RandomForestClassifier()

        gridSearch = GridSearchCV(
            estimator = model,
            para_grid = [{

            }],
            scoring = 'f1',
            n_jobs = -1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            model = RandomForestClassifier(**parameters)
        else:
            model = RandomForestClassifier()
        
        return model


def svc(tuning=False, parameters=None):
    if tuning:
        model = SVC()

        gridSearch = GridSearchCV(
            estimator = model,
            param_grid = [{
                "C": [1, 2, 3, 4, 5],
                "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                "max_iter": [100, 500, 1000],
                "decision_function_shape" : ['ovo', 'ovr']
            }],
            scoring = 'f1',
            n_jobs = -1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            model = SVC(**parameters)
        else:
            model = SVC()

        return model
