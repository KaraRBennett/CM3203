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
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
                'random_state' : [0]
            }],
            scoring = 'f1',
            n_jobs = -1,
            verbose=1
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
                'max_depth': [10, 100, 1000],
                'max_features' : [10, 100, 1000],
                'random_state' : [0]
            }],
            scoring = 'f1',
            n_jobs = -1,
            verbose=1
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
                'hidden_layer_sizes' : [(100, ), (100, 50, ), (100, 50, 20, )],
                'activation' : ['relu', 'tanh'],
                'solver' : ['lbfgs', 'sgd', 'adam'],
                'alpha' : [1e-1, 1e-2, 1e-3],
                'random_state' : [0]
            }],
            scoring = 'f1',
            n_jobs = -1,
            verbose=1
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
            param_grid = [{
                'criterion' : ['gini', 'entropy'],
                'max_depth': [10, 100, 1000],
                'max_features' : [10, 100, 1000],
                'n_estimators' : [10, 100, 1000],
                'random_state' : [0]
            }],
            scoring = 'f1',
            n_jobs = -1,
            verbose=1
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
                'C': [0.001, 0.01, 0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'sigmoid'],
                'max_iter': [100, 1000, 10000],
                'random_state' : [0]
            }],
            scoring = 'f1',
            n_jobs = -1,
            verbose=1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            model = SVC(**parameters)
        else:
            model = SVC()

        return model
