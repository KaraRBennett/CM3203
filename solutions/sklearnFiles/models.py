from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier


def decisionTree(tuning=False, parameters=None):
    if tuning:
        model = DecisionTreeClassifier()

        gridSearch = GridSearchCV(
            estimator = model,
            param_grid = [{
                'criterion' : ['gini', 'entropy'],
                'splitter' : ['best', 'random'],
                'max_depth': [10, 100, 1000],
                'max_features' : [0.1, 0.25, 0.5],
                'random_state' : [0]
            }],
            scoring = 'f1',
            n_jobs = -1,
            verbose = 1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            setDefaultRandomState(parameters)
            model = DecisionTreeClassifier(**parameters)
        else:
            model = DecisionTreeClassifier(random_state=0)
        
        return model
        

def logisticRegression(tuning=False, parameters=None):
    if tuning:
        model = LogisticRegression()

        gridSearch = GridSearchCV(
            estimator = model,
            param_grid = [{
                'C': [0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 25],
                'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
                'random_state' : [0]
            }],
            scoring = 'f1',
            n_jobs = -1,
            verbose = 1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            setDefaultRandomState(parameters)
            model = LogisticRegression(**parameters)
        else:
            model = LogisticRegression(random_state=0)
        
        return model


def nearestCentroid(tuning=False, parameters=None):
    if tuning:
        model = NearestCentroid()

        gridSearch = GridSearchCV(
            estimator = model,
            param_grid = [{
                'metric' : ['cityblock', 'cosine', 'euclidean', 'manhattan']
            }],
            scoring = 'f1',
            n_jobs = -1,
            verbose = 1
        )
        return gridSearch
        
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
            verbose = 1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            setDefaultRandomState(parameters)
            model = MLPClassifier(**parameters)
        else:
            model = MLPClassifier(random_state=0)
        
        return model


def randomForestClassifier(tuning=False, parameters=None):
    if tuning:
        model = RandomForestClassifier()

        gridSearch = GridSearchCV(
            estimator = model,
            param_grid = [{
                'criterion' : ['entropy'],
                'max_depth': [100, 1000],
                'max_features' : [0.25],
                'n_estimators' : [100, 500, 1000],
                'random_state' : [0]
            }],
            scoring = 'f1',
            n_jobs = -1,
            verbose = 1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            setDefaultRandomState(parameters)
            model = RandomForestClassifier(**parameters)
        else:
            model = RandomForestClassifier(random_state=0)
        
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
            verbose = 1
        )
        return gridSearch
    
    else:
        model = None
        if parameters:
            setDefaultRandomState(parameters)
            model = SVC(**parameters)
        else:
            model = SVC(random_state=0)

        return model


def setDefaultRandomState(parameters):
    if 'random_state' not in parameters:
        parameters.update( {'random_state' : 0} )