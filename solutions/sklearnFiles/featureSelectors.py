from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest


def defineFeatureSelector(selectionMethod, nFeatures):
    if selectionMethod:
        selectionMethod = selectionMethod.lower()
    
        if selectionMethod == 'c' or selectionMethod == 'chi' or selectionMethod == 'chi2':
            return SelectKBest(chi2, k=nFeatures)
        elif selectionMethod == 'f' or selectionMethod == 'f_classif':
            return SelectKBest(chi2, k=nFeatures)
        elif selectionMethod == 'm' or selectionMethod == 'mutual' or selectionMethod == 'mutal_info_classif':
            return SelectKBest(chi2, k=nFeatures)
        
        else:
            print('\nERROR: Feature Selection Method \'{0}\' not recongnised. Acceptable inputs are \'chi2\', \'f_classif\', or \'mutal_info_classif\'.\nExecution has continued without feature selection.\n'.format(selectionMethod))
            return None
    
    return None
