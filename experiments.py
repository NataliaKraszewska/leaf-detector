import numpy as np
import sklearn.feature_selection as feature_selection
import sklearn.model_selection as model_selection
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def read_data_from_file():
    input_file = open("/home/natalia/inzynieriaMedyczna/zbior_danych_liscie.txt", 'r')
    leaf_data = []
    leaf_target = []
    leaf_target_name = []

    for line in input_file:
        if not line.startswith("leaf_name"):
            line = line.strip().split('\t')
            data = [float(i) for i in line[2:]]
            leaf_data.append(data)
            leaf_target_name.append(line[1])

    targets = []
    [targets.append(x) for x in leaf_target_name if x not in targets]
    targets_with_indexes = {}

    for index, elem in enumerate(targets):
        targets_with_indexes[elem] = index

    for elem in leaf_target_name:
        leaf_target.append(targets_with_indexes[elem])

    leaf_data_array = np.asanyarray(leaf_data, dtype = np.float32)
    leaf_target_array = np.asanyarray(leaf_target)
    leaf_target_name_array = np.asanyarray(leaf_target_name) 

    return leaf_data_array, leaf_target_array, leaf_target_name_array


def test_classificator(data, target, classificator, classificator_name, parameters, output_file, classificator_dict):
    output_file.write(f'\nTest for {classificator_name}:\n')
    number_of_features = len(data[0])

    for column_number in range(1, number_of_features + 1):
        best_data = feature_selection.SelectKBest(score_func = feature_selection.chi2, k=column_number).fit_transform(data, target)
        search = model_selection.GridSearchCV(classificator, parameters, cv=5)  
        search.fit(best_data, target)
        
        print(column_number, search.best_score_)
        output_file.write(f'column_number: {column_number}\n')
        output_file.write(f'score: {search.best_score_}\n')
        output_file.write(f'best_estimator: {search.best_estimator_}\n')
        output_file.write(f'best_params: {search.best_params_}\n')
        output_file.write('\n')

        key = classificator_name + str(column_number)
        classificator_dict[key] = {
            'classificator_name': classificator_name,
            'classificator': classificator,
            'features': column_number, 
            'score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'best_params': search.best_params_
        }


def get_parameters(parameters, classificator_name):
    if classificator_name == 'KNeighborsClassifier':
        best_parameters = {'algorithm': [parameters['algorithm']], 'n_neighbors': [parameters['n_neighbors']], 'p': [parameters['p']], 'leaf_size': [parameters['leaf_size']], 'weights': [parameters['weights']]}
    
    if classificator_name == 'RandomForestClassifier':
        best_parameters = {'n_estimators': [parameters['n_estimators']], 'criterion': [parameters['criterion']], 'max_features': [parameters['max_features']], 'min_samples_split': [parameters['min_samples_split']], 'bootstrap': [parameters['bootstrap']]}

    if classificator_name == 'DecisionTreeClassifier':
        best_parameters = {'criterion': [parameters['criterion']], 'splitter': [parameters['splitter']], 'max_features': [parameters['max_features']], 'min_samples_split': [parameters['min_samples_split']], 'presort': [parameters['presort']]}

    if classificator_name == 'MLPClassifier':
        best_parameters = {'activation': [parameters['activation']], 'solver': [parameters['solver']], 'learning_rate': [parameters['learning_vote']], 'shuffle': [parameters['shuffle']], 'early_stopping': [parameters['early_stopping']]}

    return best_parameters


def find_best_classificator(classificator_dict, data, target):
    best_classificator_key = None
    score = 0
    for key in classificator_dict.keys():
        if classificator_dict[key]['score'] > score:
            best_classificator_key = key
            score = classificator_dict[key]['score']

    column_number = classificator_dict[best_classificator_key]['features']
    parameters = classificator_dict[best_classificator_key]['best_params']
    classificator = classificator_dict[best_classificator_key]['classificator']
    best_data = feature_selection.SelectKBest(score_func = feature_selection.chi2, k=column_number).fit_transform(data, target)
    classificator_name = classificator_dict[best_classificator_key]['classificator_name']

    best_parameters = get_parameters(parameters, classificator_name)
    best_classificator = model_selection.GridSearchCV(classificator, best_parameters, cv=5)
    best_classificator.fit(best_data, target)
    joblib.dump(best_classificator, 'best_classificator.pkl')

    print(f'Best classificator: {classificator_name}')
    print(f'Parameters: {best_parameters}')
    print(f'Score: {score}')
    print(f'Column_number: {column_number}')


if __name__ == "__main__":
    output_file = open('classifier_result2.txt','w+')
    leaf_data_array, leaf_target_array, leaf_target_name = read_data_from_file()
    classificator_dict = {}

    knn = KNeighborsClassifier()
    knn_parameters = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'n_neighbors': list(range(1, 10)), 'p': [1, 2, 3, 4], 'leaf_size': [10, 20, 30, 40, 50], 'weights':['uniform', 'distance']}
    test_classificator(leaf_data_array, leaf_target_array, knn, 'KNeighborsClassifier', knn_parameters, output_file, classificator_dict)
    
    rfc = RandomForestClassifier()
    rfc_parameters = {'n_estimators': list(range(10, 100, 20)), 'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2', None], 'min_samples_split': list(range(2, 6)), 'bootstrap': [True, False]}
    test_classificator(leaf_data_array, leaf_target_array, rfc, 'RandomForestClassifier', rfc_parameters, output_file, classificator_dict)

    dtc = DecisionTreeClassifier()
    dtc_parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_features': ['sqrt', 'log2', None], 'min_samples_split': list(range(2, 6)), 'presort': [True, False]}
    test_classificator(leaf_data_array, leaf_target_array, dtc, 'DecisionTreeClassifier', dtc_parameters, output_file, classificator_dict)

    nn = MLPClassifier()
    nn_parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'learning_rate': ['constant', 'invscaling', 'adaptive'], 'shuffle': [True, False], 'early_stopping': [True, False]}
    test_classificator(leaf_data_array, leaf_target_array, nn, 'MLPClassifier', nn_parameters, output_file, classificator_dict)

    find_best_classificator(classificator_dict, leaf_data_array, leaf_target_array)