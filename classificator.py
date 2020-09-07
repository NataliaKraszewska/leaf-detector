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
import extract_features
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help='path to images file')
args=parser.parse_args()

def read_data_from_file(input_file):
    leaf_data = []
    leaf_target = []
    leaf_target_name = []
    leaf_file_name = []

    for line in input_file:
        if not line.startswith("leaf_name"):
            line = line.strip().split('\t')
            data = [float(i) for i in line[2:]]
            leaf_data.append(data)
            leaf_target_name.append(line[1])
            leaf_file_name.append(line[0])

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

    return leaf_data_array, leaf_target_array, leaf_target_name_array, targets_with_indexes, leaf_file_name


def get_target_dict_with_index_in_key(targets_with_indexes):
    targets = {}
    for key, item in targets_with_indexes.items():
        targets[item] = key

    return targets


if __name__ == "__main__":
    result_file = open('classification_result.txt', 'w+')
    inputPath = args.path
    outputFile = open('leaf_features.txt', 'w')
    kraszewska_nawrocka_extract_features.get_leaf_features(inputPath, outputFile)
    
    leaf_features_file = open('leaf_features.txt', 'r')
    leaf_data_array, leaf_target_array, leaf_target_name, targets_with_indexes, leaf_file_name = read_data_from_file(leaf_features_file)
    targets = get_target_dict_with_index_in_key(targets_with_indexes)
    classifier = joblib.load('best_classificator.pkl')
    
    for index, leaf in enumerate(leaf_data_array):
        target = leaf_target_array[index]
        result = classifier.predict(leaf.reshape(1,-1))
        file_name = leaf_file_name[index]
        leaf_name = leaf_target_name[index]
        prediction = targets[result[0]]
        print(f'file: {file_name} classification result: {prediction}')
        result_file.write(f'file: {file_name}, leaf_name: {leaf_name} classification result: {prediction}\n')
        
