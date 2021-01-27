
# Python scripts for extracting leaf features from jpg files and searching the best classification method for leaf detection

## extract_features 

Script extracts leaf features from jpg files using skimage python library. 

### How to run
```python extract_features.py leaf_image_folder/```

## experiments

Script for searching for the best classification method for leaf detection. 
### Tested method
- KNeighborsClassifier()
- MLPClassifier()
- RandomForestClassifier()
- DecisionTreeClassifier()

### How to run 
```python experiments.py```

## classificator 

Script for leaf classification using the best method extracted from experiments.

### How to run
```python classificator.py leaf_image_folder/```

## Requirements

- python 3.6
- numpy
- sklearn
- pyplot
- skimage
