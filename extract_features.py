import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage import io, color, data, measure, util, segmentation, exposure
import skimage
import operator
import skimage.filters as filters
import argparse
import glob
import os
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser()
parser.add_argument("path", help='path to images file')
args=parser.parse_args()


def getLabelOfLeaf(imagePath):
    img = io.imread(imagePath)
    black_white_treshold = 0.6
    crop = 115
    min_size_of_label = 500

    hsv_img = color.rgb2hsv(img)
    hue_img = hsv_img[:, :, 0] 
    value_img = hsv_img[:, :, 2]
    cropped_img = util.crop(value_img, ((0, crop), (0, crop)))
    v_min, v_max = np.percentile(cropped_img, (1, 99))
    better_contrast = exposure.rescale_intensity(cropped_img, in_range=(v_min, v_max))
    black_white_img = np.where(np.asarray(better_contrast) < black_white_treshold, 1, 0)
    clear_border_image = segmentation.clear_border(black_white_img)        
    label_image = measure.label(clear_border_image)

    return label_image


def collectLeafInformations(imagePath, leafDictInformations):
    labelOfLeaf = getLabelOfLeaf(imagePath)
    regions = skimage.measure.regionprops(labelOfLeaf)
    
    regionsDict = {}
    for index, region in enumerate(regions):
        regionsDict[index] = region.area
    
    leafIndex = max(regionsDict.items(), key=operator.itemgetter(1))[0]
    leaf = regions[leafIndex]

    fileName = os.path.basename(imagePath)
    folderName = os.path.basename(os.path.dirname(imagePath))
    leafDictInformations[fileName] = {
        "folder_name": folderName,
        "area": leaf.area,
        "bbox_area": leaf.bbox_area,
        "perimeter": leaf.perimeter,
        "convex_area": leaf.convex_area,
        "extent": leaf.extent,
        "eccentricity": leaf.eccentricity,
        "equivalent_diameter": leaf.equivalent_diameter,
        "major_axis_length": leaf.major_axis_length,
        "minor_axis_length": leaf.minor_axis_length, 
        "solidity": leaf.solidity
    }
    

def getListOfFiles(path):
    listOfFile = os.listdir(path)
    allPaths = list()

    for entry in listOfFile:
        fullPath = os.path.join(path, entry)
        allPaths.append(fullPath)
        
    return allPaths


def writeToOutputFile(leafDictInformations, outputFile):
    header = 'leaf_name\tleaf_type\tarea\tbbox_area\tperimeter\tconvex_area\textent\teccenticity\tequivalent_diameter\tmajor_axis_length\tminor_axis_length\tsolidity\n'
    outputFile.write(header)
    for leaf_name in leafDictInformations.keys():
            outputFile.write('{}\t{}\n'.format(leaf_name, '\t'.join(str(x) for x in leafDictInformations[leaf_name].values())))


def get_leaf_features(inputPath, outputFile):
    allPaths = getListOfFiles(inputPath)
    leafDictInformations = {}

    for path in allPaths:
        listOfImageFilesPath = getListOfFiles(path)
        for imageFilePath in listOfImageFilesPath:
            collectLeafInformations(imageFilePath, leafDictInformations)
    writeToOutputFile(leafDictInformations, outputFile)


if __name__ == "__main__":
    inputPath = args.path
    outputFile = open('leaf_features.txt', 'w')
    get_leaf_features(inputPath, outputFile)
