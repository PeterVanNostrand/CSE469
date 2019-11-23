
import treeplot
import sys

def loadDataSet(filepath):
    '''
    Returns
    -----------------
    data: 2-D list
        each row is the feature and label of one instance
    featNames: 1-D list
        feature names
    '''
    data=[]
    featNames = None
    fr = open(filepath)
    for (i,line) in enumerate(fr.readlines()):
        array=line.strip().split(',')
        if i == 0:
            featNames = array[:-1]
        else:
            data.append(array)
    return data, featNames


def splitData(dataSet, axis, value):
    '''
    Split the dataset based on the given axis and feature value

    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label
    axis: int 
        index of which feature to split on
    value: string
        the feature value to split on

    Returns
    ------------------
    subset: 2-D list 
        the subset of data by selecting the instances that have the given feature value
        and removing the given feature columns
    '''
    subset = []
    for instance in dataSet:
        if instance[axis] == value:    # if contains the given feature value
            reducedVec = instance[:axis] + instance[axis+1:] # remove the given axis
            subset.append(reducedVec)
    return subset


def getVals(dataSet, feature_id):
    dict_vals = {}
    for sample in dataSet:
        value = sample[feature_id]
        dict_vals[value] = 0
    return dict_vals.keys()


def calcGini(dataSet):
    label_freq = {}
    for sample in dataSet:
        label = sample[-1]
        if not (label in label_freq):
            label_freq[label] = 0
        label_freq[label] += 1
    
    total_freq = len(dataSet)
    gini = 1
    for freq in label_freq.values():
        gini -= (freq/total_freq)**2

    return gini


def chooseBestFeature(dataSet):
    '''
    choose best feature to split based on Gini index
    
    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label

    Returns
    ------------------
    bestFeatId: int
        index of the best feature
    '''
    n_samples = len(dataSet) # total number of samples
    m_features = len(dataSet[0]) - 1 # number of features (ignore last col which is the label)
    gain = [1] * m_features
    for i in range(0, m_features):
        gain[i] = calcGini(dataSet)
        for value in getVals(dataSet, i):
            subset = splitData(dataSet, i, value) 
            n_subset = len(subset)
            gini_subset = calcGini(subset)
            gain[i] -= (n_subset / n_samples) * gini_subset

    bestFeatId = gain.index(max(gain))
    return bestFeatId


def stopCriteria(dataSet):
    '''
    Criteria to stop splitting: 
    1) if all the class labels are the same, then return the class label;
    2) if there are no more features to split, then return the majority label of the subset.

    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label

    Returns
    ------------------
    assignedLabel: string
        if satisfying stop criteria, assignedLabel is the assigned class label;
        else, assignedLabel is None 
    '''
    # Count the frequency of all labels
    label_freq = {}
    for sample in dataSet:
        # Get the label of the sample
        label = sample[-1]
        # If its a novel label, start with frequency 0
        if not (label in label_freq):
            label_freq[label] = 0
        # Increment the frequency
        label_freq[label] += 1

    # Find the most frequent label
    max_freq = 0
    assignedLabel = None
    for label, freq in label_freq.items():
        if freq > max_freq:
            max_freq = freq
            assignedLabel = label

    # If there is theres more than one label and features to be split
    if max_freq!=len(dataSet) and len(dataSet[0])>1:
        assignedLabel = None

    return assignedLabel


def buildTree(dataSet, featNames):
    '''
    Build the decision tree

    Parameters
    -----------------
    dataSet: 2-D list
        [n'_sampels, m'_features + 1]
        the last column is class label

    Returns
    ------------------
        myTree: nested dictionary
    '''
    assignedLabel = stopCriteria(dataSet)
    if assignedLabel:
        return assignedLabel

    bestFeatId = chooseBestFeature(dataSet)
    bestFeatName = featNames[bestFeatId]

    myTree = {bestFeatName:{}}
    subFeatName = featNames[:]
    del(subFeatName[bestFeatId])
    featValues = [d[bestFeatId] for d in dataSet]
    uniqueVals = list(set(featValues))
    for value in uniqueVals:
        myTree[bestFeatName][value] = buildTree(splitData(dataSet, bestFeatId, value), subFeatName)
    
    return myTree



if __name__ == "__main__":
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
    else:
        filepath = "golf.csv"
    data, featNames = loadDataSet(filepath)
    dtTree = buildTree(data, featNames)
    # print (dtTree) 
    treeplot.createPlot(dtTree)
    