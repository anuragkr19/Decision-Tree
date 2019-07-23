# -*- coding: utf-8 -*-
"""
External Reference:-

https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
https://www.youtube.com/watch?v=LDRbO9a6XPU&t=517s
https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29

"""

import csv
import os
import math


 

header = ["Class", "Cap_size", "Cap_surface","Cap_color","bruises","odor"]

#class used to split dataset based on the answer to the question into true and false branch
class Node:
    
    def __init__(self,question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


#Return no of classcount at the leaf level
class Leaf:
    
    def __init__(self, rows):
        self.classCount = class_counts(rows)


#Question for partitioning the dataset
class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))


#count the number of rows for each class type and return the result in a dictionary type object
def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[0]
        if label not in counts:
            counts[label] = 0
        counts[label] +=1
    return counts

#return list with unique values for a particular column
def getListForCol(dataSet,col):
    
    listTemp = []
    
    for i in range(len(dataSet)):
        listTemp.append(dataSet[i][col])
    
    return set(listTemp)

#fetch training dataset from csv file
def loadTrainingData():
    dataPoints = []
    fileh = open(os.getcwd() + "//code//dt//MushroomTrain.csv",'r')
    try:
        csv_reader = csv.reader(fileh,delimiter=',')
        for row in csv_reader:
            tempList = []
            tempList.append(row[0]) # Class Label
            tempList.append(row[1]) # Cap_Size
            tempList.append(row[2]) # Cap_Surface
            tempList.append(row[3]) # Cap_Color
            tempList.append(row[4]) # Bruises
            #tempList.append(row[5]) # Odor
            dataPoints.append(tempList)
    except:
        pass
    fileh.close()
    return dataPoints

#fetch test dataset from csv file
def loadTestData():
    dataPoints = []
    fileh = open(os.getcwd() + "//code//dt//MushroomTest.csv",'r')
    try:
        csv_reader = csv.reader(fileh,delimiter=',')
        for row in csv_reader:
            tempList = []
            tempList.append(row[0]) # Class Label
            tempList.append(row[1]) # Cap_Size
            tempList.append(row[2]) # Cap_Surface
            tempList.append(row[3]) # Cap_Color
            tempList.append(row[4]) # Bruises
            #tempList.append(row[5]) # Odor
            dataPoints.append(tempList)
    except:
        pass
    fileh.close()
    return dataPoints

#return the question
def getQuestion(colDict,colName,value):
    condition = "=="
    return "%s %s %s?" %(colDict[colName],condition,value) 

#partition dataset based on question into two lisst true_rows,false_rows
def partitionDataSet(dataset,question):
    
    true_rows,false_rows = [],[]
    for row in dataset:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    
    return true_rows,false_rows

#calculate entropy of the dataset
def calculateEntropy(dataSet):
    counts = class_counts(dataSet)
    impurity = 0.0
    probl_of_lbl = 0.0
    try:
        for lbl in counts:
            probl_of_lbl = (counts[lbl]) / float(len(dataSet))
            impurity += ((probl_of_lbl * - (math.log(2,probl_of_lbl))))
    except:
        pass
    return impurity

#calculate info_gain 
def info_gain(left,right,parent_entropy):
    n = len(left) + len(right)
    p_left = len(left)/n
    p_right = len(right)/n
    return (parent_entropy - ((p_left * calculateEntropy(left)) - (p_right) * calculateEntropy(right)))

#returning the best split and best info_gain
def findBestSplit(colDict,dataSet):
    best_gain = 0
    best_ques = None
    parent_entropy = calculateEntropy(dataSet)
    
    n_features = len(dataSet[0])
    gain = 0.0
    
    #first column is class
    for col in range(1,n_features):
        #get unique values for column in list
        uniqueListForCol = getListForCol(dataSet,col)
        
        for value in uniqueListForCol:
            #get question for each value in the uniqueList
            question = Question(col,value)
            
            #split the dataset
            true_rows,false_rows = partitionDataSet(dataSet,question)
            
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            
            #calculate information gain from this split
            gain = info_gain(true_rows, false_rows, parent_entropy)
            
            #selecting the best gain and best question
            if gain >= best_gain:
                best_gain = gain
                best_ques = question
                
                
    return best_gain,best_ques
    
#printing decision tree
def print_decision_Tree(node,spacing = " "):
    
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.classCount)
        return
    
    #Print the question at this node
    print(spacing + str(node.question))
    
    # Call this function recursively on the true branch
    print(spacing + '--> True: ')
    print_decision_Tree(node.true_branch , spacing + "  ")
    
    # Call this function recursively on the false branch
    print(spacing + '--> False: ')
    print_decision_Tree(node.false_branch, spacing + " ")
    
#Build decision tree with dataSet,colDict is a dictionary which consists of key = col_index,value = column_name 
def build_decision_Tree(dataSet,colDict):
    
    gain, question = findBestSplit(colDict,dataSet)
        
    if gain == 0:
        return Leaf(dataSet)
    
    true_rows, false_rows = partitionDataSet(dataSet,question)

    # Recursively build the true branch.
    true_branch = build_decision_Tree(true_rows,colDict)

    # Recursively build the false branch.
    false_branch = build_decision_Tree(false_rows,colDict)
    
    return Node(question,true_branch,false_branch)
       
#classify the record on the node if the row satisfy condition asked in the form of question at this node
def classify(row, node):
   
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.classCount

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch) 
    


#printing leaf node
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    
    return probs

def main():
    
    #fetching training data from csv file in the trainData list
    trainData = loadTrainingData()
    #fetching test data from csv file in the testData list
    testData = loadTestData()
    
    colDict= {}
    colDict[0] = "Class"
    colDict[1] = "Cap_size"
    colDict[2] = "Cap_surface"
    colDict[3] = "Cap_color"
    colDict[4] = "bruises"
    #colDict[5] = "odor"
    
   
    #building decision tree in the training data
    tree = build_decision_Tree(trainData,colDict)
    
    print(' ')
    print("Printing decision tree for training data")
    print(' ')
    #printing decision tree
    print_decision_Tree(tree)
    
    result = 0
    count = 0
    
    print('   ')
    
    #Displaying classification result for training data
    for row in trainData:
        result = print_leaf(classify(row, tree))
        predicted_class = ' '
        for key,val in result.items():
            predicted_class = key
        if row[0] == predicted_class:
            count+=1
        print('Actual ' + row[0] + ' predicted ' + ' ' + predicted_class)
    
    print('   ')
    print('Accuracy for Training Data ' + str((count/len(trainData))*100) + ' %')
    
    countTest = 0
    
    print('   ')
    
    
    # applying learned decision tree on the TestData #

    
    print(' ')

    result = 0
    countTest = 0
    
    #Displaying classification result for test data
    for row in testData:
        result = print_leaf(classify(row, tree))
        predicted_class = ' '
        for key,val in result.items():
            predicted_class = key
        if row[0] == predicted_class:
            countTest+=1
        print('Actual ' + row[0] + ' predicted ' + ' ' + predicted_class)
    
    print('   ')
    print('Accuracy for Test Data ' + str((countTest/len(testData))*100) + ' %')
    
      
    
    

    
main()
    
        
