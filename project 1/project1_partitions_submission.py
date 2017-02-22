import sys
import ast
import numpy as np
from sklearn import linear_model

if len(sys.argv) != 4:
    print "Usage: python {} <preprocess_train_file> <preprocess_test_file> [linear|ridge|lasso]".format(sys.argv[0])
    exit(0)

preprocessTrainFilePath = sys.argv[1]
preprocessTestFilePath = sys.argv[2]
model = sys.argv[3]

targetsFile = open("targets.csv", 'r')
targetsTrain = [int(line.strip()) for line in targetsFile]

preprocessTrainFile = open(preprocessTrainFilePath, 'r')
featureMatrixTrain = []
for line in preprocessTrainFile:
    partitions = ast.literal_eval(line)
    flatPartitions = [value for x in partitions for y in x for value in y]
    featureMatrixTrain.append(flatPartitions)

preprocessTestFile = open(preprocessTestFilePath, 'r')
featureMatrixTest = []
for line in preprocessTestFile:
    partitions = ast.literal_eval(line)
    flatPartitions = [value for x in partitions for y in x for value in y]
    featureMatrixTest.append(flatPartitions)

if model == "linear":
    reg = linear_model.LinearRegression()
    reg.fit(featureMatrixTrain, targetsTrain)
    predictions = reg.predict(featureMatrixTest)
elif model == "ridge":
    ridgeReg = linear_model.RidgeCV()
    ridgeReg.fit(featureMatrixTrain, targetsTrain)
    predictions = ridgeReg.predict(featureMatrixTest)
elif model == "lasso":
    lassoReg = linear_model.LassoCV()
    lassoReg.fit(featureMatrixTrain, targetsTrain)
    predictions = lassoReg.predict(featureMatrixTest)

print "ID,Prediction"
for i in range(len(predictions)):
    age = int(round(predictions[i]))
    if age < 18:
        age = 18
    if age > 100:
        age = 100
    print "{},{}".format(i + 1, age)

