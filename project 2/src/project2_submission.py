import sys
import ast
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

if len(sys.argv) != 3:
    print "Usage: python {} <preprocess_train_file> <preprocess_test_file>".format(sys.argv[0])
    exit(0)

preprocessTrainFilePath = sys.argv[1]
preprocessTestFilePath = sys.argv[2]

def voteClassification(featureMatrix, targets,testFeatureMatrix): #voting  using SVM, Nearest_neighbours, RandomForest
    clf1= svm.SVC(kernel= 'poly', probability=True,degree=3)
    clf2 = KNeighborsClassifier(n_neighbors=5)
    clf3 = RandomForestClassifier(n_estimators=25)
    clf= VotingClassifier(estimators=[('svm',clf1), ('nei',clf2), ('rf',clf3)], voting = 'soft', weights=[1,1,1])
    clf.fit(featureMatrix, targets)
    return  clf.predict_proba(testFeatureMatrix)

targetsFile = open("targets.csv", 'r')
targetsAll = [int(line.strip()) for line in targetsFile]

preprocessTrainFile = open(preprocessTrainFilePath, 'r')
featureMatrixAll = []
for line in preprocessTrainFile:
    featureMatrixAll.append(ast.literal_eval(line))

preprocessTestFile = open(preprocessTestFilePath, 'r')
featureMatrixTest = []
for line in preprocessTestFile:
    featureMatrixTest.append(ast.literal_eval(line))

skb = SelectKBest(k=100)
skb.fit(np.array(featureMatrixAll),np.array(targetsAll))
featureMatrixAll = np.array(skb.transform(featureMatrixAll))
featureMatrixTest= np.array(skb.transform(featureMatrixTest))

predictions = voteClassification(featureMatrixAll, targetsAll,featureMatrixTest)
print "ID,Prediction"
for i in range(len(predictions)):
    result = predictions[i][1]
    print "{},{}".format(i + 1, result)
