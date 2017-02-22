import sys
import ast
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss, make_scorer
from scipy.stats import expon, uniform, randint
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV

if len(sys.argv) != 6:
    print "Usage: python {} <preprocess_train_file> <preprocess_test_file> <features_no> <samples_no> <targets>".format(sys.argv[0])
    exit(0)

preprocessTrainFilePath = sys.argv[1]
preprocessTestFilePath = sys.argv[2]
featuresNo = int(sys.argv[3])   # number of features after dimension reduction
samplesNo = int(sys.argv[4])    # number of samples for randomized search
cvFolds = 5

# Load targets
targetsFile = open(sys.argv[5], 'r')
targets = [[int(i) for i in line.strip().split(',')] for line in targetsFile]
targets = np.array(targets)

# Load train features
preprocessTrainFile = open(preprocessTrainFilePath, 'r')
featureMatrixTrain = []
for line in preprocessTrainFile:
    featureMatrixTrain.append(ast.literal_eval(line))
featureMatrixTrain = np.array(featureMatrixTrain)
print "Features number:", len(featureMatrixTrain[0])

# Load test features
preprocessTestFile = open(preprocessTestFilePath, 'r')
featureMatrixTest = []
for line in preprocessTestFile:
    featureMatrixTest.append(ast.literal_eval(line))
featureMatrixTest = np.array(featureMatrixTest)

# FIXME
# Select features (use either this or PCA)
# dirty fix, concatenate the labels to end up with 1D targets array
targetsSelection = [4*t[0] + 2*t[1] + t[2] for t in targets]
selection = SelectKBest(k=featuresNo)
featureMatrixTrain = selection.fit_transform(featureMatrixTrain, targetsSelection)
print "Features after SelectKBest:", len(featureMatrixTrain[0])
featureMatrixTest = selection.transform(featureMatrixTest)

# PCA
#pca = PCA(svd_solver="auto", n_components=featuresNo, whiten=True)
#featureMatrixTrain = pca.fit_transform(featureMatrixTrain)
#print "Features after PCA:", len(featureMatrixTrain[0])
#featureMatrixTest = pca.transform(featureMatrixTest)

# Scale features (not needed if we whiten with PCA)
scaler = StandardScaler()
featureMatrixTrain = scaler.fit_transform(featureMatrixTrain)
featureMatrixTest = scaler.transform(featureMatrixTest)

# One-vs-all classifier
# OneVsRestClassifier: (VotingClassifier: (SVC, KNeighborsClassifier, RandomForestClassifier))
svc = SVC(probability=True)
kNeighbors = KNeighborsClassifier()
rForest = RandomForestClassifier()
estimators = [("svc", svc), ("kNeighbors", kNeighbors), ("rForest", rForest)]
voting = VotingClassifier(estimators=estimators, voting = "soft",weights=[4,1,1])
onevsall = OneVsRestClassifier(voting, n_jobs=-1)

# Do randomized search to determine the hyperparameters
hammingLoss = make_scorer(hamming_loss, greater_is_better=False)
parameters = {
    "estimator__svc__kernel": ["rbf"], 
    "estimator__svc__C": expon(scale=100), 
    "estimator__svc__gamma": expon(scale=0.1), 
    "estimator__svc__degree": [2, 3, 4, 5],
    "estimator__kNeighbors__n_neighbors": [5,7],
    "estimator__rForest__n_estimators": randint(1, 30),
    "estimator__rForest__max_features": randint(1, featuresNo),
    "estimator__rForest__min_samples_leaf": randint(1, 100)
}
onevsall_search = RandomizedSearchCV(onevsall, param_distributions=parameters, cv=cvFolds, scoring=hammingLoss, n_jobs=-1, n_iter=samplesNo)
onevsall_search.fit(featureMatrixTrain, targets)
best = onevsall_search.best_index_
print "One-vs-all best score: {} ({}) {}".format(onevsall_search.cv_results_["mean_test_score"][best], onevsall_search.cv_results_["std_test_score"][best], onevsall_search.best_params_)

# Make predictions
predictions = onevsall_search.predict(featureMatrixTest)

# Write to submission file
submissionFile = "final_sub.csv"
f = open(submissionFile, 'w')
f.write("ID,Sample,Label,Predicted\n")
labels = ["gender", "age", "health"]
predictionBoolean = ["FALSE", "TRUE"]
ID = 0
for sample in range(len(predictions)):
    p = predictions[sample]
    for label in range(len(labels)):
        f.write("{},{},{},{}\n".format(ID + label, sample, labels[label], predictionBoolean[predictions[sample][label]]))
    ID += 3
f.close()

