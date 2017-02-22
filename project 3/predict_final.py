import os

os.system("python src/project3_feature_generator.py data/set_train train 1 278 30,140,40,180,50,140 3 3 3 10 100 1600 > features_3_10.txt")
os.system("python src/project3_feature_generator.py data/set_test test 1 138 30,140,40,180,50,140 3 3 3 10 100 1600 > features_3_10_test.txt")

os.system("python src/project3_submission.py features_3_10.txt features_3_10_test.txt 200 50 data/targets.csv")

