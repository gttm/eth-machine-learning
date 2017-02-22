import os

os.system("python src/project2_feature_generator.py data/set_train train 1 278 30,140,40,180,50,140 9 9 9 50 100 1600 > features_9_50.txt")
os.system("python src/project2_feature_generator.py data/set_test test 1 138 30,140,40,180,50,140 9 9 9 50 100 1600 > features_9_50_test.txt")

os.system("python src/project2_submission.py features_9_50.txt features_9_50_test.txt > final_sub.csv")

