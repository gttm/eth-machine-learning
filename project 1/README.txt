Machine Learning Project 1
Team name: Human Learning
Authors: Georgios Touloupas, Emmanouil Angelis

Summary of the approach
1) We perform perprocessing on the train set and test set using the project1_partitions_preprocess.py script. The 3d space of every image is partitioned into cubes, and for each cube we calculate the sum of the voxel values.
2) Using the sums of the partitions as features we train a ridge regression model in project1_partitions_submission.py. This model is used on the preprocessed test set to make the predictions.
3) We perform postprocessing on the predictions, limiting the ages between [18, 100].
