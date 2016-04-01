All the codes can be run in terminal, different argument is split by "space"
#For Fisher:
 python3 -W ignore Fisher.py /path-to-dataset number_of_fold

#For least square discriminant analysis:
 python3 SqClass.py /path-to-dataset number_of_fold

#For Logistic Regression:
 python3 Logistic.py /path-to-dataset split_proportion vectors_size_of_training
For example:
 python3 Logistic.py /home/CSci/spam.csv 0.8 5,10,25,50,100
which means 80% of the dataset is taken for training, we will use 5%,10%...of the training set for actually building the model

#For  Naive Bayes:
 python3 NaiveBayes.py /path-to-dataset split_proportion vectors_size_of_training
the usage is the same as logistic regression


