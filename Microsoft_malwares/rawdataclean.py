import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import OrderedDict
import itertools
import os
import xgboost
# # for 2-gram bytes file
# twogram_count = pd.read_csv("/media/chris/Elements/train_2gramfrequency.csv")
# max_column = twogram_count.max(axis=0)
# print len(max_column)
# including = [i for i, value in enumerate(max_column) if value > 500]
# twogram_count_final = twogram_count.iloc[:, including]
# twogram_count_final.to_csv("/media/chris/Elements/twogram_count_final.csv")
# print twogram_count_final.shape

# variable selection for 2gram_count based on random forest
# twogram_count_final = pd.read_csv("/media/chris/Elements/twogram_count_final.csv")
# print twogram_count_final.columns
# labels = pd.read_csv("/media/chris/Elements/trainLabels.csv")
# data = labels.merge(twogram_count_final, on='Id')
# data = data.drop('Unnamed: 0', axis=1)
# print data.shape
# Y = data.loc[:, 'Class']
# X = data.drop(['Id', 'Class'], axis=1)
# print X.shape
# clf = RandomForestClassifier(n_estimators=500, verbose=1)
# clf.fit(X, Y)

# importances = clf.feature_importances_

# d = {}
# for i, item in enumerate(X.columns):
#     d[item] = importances[i]

# importances_d = pd.DataFrame({"key": d.keys(), "value": d.values()})
# importances_d.to_csv("/media/chris/Elements/importances.csv",index=False)

# d = OrderedDict(sorted(d.items(), key=lambda x: x[1]))

# includings_importance = d.keys()[-1000:]

# data = data.loc[:, list(itertools.chain(['Id', 'Class'], includings_importance))]

# data.to_csv("/media/chris/Elements/2gram_count_importance.csv", index=False)




# for 2-gram operations
# twogram_operations = pd.read_csv("/media/chris/Elements/train_2gramoperation.csv")

# max_column2 = df.max(axis=0)
# including2 = [i > 50 for i in max_column]
# twogram_operations_final = two_gram[:, including2]

# twogram_final = twogram_operations_final.merge(twogram_count_final, on="Id")
# twogram_final.to_csv("/media/chris/Elements/train_final_2gram.csv")
# #

# # word_count
# train_frequency = pd.read_csv("/media/chris/Elements/train_frequency.csv")
# train_frequency2 = pd.read_csv("/media/chris/Elements/train_frequency2.csv")
# train_count_inter = train_frequency.merge(train_frequency2, on="Id")
# trainoperationcount = pd.read_csv("/media/chris/Elements/trainoperationcount.csv")
# train_count_final = trainoperationcount.merge(train_count_inter, on="Id")
# train_count_final.to_csv("/media/chris/Elements/train_count_final.csv")


# data cleaning for 2-gram operation:

# twogram_operations = pd.read_csv('/media/chris/Elements/train_2gramoperation.csv')

# max_column = twogram_operations.max(axis=0)
# print max_column
# print twogram_operations.shape
# including = [i for i, value in enumerate(max_column) if value > 0]

# twogram_operations_final = twogram_operations.iloc[:, including]
# print twogram_operations_final.shape

# twogram_operations_final.to_csv('/media/chris/Elements/twogram_operations_final.csv',index=False)

# data aggregation final

# Files = os.listdir('/media/chris/Elements/finaldata_Microsoft_train')
# #print Files
# path = os.path.join('media', 'chris', 'Elements', 'finaldata_Microsoft_train')
# #print path
# data = pd.read_csv('/'+ path + '/' + Files[0])

# for i in range(1, len(Files)):
#     print Files[i]
#     f = pd.read_csv('/' + path + '/' + Files[i])
#     print f.shape
#     data = data.merge(f, on='Id')
#     print data.shape

# print data.columns

# data.to_csv('/' + path + '/' + 'train_dataset.csv',index = False)

labels = np.load('/media/chris/Elements/finaldata_Microsoft_train/2gram_operations.npy')
