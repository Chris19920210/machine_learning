from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

X = np.load('/home/chris/Microsoft_malware/data_all.npy')
Y = np.load('/home/chris/Microsoft_malware/label_y.npy')

clf = RandomForestClassifier(n_estimators=500)

clf.fit(X, Y)

importances = clf.feature_importances_
importances = np.array(importances)
np.save('/home/chris/Microsoft_malware/importances', importances)

# return top 500 features
def sort_500(a, N):
    return np.argsort(a)[::-1][:N]

important_features = sort_500(importances, 500)
feature_list = pickle.load(open('/home/chris/Microsoft_malware/feature_list.obj', 'rb'))
feature_list = np.array(feature_list)
feature_lists_importance = feature_list[important_features]

X_train = X[:, important_features]


np.save('/home/chris/Microsoft_malware/X_train', X_train)
