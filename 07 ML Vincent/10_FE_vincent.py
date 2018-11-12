import pandas
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


with open('mc_feat_names.txt') as name_file:
    names = name_file.read().strip().split('\t')
len_names = len(names)
with open('mc_features.csv') as mc_file:
    dataset = pandas.read_csv(mc_file, names=names,  # pandas DataFrame object
                              keep_default_na=False, na_values=['_'])  # avoid 'NA' category being interpreted as missing data  # noqa
print(list(dataset))  # easy way to get feature (column) names

array = dataset.values  # numpy array
feats = array[:,0:len_names - 1]  # to understand comma, see url in next line:
labels = array[:,-1]  
# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing

ETCmodel = ExtraTreesClassifier()
ETCmodel.fit(feats, labels)

LDAmodel = LinearDiscriminantAnalysis()

# train model based on all features
LDAmodel_allfeats = LDAmodel.fit(feats, labels)
print(LDAmodel_allfeats.coef_)


# create the RFE feature selection model and select 3 features
rfe = RFE(LDAmodel, 6)
rfe = rfe.fit(feats, labels)

print('summarize the selection of the features')
print(rfe.support_)  # did the feature make the cut?
print(rfe.ranking_)  # the feature's rank (all "passing" features share 1st)

print('comparing predictions of full model and RFE model...')
print('RFE: ', rfe.predict(feats))

final_model = SVC()
final_model.fit(feats, labels)
predictions = final_model.predict(feats)
print('Accuracy:', accuracy_score(labels, predictions))
