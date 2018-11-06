"""Jesse Vincent's Machine Learning Model."""

import glob
import re
from string import punctuation as punct  # string of common punctuation chars

import matplotlib.pyplot as plt
import nltk
import pandas
from pandas import scatter_matrix
from sklearn import model_selection
import warnings
# import model classes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

dr = 'C:\\Users\\vince\\PycharmProjects\\F18_DIGHT360\\assignments\\Mini-CORE\\'

# Makes Academic Word List into a set.
with open('awl.txt', 'r') as awl_list:
    awl = set(awl_list.read().splitlines())

# Makes hapax lists for each register into a set.
with open('SP-hapaxes.txt', 'r') as sp_hapax:
    sp_hapaxis = sp_hapax.read().splitlines()
sp_hapaxes = set(sp_hapaxis)

with open('OP-hapaxes.txt', 'r') as op_hapax:
    op_hapaxis = op_hapax.read().splitlines()
op_hapaxes = set(op_hapaxis)

with open('NA-hapaxes.txt', 'r') as na_hapax:
    na_hapaxis = na_hapax.read().splitlines()
na_hapaxes = set(na_hapaxis)

with open('LY-hapaxes.txt', 'r') as ly_hapax:
    ly_hapaxis = ly_hapax.read().splitlines()
ly_hapaxes = set(ly_hapaxis)

with open('IP-hapaxes.txt', 'r') as ip_hapax:
    ip_hapaxis = ip_hapax.read().splitlines()
ip_hapaxes = set(ip_hapaxis)

with open('IN-hapaxes.txt', 'r') as news_in_hapax:
    news_in_hapaxis = news_in_hapax.read().splitlines()
news_in_hapaxes = set(news_in_hapaxis)

with open('freq_SP.txt', 'r') as freq_SP:
    SP = freq_SP.read().splitlines()
sp = set(SP)

with open('freq_OP.txt', 'r') as freq_OP:
    OP = freq_OP.read().splitlines()
op = set(OP)

with open('freq_NA.txt', 'r') as freq_NA:
    NA = freq_NA.read().splitlines()
na = set(SP)

with open('freq_LY.txt', 'r') as freq_LY:
    LY = freq_LY.read().splitlines()
ly = set(LY)

with open('freq_IP.txt', 'r') as freq_IP:
    IP = freq_IP.read().splitlines()
ip = set(IP)

with open('freq_IN.txt', 'r') as freq_IN:
    IN = freq_IN.read().splitlines()
news_in = set(IN)


def subcorp(name):
    """Extract subcorpus from filename.

    name -- filename

    The subcorpus is the first abbreviation after `1+`.
    """
    return name.split('+')[1]


def ttr(in_Text):
    """Compute type-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    return len(set(in_Text)) / len(in_Text)


def pro1_tr(in_Text):
    """Compute 1st person pronoun-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    regex = r'(?:i|me|my|mine)$'
    pro1_count = len([True for i in in_Text if re.match(regex, i, re.I)])
    return pro1_count / len(in_Text)


def pro2_tr(in_Text):
    """Compute 2nd person pronoun-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    regex = r'(?:ye|you(?:rs?)?)$'
    pro2_count = len([True for i in in_Text if re.match(regex, i, re.I)])
    return pro2_count / len(in_Text)


def pro3_tr(in_Text):
    """Compute 3rd person pronoun-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    regex = r'(?:he|him|his|she|hers?|its?|they|them|theirs?)$'
    pro3_count = len([True for i in in_Text if re.match(regex, i, re.I)])
    return pro3_count / len(in_Text)


def punct_tr(in_Text):
    """Compute punctuation-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    punct_ct = len([True for i in in_Text if re.match('[' + punct + ']+$', i)])
    return punct_ct / len(in_Text)


def prop_of_N(tagged):
    """Compute the proportion of tagged words whose tags begin with 'N'."""
    prp = len([True for tok, tag in tagged if tag.startswith('N')])
    return prp / len(tagged)


def prop_of_modal(tagged):
    """Compute the proportion of tagged words whose tags begin with 'MD'."""
    return len([True for tok, tag in tagged if tag.startswith('MD')])


def prop_of_propern(tagged):
    """Compute the proportion of tagged words whose words begin with 'NNP'."""
    return len([True for tok, tag in tagged if tag.startswith('NNP')])


def prop_of_prep(tagged):
    """Compute the proportion of tagged words whose words begin with 'IN'."""
    return len([True for tok, tag in tagged if tag.startswith('IN')])


def prop_of_cc(tagged):
    """Compute the proportion of tagged words whose words begin with 'CC'."""
    return len([True for tok, tag in tagged if tag.startswith('CC')])


def prop_of_exist(tagged):
    """Compute the proportion of tagged words whose words begin with 'EX'."""
    return len([True for tok, tag in tagged if tag.startswith('EX')])


def prop_of_adj(tagged):
    """Compute the proportion of tagged words whose words begin with 'JJ'."""
    return len([True for tok, tag in tagged if tag.startswith('JJ')])


def prop_of_adjer(tagged):
    """Compute the proportion of tagged words whose words begin with 'JJR'."""
    return len([True for tok, tag in tagged if tag.startswith('JJR')])


def prop_of_adjest(tagged):
    """Compute the proportion of tagged words whose words begin with 'JJS'."""
    return len([True for tok, tag in tagged if tag.startswith('JJS')])


def prop_of_pos(tagged):
    """Compute the proportion of tagged words whose words begin with 'POS'."""
    return len([True for tok, tag in tagged if tag.startswith('POS')])


def prop_of_advb(tagged):
    """Compute the proportion of tagged words whose words begin with 'RB'."""
    return len([True for tok, tag in tagged if tag.startswith('RB')])


def prop_of_advber(tagged):
    """Compute the proportion of tagged words whose words begin with 'POS'."""
    return len([True for tok, tag in tagged if tag.startswith('RBR')])


def prop_of_advbest(tagged):
    """Compute the proportion of tagged words whose words begin with 'POS'."""
    return len([True for tok, tag in tagged if tag.startswith('RBS')])


def prop_of_awl(in_Text):
        return len([True for i in in_Text if i in awl]) / len(in_Text)


def freq_sp_words(in_Text):
    """Computes the frequency of highly frequent words in SP register."""
    return len([True for i in in_Text if i in sp]) / len(in_Text)


def freq_op_words(in_Text):
    """Computes the frequency of highly frequent words in OP register."""
    return len([True for i in in_Text if i in op]) / len(in_Text)


def freq_na_words(in_Text):
    """Computes the frequency of highly frequent words in NA register."""
    return len([True for i in in_Text if i in na]) / len(in_Text)


def freq_ly_words(in_Text):
    """Computes the frequency of highly frequent words in LY register."""
    return len([True for i in in_Text if i in ly]) / len(in_Text)


def freq_ip_words(in_Text):
    """Computes the frequency of highly frequent words in SP register."""
    return len([True for i in in_Text if i in ip]) / len(in_Text)


def freq_in_words(in_Text):
    """Computes the frequency of highly frequent words in IN register."""
    return len([True for i in in_Text if i in news_in]) / len(in_Text)


def freq_sp_hapaxes(in_Text):
    """Computes the count of words that are hapaxes in SP register."""
    return len([True for i in in_Text if i in sp_hapaxes]) / len(in_Text)


def freq_op_hapaxes(in_Text):
    """Computes the count of words that are hapaxes in OP register."""
    return len([True for i in in_Text if i in op_hapaxes]) / len(in_Text)


def freq_na_hapaxes(in_Text):
    """Computes the count of words that are hapaxes in NA register."""
    return len([True for i in in_Text if i in na_hapaxes]) / len(in_Text)


def freq_ly_hapaxes(in_Text):
    """Computes the count of words that are hapaxes in LY register."""
    return len([True for i in in_Text if i in ly_hapaxes]) / len(in_Text)


def freq_ip_hapaxes(in_Text):
    """Computes the count of words that are hapaxes in IP register."""
    return len([True for i in in_Text if i in ip_hapaxes]) / len(in_Text)


def freq_in_hapaxes(in_Text):
    """Computes the count of words that are hapaxes in IN register."""
    return len([True for i in in_Text if i in news_in_hapaxes]) / len(in_Text)


feat_names = ['ttr', '1st-pro', '2nd-pro', '3rd-pro', 'punct', 'prop_of_N',
              'prop_of_modal', 'prop_of_propern', 'prop_of_prep', 'prop_of_cc',
              'prop_of_exist', 'prop_of_adj', 'prop_of_adjer', 'prop_of_adjest',
              'prop_of_pos', 'prop_of_advb', 'prop_of_advber',
              'prop_of_advbest', 'prop_of_awl', 'freq_sp_words',
              'freq_op_words', 'freq_na_words', 'freq_ly_words',
              'freq_ip_words', 'freq_in_words', 'sp_hapaxes', 'op_hapaxes',
              'na_hapaxes', 'ly_hapaxes', 'ip_hapaxes', 'in_hapaxes', 'genre']
with open('mc_feat_names.txt', 'w') as name_file:
    name_file.write('\t'.join(feat_names))

with open('mc_features.csv', 'w') as out_file:
    for f in glob.glob(dr + '*.txt'):
        print('.', end='', flush=True)  # show progress; print 1 dot per file
        with open(f) as the_file:
            read_text = the_file.read()
            cleaned_text = re.sub(r'<[hp]>', '', read_text)
            tok_text = nltk.word_tokenize(cleaned_text)
            lower_txt = [word.lower() for word in tok_text]
        tagged = nltk.pos_tag(tok_text)
        print(ttr(tok_text), pro1_tr(tok_text), pro2_tr(tok_text),
              pro3_tr(tok_text), punct_tr(tok_text), prop_of_N(tagged),
              prop_of_modal(tagged), prop_of_propern(tagged),
              prop_of_prep(tagged), prop_of_cc(tagged), prop_of_exist(tagged),
              prop_of_adj(tagged), prop_of_adjer(tagged),
              prop_of_adjest(tagged), prop_of_pos(tagged), prop_of_advb(tagged),
              prop_of_advber(tagged), prop_of_advbest(tagged),
              prop_of_awl(tok_text), freq_sp_words(tok_text),
              freq_op_words(tok_text), freq_na_words(tok_text),
              freq_ly_words(tok_text), freq_ip_words(tok_text),
              freq_in_words(tok_text), freq_sp_hapaxes(lower_txt),
              freq_op_hapaxes(lower_txt), freq_na_hapaxes(lower_txt),
              freq_ly_hapaxes(lower_txt), freq_ip_hapaxes(lower_txt),
              freq_in_hapaxes(lower_txt), subcorp(f), sep=',', file=out_file)
    print()  # newline after progress dots

# ##############################################################################
# Do not change anything below this line! The assignment is simply to try to
# design useful features for the task by writing functions to extract those
# features. Simply write new functions and add a label to feat_names and call
# the function in the `print` function above that writes to out_file. MAKE SURE
# TO KEEP the order the same between feat_names and the print function, ALWAYS
# KEEPING `'genre'` AND `subcorp(f)` AS THE LAST ITEM!!
#
# ##############################################################################
# Load dataset
with open('mc_feat_names.txt') as name_file:
    names = name_file.read().strip().split('\t')
len_names = len(names)
with open('mc_features.csv') as mc_file:
    dataset = pandas.read_csv(mc_file, names=names,  # pandas DataFrame object
                              keep_default_na=False, na_values=['_'])  # avoid 'NA' category being interpreted as missing data  # noqa
print(type(dataset))

# Summarize the data
print('"Shape" of dataset:', dataset.shape,
      '({} instances of {} attributes)'.format(*dataset.shape))
print()
print('"head" of data:\n', dataset.head(20))  # head() is a method of DataFrame
print()
print('Description of data:\n:', dataset.describe())
print()
print('Class distribution:\n', dataset.groupby('genre').size())
print()

# Visualize the data
print('Drawing boxplot...')
grid_size = 0
while grid_size ** 2 < len_names:
    grid_size += 1
dataset.plot(kind='box', subplots=True, layout=(grid_size, grid_size),
             sharex=False, sharey=False)
fig = plt.gcf()  # get current figure
fig.savefig('boxplots.png')

# histograms
print('Drawing histograms...')
dataset.hist()
fig = plt.gcf()
fig.savefig('histograms.png')

# scatter plot matrix
print('Drawing scatterplot matrix...')
scatter_matrix(dataset)
fig = plt.gcf()
fig.savefig('scatter_matrix.png')
print()

print('Splitting training/development set and validation set...')
# Split-out validation dataset
array = dataset.values  # numpy array
feats = array[:,0:len_names - 1]  # to understand comma, see url in next line:
labels = array[:,-1]  # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
print('\tfull original data ([:5]) and their respective labels:')
print(feats[:5], labels[:5], sep='\n\n', end='\n\n\n')
validation_size = 0.20
seed = 7  # used to make 'random' choices the same in each run
split = model_selection.train_test_split(feats, labels,
                                         test_size=validation_size,
                                         random_state=seed)
feats_train, feats_validation, labels_train, labels_validation = split
# print('\ttraining data:\n', feats_train[:5],
#       '\ttraining labels:\n', labels_train[:5],
#       '\tvalidation data:\n', feats_validation[:5],
#       '\tvalidation labels:\n', labels_validation[:5], sep='\n\n')

# Test options and evaluation metric
print()

print('Initializing models...')
# Spot Check Algorithms
models = [('LR', LogisticRegression()),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('SVM', SVC())]
print('Training and testing each model using 10-fold cross-validation...')
# evaluate each model in turn
results = []
names = []
for name, model in models:
    # https://chrisjmccormick.files.wordpress.com/2013/07/10_fold_cv.png
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, feats_train,
                                                 labels_train, cv=kfold,
                                                 scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = '{}: {} ({})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)
print()

print('Drawing algorithm comparison boxplots...')
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
fig = plt.gcf()
fig.savefig('compare_algorithms.png')
print()

# Make predictions on validation dataset
# best_model = KNeighborsClassifier()
# best_model.fit(feats_train, labels_train)
# predictions = best_model.predict(feats_validation)
# print('Accuracy:', accuracy_score(labels_validation, predictions))
# print()
# print('Confusion matrix:')
# cm_labels = 'Iris-setosa Iris-versicolor Iris-virginica'.split()
# print('labels:', cm_labels)
# print(confusion_matrix(labels_validation, predictions, labels=cm_labels))
# print()
# print('Classification report:')
# print(classification_report(labels_validation, predictions))
