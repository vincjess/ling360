from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from nltk import word_tokenize, pos_tag
import matplotlib.pyplot as plt

with open('ML Malay Tagged Text.txt', 'r') as f:
    words = []
    sentence_breaks = [0]
    # Looks for a new line to break sentences into sublists.
    for i, l in enumerate(f):
        if l in ['\n', '\r\n'] or not l.strip():
            sentence_breaks.append(i - len(sentence_breaks))
        else:
            l = l.strip()
            words.append(l)
    # Divides data into two lists: tagged sentences and tagged tuples.
    tsentences = []
    ttuples = []
    for i in words:
        splits = tuple(i.split('\t'))
        ttuples.append(splits)
    for i, sentence_break in enumerate(sentence_breaks):
        if i == 0:
            continue
        tsentences.append(ttuples[sentence_breaks[i - 1]:sentence_break])

noun_prefixes = ['ke', 'pe']
pronoun_list = {'saya', 'aku', 'kamu', 'kau', 'awak', 'anda', 'baginda',
                'beliau', 'dia', 'ia', 'kita', 'kami', 'mereka'}
numbers = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'kosong', 'sifar',
           'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'lapan',
           'sembilan', 'sepuluh', 'puluh', 'belas', 'sebelas', 'ratus',
           'seratus', 'ribu', 'seribu', 'juta', 'sejuta'}


# Uses words from the tagged list to tag words.
def maintag(word):
    word_list = []
    for word, tag in ttuples:
        word_list.append(word.lower())
        if word.lower() in word_list:
            return tag


def verbtag(word):
    # Tags agent focus verbs.
    if word.startswith('me'):
        if word.endswith('kan'):
            return 'VB_AF_CAUS'
        elif word.endswith('kannya'):
            return 'VB_AF_CAUS'
        elif word.endswith('i'):
            return 'VB_AF_LOC'
        else:
            return 'VB_AF'
    # Tags patient focus verbs.
    if word.startswith('di'):
        if word.endswith('kan'):
            return 'VB_PF_CAUS'
        elif word.endswith('i'):
            return 'VB_PF_LOC'
        else:
            return 'VB_PF'
    # Tags verbs that don't have prefix but have suffix.
    if word.endswith('i'):
        return 'VB_LOC'
    elif word.endswith('kan'):
        return 'VB_CAUS'
    # Tags middle voice verbs.
    if word.startswith('ter'):
        return 'VB_MID'
    # Tags stative verbs.
    if word.startswith('ber'):
        if word.endswith('kan'):
            return 'VB_STAT_CAUS'
        elif word.endswith('i'):
            return 'VB_STAT_LOC'
        else:
            return 'VB_STAT'
    else:
        return ''


# Tags nouns based on a set of rules.
def nountag(word):
    # Tags known nouns from verb list.
    for prefix in noun_prefixes:
        if word.startswith(prefix):
            if word.endswith('nya'):
                if '-' in word:
                    return 'NN_PLUR_POS'
                else:
                    return 'NN_POS'
            else:
                return 'NN'
        elif '-' in word:
            return 'NN_PLUR'
    if word.endswith('annya'):
        return 'NN_POS'
    else:
        return ''


# Tags numbers of near-number words.
def numtag(word):
    for number in numbers:
        if word.startswith(number):
            return 'NUM'
        elif word.endswith(number):
            return 'NUM'
        else:
            return ''

print("Tagged sentences: ", len(tsentences))
print("Tagged words:", len(ttuples))


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    feature_dict = {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'ke': sentence[index][:2],
        'mem': sentence[index][:3],
        'i': sentence[index][-1],
        'an': sentence[index][-2:],
        'kan': sentence[index][-3:],
        'menge': sentence[index][:4],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'prev_prev_word': '' if index == 0 else sentence[index - 2],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:],
        'verb': verbtag(sentence[index]),
        'noun': nountag(sentence[index]),
        'numeral': numtag(sentence[index]),
        'pronoun': sentence[index] in pronoun_list,
        'other-tags': maintag(sentence[index])
    }
    return feature_dict


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


# Splits the dataset for training and testing.
cutoff = int(.75 * len(tsentences))
training_sentences = tsentences[:cutoff]
test_sentences = tsentences[cutoff:]

print('\nTraining Sentences: ', len(training_sentences))
print('Test Sentences: ', len(test_sentences), '\n')


def transform_to_dataset(tsentences):
    X, y = [], []

    for tagged in tsentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y

box_plot_data = []

for i in range(1, 20):
    X, y = transform_to_dataset(training_sentences)

    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', DecisionTreeClassifier(criterion='entropy'))
    ])

    clf.fit(X[:10000], y[
                       :10000])

    print(f'Training #{i} completed')

    X_test, y_test = transform_to_dataset(test_sentences)
    accuracy = round(clf.score(X_test, y_test) * 100, 2)
    print(f'Accuracy: {accuracy} %')
    box_plot_data.append(accuracy)

print('All accuracy results: ', box_plot_data)
print('Plotting data...')
plt.boxplot(box_plot_data)
plt.ylabel('Accuracy')
plt.xlabel('Machine Learning Malay POS Tagger')
fig = plt.gcf()
fig.savefig('Machine Learning Accuracy Chart.png')
print('Done!')


# Predicts the tags of any unattested sentence.
def pos_tag(sentence):
    tags = clf.predict(
        [features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)


print('\nTest on any sentence below:\n')
print(list(pos_tag(word_tokenize('Saya percaya dalam Yesus Kristus'))))
