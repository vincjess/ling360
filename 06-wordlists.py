import re
from glob import glob
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize


# Creates a set to remove stopwords from wordlists.
stopword = set(stopwords.words('english'))
stopword.update(',', '.', ']', '[', '(', ')', '?', ':', ';', 'the', '^', '``',
                'in', 'one', 'i', 'a', 'this', '1', '2', '3', '4', '5', '!'
                '6', '7', '8', '9', '10', '-', '%', "''", "'", "...", 'The')


def directory():
    folder = 'Mini-CORE/*.txt'
    return glob(folder)


def register(i):
    register = re.sub(r'Mini-CORE\\1\+(\w\w).+', r'\1', i)
    return register


# Divides the directory to filter the text files for each register.
IN = [txt_file for txt_file in directory() if register(txt_file) == 'IN']
IP = [txt_file for txt_file in directory() if register(txt_file) == 'IP']
LY = [txt_file for txt_file in directory() if register(txt_file) == 'LY']
NA = [txt_file for txt_file in directory() if register(txt_file) == 'NA']
OP = [txt_file for txt_file in directory() if register(txt_file) == 'OP']
SP = [txt_file for txt_file in directory() if register(txt_file) == 'SP']

registers = [IN, IP, LY, NA, OP, SP]
register_text = ['IN', 'IP', 'LY', 'NA', 'OP', 'SP']
i = 0

for register in registers:
    print('\nProcessing', register_text[i], 'register...')
    # Joins all of the text files for a single register into one string.
    concat_list = ''.join([open(f).read() for f in register])
    concat_list = re.sub(r'<.+?>(.+)?', r'\1', concat_list)
    # Tokenizes each word in the string and removes any stopwords found.
    tokenized_list = word_tokenize(concat_list)
    print('All words: ', len(tokenized_list))
    cleaned = [word.lower() for word in tokenized_list if word not in stopword]
    print('Cleaned words: ', len(cleaned))
    # Creates a frequency list for the register and puts it in an output txt.
    word_list = FreqDist(cleaned).most_common()
    with open(register_text[i] + '-word-list.txt', 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in word_list))
    print(register_text[i], 'output text file has been created!')
    i += 1

print('\nAll register output text files have been created.\n'
      'Have a glamorous day!')
