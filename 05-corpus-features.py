from glob import glob
import re


def directory():
    folder = 'Mini-CORE/*.txt'
    return glob(folder)


# Cleans up the name to remove the containing folder.
def filename(i):
    name = re.search(r'Mini-CORE\\(.*.txt)', i).group(1)
    return name


# Returns the register of the file.
def register(i):
    register = re.sub(r'Mini-CORE\\1\+(\w\w).+', r'\1', i)
    return register


# Returns the type-to-token ratio of a single file.
def ttr(i):
    ttr = len(set(i)) / len(i)
    return ttr


# Returns the word count used for calculating ratios.
def word_count(i):
    word_count = re.findall(r'\b\w+\b', i, flags=re.IGNORECASE)
    return len(word_count)


# Returns the ratio of modals to tokens in a single text.
def modal_count(i):
    modal = re.findall(r'\bcan|could|will|would|shall|should|may|might|must\b',
                       i, flags=re.IGNORECASE)
    modal_ratio = len(modal) / word_count(i)
    if modal is not None:
        return modal_ratio


# Returns the ratio of contractions to tokens in a single text.
def contraction_count(i):
    contraction_count = re.findall(r'\'s|\'ll|\'d|\'nt\b', i)
    contraction_ratio = len(contraction_count) / word_count(i)
    if contraction_count is not None:
        return contraction_ratio
    if contraction_count is None:
        return '0'


# Creates the headers for the file results.txt.
with open('results.txt', 'a', encoding='utf-8') as results:
    results.write('{} {} {} {} {}'.format('filename'.ljust(12),
                                          'register'.rjust(45),
                                          'type-token-ratio'.rjust(15),
                                          'modal-ratio'.rjust(7),
                                          'contract_ratio'.rjust(14)) + '\n')
for i in directory():
    with open(i, 'r') as text_file:
        text_file = text_file.read()
        i = re.sub(r'<.+?>(.+)?', r'\1', i)
        with open('results.txt', 'a', encoding='utf-8') as results:
            results.write(filename(i).ljust(12) +
                          register(i).rjust(5) +
                          '{:15f}'.format(ttr(text_file)) +
                          '{:16f}'.format(modal_count(text_file)) +
                          '{:14f}'.format(contraction_count(text_file)) + '\n')
print('Your data has been gathered in results.txt. Thank you for playing!')
