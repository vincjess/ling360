import re
import requests as r
from bs4 import BeautifulSoup
import time
from random import randint

glosbe = 'https://glosbe.com/'
source_language = 'ms'
target_language = 'en'


def glosbe_search(keyword):
    return mystring


# Searches for the POS element tag on a search.
def pos(keyword):
    pos = re.compile(r'class=\"defmeta\">Type:</span><span>(.+?);\s?</span>')
    matchResult = pos.search(mystring)
    if matchResult is not None:
        return matchResult.group(1)
    else:
        return ''


# Searches for the best POS element tag if no accepted tag is found.
def alt_pos(keyword):
    cut_string = re.sub(r'\n?\t?', '', mystring)
    pos = re.compile(r'class=\"gender-n-phrase\">\n?{\n?(.+?)\n?}\n?</div>')
    matchResult = pos.search(cut_string)
    if matchResult is not None:
            return matchResult.group(1)
    else:
        return 'No POS found'


# This list is the corpus types from most common to least.
with open('corpus_frequency.txt', 'r') as key_words:
    key_words = key_words.read().splitlines()
    list_key_words = list(key_words)
    print(list_key_words)


count = 1


for i in list_key_words:
    for n in range(5000):
        try:
            url = glosbe + source_language + '/' + target_language + '/' + i
            h = {'user-agent': 'Jesse Vincent (vincenjes@gmail.com)'}
            response = r.get(url, headers=h)
            soup = BeautifulSoup(response.content, 'html.parser')
            mystring = str(soup)
            print(count, i, pos(i), alt_pos(i), sep='\t')
            with open('glosbe-results.txt', 'a') as pos_file:
                print(i, pos(i), alt_pos(i), file=pos_file, sep='\t')
            count += 1
        # The time has to be slowed down significantly.
            time.sleep(randint(10, 90))
            break
        except TimeoutError:
            pass
