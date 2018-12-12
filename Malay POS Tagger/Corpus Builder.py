import nltk
import justext
from glob import glob

directory = 'html\\*-*.html'
html_files = glob(directory)

# Requests the article and prints out a string of the whole article.
for i in html_files:
    with open(i, 'r', errors='ignore') as webpage:
        page_content = webpage.read()
    paragraphs = justext.justext(page_content, justext.get_stoplist("Malay"))
    content = [p.text for p in paragraphs if not p.is_boilerplate]
    content_string = ' '.join(map(str, content))
    tok_text = nltk.word_tokenize(content_string)
    lower_tok_text = [i.lower() for i in tok_text]
    with open('corpus.txt', 'a') as f:
        print(f'Processing {i}...')
        print(content_string, file=f)

print('Corpus text extracted!')
with open('Corpus.txt', 'r') as f:
    corpus = f.read()
    tok_text = nltk.word_tokenize(corpus)
    fdist = nltk.FreqDist(tok_text)

print('Splicing corpus into single-word chunks...')
with open('Corpus Untagged Text.txt', 'a') as untagged:
    for i in tok_text:
        print(i, file=untagged)

print('Building corpus frequency...')
with open('corpus_frequency.txt', 'a') as cor:
    for word, frequency in fdist.most_common():
        print(word, file=cor)
print('Corpus frequency has been created!')
