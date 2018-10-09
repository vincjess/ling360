import os
import requests
import re
from bs4 import BeautifulSoup
import time

# Creates the folder /scrape/ in the local directory.
if not os.path.exists('scrape'):
    os.mkdir('scrape')
    print("Directory", '/scrape/',  "created.")
else:
    print("Directory", '/scrape/',  "already exists.")


def get_rnlp(filename):
    url = 'https://reynoldsnlp.com/scrape/'
    headers = {'user-agent': 'Jesse Vincent (vincenjes@gmail.com)'}
    response = requests.get(url + filename, headers=headers)
    html_file = response.text
    with open('scrape/' + filename, 'w', encoding='utf-8') as html:
        html.write(html_file)
    time.sleep(2)


def get_hrefs(filename):
    with open('scrape/' + filename) as html:
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a', href=True)
        links = [links.get('href') for links in soup.find_all('a')]
        unused_filename = []
        for n in links:
            n = re.search('\w\w.html', n)
            if n:
                n = n.group()
            unused_filename.append(n)
        return unused_filename

# List based on 'aa.html' is created to start the loop based on aa.html.
get_rnlp('aa.html')
get_hrefs('aa.html')
unused_filename = get_hrefs('aa.html')
used_filename = []
# The loop continues until all URLs are checked.
while len(unused_filename) > 0:
    # File is popped after being checked if it is not in used_filename list.
    if unused_filename[0] not in used_filename:
        get_rnlp(unused_filename[0])
        unused_filename = unused_filename + get_hrefs(unused_filename[0])
        used_filename.append(unused_filename.pop(0))
# Lists are sorted to see the pattern printed while downloading URLs.
        unused_filename.sort()
        used_filename.sort()
# If the first URL is in the list
    else:
        print('A scallywag duplicate found! ' + unused_filename.pop(0))
    print('These need to be checked: ', unused_filename)
    print('These have been checked: ', used_filename)
