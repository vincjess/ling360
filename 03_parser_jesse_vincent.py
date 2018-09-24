import re
from bs4 import BeautifulSoup

my_file = open('malay_gc_talk.html', 'r', encoding='utf8')
html = my_file.read()
soup = BeautifulSoup(html, 'html.parser')
html = soup.find_all('p')
html = re.sub(r'<.+?>', '', str(html))
html = re.sub(r'\[|\]', '', str(html))
my_text_file = open('malay_gc_talk.txt', 'w+')
for a in html:
    my_text_file.write(str(a))
print('Done! Your text file should be saved as malay_gc_talk.txt.')
