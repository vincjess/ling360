import requests as r

url = 'https://www.lds.org/general-conference?lang=msa'
h = {'user-agent': 'Jesse Vincent (vincenjes@gmail.com)'}

url = 'https://www.lds.org/general-conference/2018/04/solemn-assembly?lang=msa'
h = {'user-agent': 'Jesse Vincent (vincenjes@gmail.com)'}

response = r.get(url, headers=h)
my_file = open('malay_gc_talk.html', 'a', encoding='utf-8')
my_file.write(str(response.text))
print('Done! Your html file should be saved in the directory.')
