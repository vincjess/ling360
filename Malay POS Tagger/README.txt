All text files are supplemental files for the taggers, manually created. Everything else will be made through runnings these scripts.
One module (jusText) is not available by requesting it with pip. I got it from Dr. Davies. Instructions to install it are below. If you can't get Corpus Builder.py to work, the output files are also available in the directory.

1. Run Corpus Builder.py. This will extract from the /html/ folder and make corpus_frequency.txt, Corpus.txt, and Corpus Untagged Text.txt
2. Run Glosbe Scraper.py. This will extract from glosbe-keys.txt and give output glosbe-results.txt. (This is something I let run for 2 weeks straight, so probably just test it to make sure it works.)
3. Run ML Malay.py. This will use ML Malay Tagged Text.txt as the source for text, and it will create a boxplot for 20 Machine Learning tries.
4. Run MyTagger.py. This will use all of the MyTagger *.txt files and the MyTagger Tagged Text.txt to make a rule-based approach to the POS tagger.

-----

Download and install JusText from the following website:
http://corpus.tools/raw-attachment/wiki/Downloads/justext-1.2.tar.gz 

Place this file in a folder of your choice. For instruction purposes, we will put it in: C:\justext\

Extract the file by going to the Command Prompt/Terminal and going to that folder. In Windows, we get to that folder by typing 'cd C:\justext\'

Type in: tar xzvf justext-1.2.tar.gz 
 
Then 'cd' to the folder that it just created: C:\justext\justext-1.2

Then run the following: c:\justext\justext-1.2 python setup.py install

Then, find where Python is installed. If you don't know where this is, then check the Environment Variables (Windows assumed, this link might help for Mac users) (an easy search in Cortana should find it) and see where Python is installed in your Path. For these instructions, the folder is Python37, but it may be Python32, 27, etc.

Go to the folder C:\Python37\Lib\site-packages\justext\

In this folder, copy all files EXCEPT stoplists into a temp folder.

Download good JusText files from https://drive.google.com/open?id=0B1_UiH6KjdBIbDBYT3Z6OXptOUk
and unzip to C:\Python32\Lib\site-packages\justext\
