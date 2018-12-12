noun_prefixes = ('ke', 'pe')

pronoun_list = {'saya', 'aku', 'kamu', 'kau', 'awak', 'anda', 'baginda',
                'beliau', 'dia', 'ia', 'kita', 'kami', 'mereka', 'ini', 'itu'}

numbers = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'kosong', 'sifar',
           'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'lapan',
           'sembilan', 'sepuluh', 'puluh', 'belas', 'sebelas', 'ratus',
           'seratus', 'ribu', 'seribu', 'juta', 'sejuta')

modals = {'akan', 'telah', 'kena', 'pernah', 'boleh', 'sudah'}

punct = {'.', ':', '“', ',', '?', '!', '”', ';', '—', '(', ')', '-', '``',
         "$", '‘', '’', '...', "'", "'"}

copula = {'adalah', 'ialah', 'iaitu', 'ianya'}

negation_list = {'tidak', 'bukan'}

complementizer = {'bahawa'}

with open('MyTagger adj_tags.txt', 'r') as adjectives:
    adj_list = [i.lower() for i in adjectives.read().splitlines()]

with open('MyTagger adp_tags.txt', 'r') as adp:
    adp_list = [i.lower() for i in adp.read().splitlines()]

with open('MyTagger adv_tags.txt', 'r') as adv:
    adv_list = [i.lower() for i in adv.read().splitlines()]

with open('MyTagger conj_tags.txt', 'r') as conj:
    conj_list = [i.lower() for i in conj.read().splitlines()]

with open('MyTagger glosbe_tags.txt', 'r') as glosbe:
    glosbe_words = glosbe.read().splitlines()
    glosbe_tuples = []
    for i in glosbe_words:
        splits = tuple(i.split('\t'))
        glosbe_tuples.append(splits)
        glosbe_dict = dict((w.lower(), t) for w, t in glosbe_tuples)

with open('MyTagger Tagged Text.txt', 'r') as f:
    words = f.read().splitlines()
    tagged_tuples = []
    for i in words:
        splits = tuple(i.split('\t'))
        tagged_tuples.append(splits)


# Tags modals based on the modal list.
def modtag(word):
    if lw in modals:
        return 'MOD'


# Tags punctuation based on list.
def puncttag(word):
    if lw in punct:
        return 'PUNCT'
    else:
        return None


# Tags the complementizer.
def comptag(word):
    if lw == 'bahawa':
        return 'COMP'
    else:
        return None


# Tags the adpositions based on a list.
def adptag(word):
    if lw in adp_list:
        return 'ADP'


# Tags the adverbs based on a list.
def advtag(word):
    if lw in adv_list:
        return 'ADV'


# Tags the conjunctions based on a list of conjunctions.
def conjtag(word):
    if lw in conj_list:
        return 'CONJ'


# Tags copula based on list.
def coptag(word):
    if lw in copula:
        return 'COP'
    else:
        return None


# Tags the relativizer in Malay.
def reltag(word):
    if lw.startswith('yang'):
        return 'REL'
    else:
        return None


# Tags the negation words in Malay.
def negtag(word):
    if lw in negation_list:
        return 'NEG'
    else:
        return None


# Tags known adjectives based on adjective list.
def adjtag(word):
    if lw in adj_list:
        return 'ADJ'
    elif lw[3:] in adj_list:
        return 'ADJ_SUP'


# Tags pronouns and determiners based on position in sentence.
def dettag(word):
    if index > 0:
        if lw in pronoun_list:
            if previous_pos == 'NN':
                return 'DET'
            elif previous_pos == 'ADJ':
                return 'DET'
            else:
                return 'PRN'
        else:
            return None
    elif index == 0:
        if lw in pronoun_list:
            return 'PRN'
    else:
        return None


# Tags numbers of near-number words.
def numtag(word):
    if lw.startswith(numbers):
        return 'NUM'
    elif lw.endswith(numbers):
        return 'NUM'
    else:
        return None


# Tags any other word found in Glosbe results but not found in rules above.
def glosbetag(word):
    if lw in glosbe_dict:
        return glosbe_dict[lw]


# Tags verbs based on a set of rules.
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
        return None


# Tags nouns based on a set of rules.
def nountag(word):
    if word.startswith(noun_prefixes):
        if word.endswith('nya'):
            # If there is a - in the word, it is most likely plural.
            if '-' in word:
                return 'NN_PLUR_POS'
            else:
                return 'NN_POS'
        else:
            return 'NN'
    elif word.endswith('an'):
        return 'NN'
    if word.endswith(('annya', '-nya')):
        return 'NN_POS'
    else:
        return None


count = 0

with open('MyTagger Results.txt', 'a') as tagger:
    # Index created when POS is dependent on the previous word.
    for index, word in enumerate(tagged_tuples):
        lw = word[0].lower()
        actual_tag = word[1]
        previous_tuple = tagged_tuples[index - 1]
        previous_pos = previous_tuple[1]
        if modtag(lw) is not None:
            print(lw, modtag(lw), actual_tag, sep='\t', file=tagger)
        # Count is added if the rule-based tag and actual tag are the same.
            if modtag(lw) == actual_tag:
                count += 1
        elif puncttag(lw) is not None:
            print(lw, puncttag(lw), actual_tag, sep='\t', file=tagger)
            if puncttag(lw) == actual_tag:
                count += 1
        elif comptag(lw) is not None:
            print(lw, comptag(lw), actual_tag, sep='\t', file=tagger)
            if comptag(lw) == actual_tag:
                count += 1
        elif adptag(lw) is not None:
            print(lw, adptag(lw), actual_tag, sep='\t', file=tagger)
            if adptag(lw) == actual_tag:
                count += 1
        elif advtag(lw) is not None:
            print(lw, advtag(lw), actual_tag, sep='\t', file=tagger)
            if advtag(lw) == actual_tag:
                count += 1
        elif conjtag(lw) is not None:
            print(lw, conjtag(lw), actual_tag, sep='\t', file=tagger)
            if conjtag(lw) == actual_tag:
                count += 1
        elif coptag(lw) is not None:
            print(lw, coptag(lw), actual_tag, sep='\t', file=tagger)
            if coptag(lw) == actual_tag:
                count += 1
        elif reltag(lw) is not None:
            print(lw, reltag(lw), actual_tag, sep='\t', file=tagger)
            if reltag(lw) == actual_tag:
                count += 1
        elif negtag(lw) is not None:
            print(lw, negtag(lw), actual_tag, sep='\t', file=tagger)
            if negtag(lw) == actual_tag:
                count += 1
        elif adjtag(lw) is not None:
            print(lw, adjtag(lw), actual_tag, sep='\t', file=tagger)
            if adjtag(lw) == actual_tag:
                count += 1
        elif dettag(lw) is not None:
            print(lw, dettag(lw), actual_tag, sep='\t', file=tagger)
            if dettag(lw) == actual_tag:
                count += 1
        elif numtag(lw) is not None:
            print(lw, numtag(lw), actual_tag, sep='\t', file=tagger)
            if numtag(lw) == actual_tag:
                count += 1
        elif glosbetag(lw) is not None:
            print(lw, glosbetag(lw), actual_tag, sep='\t', file=tagger)
            if glosbetag(lw) == actual_tag:
                count += 1
        # glosbetag before noun, verb tag because they are more inaccurate.
        elif verbtag(lw) is not None:
            print(lw, verbtag(lw), actual_tag, sep='\t', file=tagger)
            if verbtag(lw) == actual_tag:
                count += 1
        elif nountag(lw) is not None:
            print(lw, nountag(lw), actual_tag, sep='\t', file=tagger)
            if nountag(lw) == actual_tag:
                count += 1
        else:
            print(lw, '...', actual_tag, sep='\t', file=tagger)

print('Tags have been successfully copied to the MyTagger Results.txt.')
print('Correct tags: ', count, '/', len(tagged_tuples), sep='')
print('Accuracy: ', round(count/len(tagged_tuples) * 100, 2), "%", sep='')
