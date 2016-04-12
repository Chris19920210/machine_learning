import os
from csv import writer
import nltk
import itertools
import re
paths = '/media/chris/Elements/test'


def consolidate(path):
    s_path = path
    Files = os.listdir(s_path)
    byteFiles = [i for i in Files if '.asm' in i]
    consolidatedFile = s_path + '_2gramoperation.csv'
    operationlist = ['mov', 'push', 'add', 'xor', 'cmp', 'jz', 'jnz', 'test', 'or', 'lea', 'and', 'call', 'sub', 'shr', 'pop', 'retn', 'endp']
    arglist = ['eax', 'ebx', 'ecx', 'edx', 'short', 'edi', 'esi', 'ebp']
    two_gram = [i for i in itertools.product(operationlist, arglist)]

    with open(consolidatedFile, 'w') as f:
        fw = writer(f)
        colnames = ['Id']
        colnames += ['2_' + str(i) for i in two_gram]
        fw.writerow(colnames)
        for t, fname in enumerate(byteFiles):
            consolidation = []
            f = open(s_path + '/' + fname, 'rU')
            chars_to_remove = ['!', '[', ']', ',']
            a = f.read().translate(None, ''.join(chars_to_remove))
            raw = re.split(r'\s+|\+|\t+', a)
            raw2 = nltk.bigrams(raw)
            freq = nltk.FreqDist(raw2)
            hexByte = [0]*len(two_gram)
            for i in range(len(two_gram)):
                hexByte[i] = freq[two_gram[i]]
            consolidation += [fname[:fname.find('.asm')]]
            consolidation += hexByte
            fw.writerow(consolidation)

if __name__ == '__main__':
    consolidate(paths)
    print("DONE 2-gram-bytes count!")
