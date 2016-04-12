import os
from csv import writer
import nltk
paths = '/media/chris/Elements/test'


def consolidate(path):
    s_path = path
    Files = os.listdir(s_path)
    byteFiles = [i for i in Files if '.asm' in i]
    consolidatedFile = s_path + 'operationcount.csv'
    operationlist = ['mov', 'push', 'add', 'xor', 'cmp', 'jz', 'jnz', 'test', 'or', 'lea', 'and', 'call', 'sub', 'shr', 'pop', 'retn', 'endp']

    with open(consolidatedFile, 'w') as f:
        fw = writer(f)
        colnames = ['Id']
        colnames += operationlist
        fw.writerow(colnames)
        for t, fname in enumerate(byteFiles):
            consolidation = []
            f = open(s_path + '/' + fname)
            raw = f.read().split()
            freq = nltk.FreqDist(raw)
            opercount = [0]*len(operationlist)
            for i in range(len(operationlist)):
                opercount[i] = freq[operationlist[i]]
            consolidation += [fname[:fname.find('.asm')]]
            consolidation += opercount
            fw.writerow(consolidation)

if __name__ == '__main__':
    consolidate(paths)
    print("DONE bytes count!")
