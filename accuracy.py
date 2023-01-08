import re
f = open('result.txt',encoding='utf-8')
f1 = open('true.txt',encoding='utf-8')
max_len = 320
def getLabel(file):
    result = []
    for p in file:
        p = p.strip()
        # t = t.strip()
        sentence = []
        label = []
        p = p.replace('!##!2', '_')
        p = re.sub('_+','_',p)
        p = p.split('_')
        for pp in p:
            if '!##!' in pp:
                # pp = pp.replace('!##!', '!##! ')
                pp = pp.split('!##!')
                c = 0
                # print(1,pp)
                for index,ppp in enumerate(pp[1:]):
                    if '' == ppp:
                        continue
                    # if '!#!' ppp in 
                    
                    # if index == 0:
                    ppp = ppp.split('!#!')
                    # print(p)
                    # print(2,ppp)
                    for index2, pppp in enumerate(ppp):
                        # print(3,ppp,pppp)
                        if pppp != '':
                            if c == 0:
                                label.append(pp[0])
                                c+=1
                            else:
                                label.append(pp[0].replace('B-','I-'))
                                # sentence.append(pppp)
                    # else:
                    #     label.append('O')
            else:
                label.append('O')
                # sentence.append(pp)
        result.append(label+['O'] * (max_len - len(label)))
    return result

pred = getLabel(f)
true = getLabel(f1)

from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
print(classification_report(true,pred))