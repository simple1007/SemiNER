from tabnanny import verbose
import tensorflow as tf
import numpy as np
import pickle
import fasttext

with open('tokenizer_x.pkl','rb') as f:
    tokenizer_x = pickle.load(f)
with open('tokenizer_y.pkl','rb') as f:
    tokenizer_y = pickle.load(f)
max_len = 320
file = open('tagging.txt','w',encoding='utf-8')

model = tf.keras.models.load_model('ner.model')

from tqdm import tqdm

with open('answer.txt',encoding='utf-8') as f:
    for _ in tqdm(range(78012)):
        l = f.readline()
        l = l.strip()
        l = l.replace(' ','_')
        l = list(l)

        x = tokenizer_x.texts_to_sequences([l])
        # print(x)
        # break
        X = x
        X = tf.keras.utils.pad_sequences(
            X,
            maxlen=max_len,
            dtype='int32',
            padding='post',
            truncating='post',
            # value=len(tokenizer_y.word_index.keys())+1
        )

        X = np.array(X)

        pred = model.predict(X,verbose=0)
        pred = tf.argmax(pred,axis=-1)
        pred = pred.numpy()
        y = tokenizer_y.sequences_to_texts(pred)

        # for xx, yy in zip(l,y[0].split(' ')):
        #     file.write(xx+'|'+yy+' ')
        # file.write('\n')
        # print(y,l)
        X_ = [' '.join(l)]
        pr_tag = [y[0]]
        # print(pr_tag)
        for p,xx in zip(pr_tag,X_):
            p = p.split(' ')
            xx = xx.split(' ')
            # result = []
            temp_tag = []
            prev = ''   
            count = 0
            temp = ''
            start = -1
            end = -1
            for index,(pp,xx_) in enumerate(zip(p,xx)):
                if pp.startswith('B-'):
                    prev = pp
                    start = index
                elif pp != 'O' and prev != '' and ((pp.startswith('I-') and prev.replace('B-','') == pp.replace('I-','')) or xx_ == '_'):
                    end = index
                    if xx_ == '_':
                        xx[index] = '!#!'
                else:
                    if pp == 'O' and prev != '' and start != -1 and end != -1:
                        xx[start] = '_'+p[start]+'!##!'+xx[start]
                        xx[end] = xx[end] + '!##!2'+'_'
                        # print(xx[start:end+1])
                        # print(p[start:end+1])
                
                    prev = ''
                    start = -1
                    end = -1
            file.write(''.join(xx)+'\n')
file.close()