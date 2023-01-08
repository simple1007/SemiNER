import os
import re
import json
import string
import tensorflow as tf
import numpy as np
import pickle
# from string_type_chk import string_chk

filelist = os.listdir()
tk_x = open('data_x.txt','w',encoding='utf-8')
tk_y = open('data_y.txt','w',encoding='utf-8')

error = []
max_len = 320
X = []
Y = []

tokenizer_x = tf.keras.preprocessing.text.Tokenizer(lower=False)
tokenizer_y = tf.keras.preprocessing.text.Tokenizer(lower=False)

count = 0

for file in filelist:
    if os.path.isdir(file):
        filelist2 = os.listdir(file)
        for file2 in filelist2:
            if file2.endswith('.json'):
                with open(os.path.join(file,file2),encoding='utf-8') as f:
                    # print(os.path.join(file,file2))
                    json_obj = json.load(f)
                    
                    json_obj = json_obj['document']

                    for jobj in json_obj:
                        jobj = jobj['sentence']
                        for jobj_ in jobj:
                            sentence = jobj_['form']
                            sentence_ = sentence
                            sentence__ = sentence
                            temp_s = sentence
                            # print(sentence)
                            if sentence.strip() == '':
                                continue
                            # if '__' in sentence:
                            #     # print(sentence)
                            #     error.append(sentence)
                            #     continue
                            entity = jobj_['NE']
                            word_ = jobj_['word']
                            # print(entity)
                            word_index = []
                    
                            temp_label = []#['O'] * length

                            temp_sentence = []
                            memory = {}
                            # temp_x = []
                            temp = []
                            temp_index = -1
                            ner_flag = False
                            temp_tag = ''
                            # count = 0
                            label = ['O'] * len(sentence)
                            start = 0
                            cc = 0
                            sentence = sentence.replace(' ','_')
                            s = list(sentence)
                            for char_index, en in enumerate(entity):
                                begin = en['begin']
                                end = en['end']

                                for i in range(begin,end):
                                    if s[i] == '_':
                                        label[i] = '_'
                                    elif i == begin:
                                        label[i] = 'B-'+en['label'][:2]
                                    else:
                                        label[i] = 'I-'+en['label'][:2]
                            
                            tk_x.write(' '.join(s)+'\n')
                            tk_y.write(' '.join(label)+'\n')

                            # X.append(s)
                            # Y.append(label)

                            # print(Y)
                            if True:     
                                with open('tokenizer_y.pkl','rb') as f:
                                    tokenizer_y = pickle.load(f)
                                with open('tokenizer_x.pkl','rb') as f:
                                    tokenizer_x = pickle.load(f)
                                X.append(s)
                                Y.append(label)           
                                if len(X) == 50:
                                    # tokenizer_x.fit_on_texts(X)
                                    # tokenizer_y.fit_on_texts(Y)

                                    X = tokenizer_x.texts_to_sequences(X)
                                    Y = tokenizer_y.texts_to_sequences(Y)
                                    
                                    X = tf.keras.utils.pad_sequences(
                                        X,
                                        maxlen=max_len,
                                        dtype='int32',
                                        padding='post',
                                        truncating='post',
                                        # value=len(tokenizer_y.word_index.keys())+1
                                    )

                                    Y = tf.keras.utils.pad_sequences(
                                        Y,
                                        maxlen=max_len,
                                        dtype='int32',
                                        padding='post',
                                        truncating='post',
                                        # value=len(tokenizer_y.word_index.keys())+1
                                    )

                                    np.save('data/%05d_x' % count,X)
                                    np.save('data/%05d_y' % count,Y)

                                    count += 1

                                    X = []
                                    Y = []
                            else:
                                X.append(s)
                                Y.append(label)

if True:
    tokenizer_y.fit_on_texts(Y)
    with open('tokenizer_y.pkl','wb') as f:
        pickle.dump(tokenizer_y,f)
    tokenizer_x.fit_on_texts(X)    
    with open('tokenizer_x.pkl','wb') as f:
        pickle.dump(tokenizer_x,f)
                           
tk_x.close()
tk_y.close()




