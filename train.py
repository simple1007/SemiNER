import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Reshape,GlobalMaxPooling1D, Conv1D, Conv2D, LSTM, Input, Bidirectional, Embedding, Dense, GlobalAvgPool1D, Concatenate, TimeDistributed
from tensorflow.keras import Model
import pickle
import fasttext
with open('tokenizer_x.pkl','rb') as f:
    tokenizer_x = pickle.load(f)
with open('tokenizer_y.pkl','rb') as f:
    tokenizer_y = pickle.load(f)
# X = np.load('X.npy')
# Y = np.load('Y.npy')

# import sys
# sys,exit()
def make_emb(model,l:str,index=0):
    l = l.strip()
    morphs = l.split(' ')
    emb = [model.get_word_vector(tk) for tk in morphs]
    return emb

def make_seq(model,line:list,index=0):
    result = []
    seq_len = max_len

    for l in line:
        l = l.strip()
        # if l == '.' or l == '?' or l == ',' or l == '!':
        #     continue
        seq_temp = make_emb(model,l,index)

        seq_temp = seq_temp + [
            np.zeros((300))
        ] * (seq_len - len(seq_temp))
        result.append(seq_temp)
    return result

# max_len = 300
# max_morph_len = 120
# model_wv = fasttext.load_model('cc.ko.300.bin')
EPOCH = 10
BATCH = 50
max_len = 320
w_len = 25

# print(np.array(x).shape)
# print(X.shape)
# import sys
# sys.exit()
# X = np.array(x)
# train_ = int(X.shape[0] * 0.9)
# # val_ = int(X.shape[0] * 0.2)
# X_temp = X
# Y_temp = Y
# X = X_temp[:train_]
# X_test = X_temp[train_:]
# Y = Y_temp[:train_]
# Y_test = Y_temp[train_:]
# print(X_test.shape,X.shape)
# print(Y_test.shape,Y.shape)

import numpy as np
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report


class F1Metrics(Callback):

    def __init__(self, id2label, pad_value=0, validation_data=None):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
        """
        super(F1Metrics, self).__init__()
        self.id2label = id2label
        self.pad_value = pad_value
        self.validation_data = validation_data
        self.is_fit = validation_data is None

    def find_pad_index(self, array):
        """Find padding index.
        Args:
            array (list): integer list.
        Returns:
            idx: padding index.
        Examples:
             >>> array = [1, 2, 0]
             >>> self.find_pad_index(array)
             2
        """
        try:
            return list(array).index(self.pad_value)
        except ValueError:
            return len(array)

    def get_length(self, y):
        """Get true length of y.
        Args:
            y (list): padded list.
        Returns:
            lens: true length of y.
        Examples:
            >>> y = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
            >>> self.get_length(y)
            [1, 2, 3]
        """
        lens = [self.find_pad_index(row) for row in y]
        return lens

    def convert_idx_to_name(self, y, lens):
        """Convert label index to name.
        Args:
            y (list): label index list.
            lens (list): true length of y.
        Returns:
            y: label name list.
        Examples:
            >>> # assumes that id2label = {1: 'B-LOC', 2: 'I-LOC'}
            >>> y = [[1, 0, 0], [1, 2, 0], [1, 1, 1]]
            >>> lens = [1, 2, 3]
            >>> self.convert_idx_to_name(y, lens)
            [['B-LOC'], ['B-LOC', 'I-LOC'], ['B-LOC', 'B-LOC', 'B-LOC']]
        """
        y = [[self.id2label[idx] for idx in row[:l]]
             for row, l in zip(y, lens)]
        return y

    def predict(self, X, y):
        """Predict sequences.
        Args:
            X (list): input data.
            y (list): tags.
        Returns:
            y_true: true sequences.
            y_pred: predicted sequences.
        """
        y_pred = self.model.predict_on_batch(X)

        # reduce dimension.
        y_true = np.argmax(y, -1)
        y_pred = np.argmax(y_pred, -1)

        lens = self.get_length(y_true)

        y_true = self.convert_idx_to_name(y_true, lens)
        y_pred = self.convert_idx_to_name(y_pred, lens)

        return y_true, y_pred

    def score(self, y_true, y_pred):
        """Calculate f1 score.
        Args:
            y_true (list): true sequences.
            y_pred (list): predicted sequences.
        Returns:
            score: f1 score.
        """
        score = f1_score(y_true, y_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(y_true, y_pred, digits=4))
        return score

    def on_epoch_end(self, epoch, logs={}):
        if self.is_fit:
            self.on_epoch_end_fit(epoch, logs)
        else:
            self.on_epoch_end_fit_generator(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        X = self.validation_data[0]
        y = self.validation_data[1]
        y_true, y_pred = self.predict(X, y)
        score = self.score(y_true, y_pred)
        logs['f1'] = score

    def on_epoch_end_fit_generator(self, epoch, logs={}):
        y_true = []
        y_pred = []
        for X, y in self.validation_data:
            y_true_batch, y_pred_batch = self.predict(X, y)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
        score = self.score(y_true, y_pred)
        logs['f1'] = score

id2label = {1: 'B-LOC', 2: 'I-LOC'}
id2label = {v:k for k,v in tokenizer_y.word_index.items()}
f1score = F1Metrics(id2label)

def dataset():
    for i in range(EPOCH):
        count = 0
        with open('data_x.txt',encoding='utf-8') as f:
            with open('data_y.txt',encoding='utf-8') as ff:
                x = []
                y = []
                for l,yd in zip(f,ff):
                    l = l.strip()
                    x.append(l)

                    yd = yd.strip()
                    y.append(yd.split(' '))
                    if len(x) == BATCH:
                        x_emb = make_seq(model_wv,x)
                        y_temp = y
                        y = []
                        x = []
                        count += 1
                        y_temp = tokenizer_y.sequences_to_texts(y_temp)
                        y_temp = tf.keras.utils.pad_sequences(
                            y_temp,
                            maxlen=max_len,
                            dtype='int32',
                            padding='post',
                            truncating='post',
                            # value=len(tokenizer_y.word_index.keys())+1
                        )
                        yield np.array(x_emb),np.array(y_temp)
                    if int(X.shape[0]//BATCH * 0.9) <= count:
                        break
steps_per_epoch = 9318 * 0.9 * 0.9#int((X.shape[0]//BATCH * 0.9) * 0.9)
steps_per_val = 9318 * 0.9 * 0.1#(X.shape[0]//BATCH*0.9) - steps_per_epoch#int((X.shape[0]//BATCH *0.9) * 0.1)
# print(X_test.shape)
# import sys
# sys.exit()


# d = { v:k for k,v in tokenizer_y.word_index.items()}
# print("FFFF",d[43],"GGGGG")
# import sys
# sys.exit()

if False:
    def dataset2():
        for _ in range(EPOCH):
            for i in range(int(9318 * 0.9)):
                X = np.load('data/%05d_x.npy'%i)
                Y = np.load('data/%05d_y.npy'%i)

                yield X, Y
    # in_word = Input(shape=(max_len,300))
    in_word = Input(shape=(max_len))
    emb = Embedding(len(tokenizer_x.word_index.keys())+1,80)(in_word)
    # emb = tf.expand_dims(emb,1)
    # emb = tf.expand_dims(emb,-1)
    # # print(emb.shape)
    # conv = Conv2D(30,(3,100),activation='relu',input_shape=((w_len,100,1)))(emb)
    # conv = Reshape((conv.shape[1],conv.shape[2],conv.shape[4]))(conv)
    # conv = tf.reshape(conv,(conv.shape[1],conv.shape[2]*conv.shape[3]))
    # print(conv.shape)
    # print(conv.shape)
    # import sys
    # sys.exit()
    # conv = TimeDistributed(GlobalMaxPooling1D())(conv)
    # print(conv.shape)
    # import sys
    # sys.exit()
    # in_word2 = Input(shape=(max_morph_len,))
    # emb2 = emb_layer2(in_word2)
    # bilstm = tf.keras.layers.TimeDistributed(Bidirectional(LSTM(60)))(emb)
    bilstm = Bidirectional(LSTM(100,return_sequences=True))(emb)
    # print(bilstm.shape)
    # import sys
    # sys.exit()
    # avg = Average()([bilstm])
    # time = TimeDistributed(Dense(20))(emb)
    # avg = GlobalAvgPool1D()(bilstm)
    # avg = tf.expand_dims(avg,1)
    # print(avg.shape)
    # print(bilstm.shape)
    # import sys
    # sys.exit()
    # concat = Concatenate(axis=1)([avg,bilstm])
    # bilstm = Bidirectional(LSTM(20))(bilstm)
    # bilstm2 = Bidirectional(LSTM(250))(emb2)
    # bilstm = Concatenate()([bilstm,bilstm2])
    # bilstm = Dense(100)(bilstm)
    # concat = Concatenate()([time,bilstm])
    # bilstm = TimeDistributed(Dense(100,activation='relu'))(bilstm)

    # print(in_word.shape())
    # import sys
    # sys.exit()

    # print(in_word.shape())
    # import sys
    # sys.exit()
    # bilstm = Dense(3000,activation='relu')(bilstm)
    # output = Bidirectional(LSTM(len(tokenizer_y.word_index.keys())+1,return_sequences=True),merge_mode='sum')(bilstm)
    output = TimeDistributed(Dense(len(tokenizer_y.word_index.keys())+1,activation='softmax'))(bilstm)
    # output = bilstm
    model = Model(inputs=in_word,outputs=output)

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    # print(len(set(y_dict.keys())))
    #train_data = dataset()
    #val_data = validation()
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2),
                tf.keras.callbacks.ModelCheckpoint(filepath='ner.model',monitor='val_loss',save_best_only=True)]
    ds = dataset2()
    # while True:
    #     d = next(ds)
    #     print(d[0].shape,d[1].shape)
    # print(next(ds)[1].shape)
    # import sys
    # sys.exit()
    model.fit(ds,epochs=EPOCH,batch_size=BATCH,steps_per_epoch=steps_per_epoch,validation_data=ds,validation_steps=steps_per_val,callbacks=callback)

    model.save('ner.model_last')
# import sys
# sys.exit()
# import sys
# sys.exit()
model = tf.keras.models.load_model('ner.model')
# print("ff",X_test.shape)
# Y_corr = Y_test\
res = []
corr = []
pred_temp = []
from tqdm import tqdm
def testset():
    for i in range(1):
        count = 0
        with open('data_x.txt',encoding='utf-8') as f:
            with open('data_y.txt',encoding='utf-8') as ff:
                x = []
                y = []
                for l,yd in zip(f,ff):
                    l = l.strip()
                    x.append(l)

                    yd = yd.strip()
                    y.append(yd.split(' '))
                    if len(x) == BATCH:
                        x_emb = make_seq(model_wv,x)
                        y_temp = y
                        y = []
                        x = []
                        count += 1
                        y_temp = tokenizer_y.sequences_to_texts(y_temp)
                        y_temp = tf.keras.utils.pad_sequences(
                            y_temp,
                            maxlen=max_len,
                            dtype='int32',
                            padding='post',
                            truncating='post',
                            # value=len(tokenizer_y.word_index.keys())+1
                        )
                        yield np.array(x_emb),np.array(y_temp)
                    # if int(X.shape[0]//BATCH * 0.9) <= count:
                    #     continue

# for ii in tqdm(range(X_test.shape[0])):#enumerate(X_test):
    # xt = X_test[ii]
    # pred_temp.append(xt)
# ts = testset()
# print("garbage start")
# for _ in tqdm(range(int(X.shape[0]//BATCH * 0.9))):
    # next(ts)
    # print('1')
print("test start")
Y_pr = []
corr = []
X = []
c = 0
for i in tqdm(range(int(9318*0.9),9318)):
    # pred_temp,Y_test = next(ts)
    pred_temp = np.load('data/%05d_x.npy'%i)
    Y_test = np.load('data/%05d_y.npy'%i)
    # print(pred_temp.shape,Y_test.shape)
        # if len(pred_temp) == BATCH or ii >= X_test.shape[0]-1:
            # pred_temp = np.array(pred_temp)
            # x_emb = make_seq(model_wv,pred_temp)
    Y_test_pr = model.predict(np.array(pred_temp),verbose=0)
    Y_test_pr = np.argmax(Y_test_pr,axis=-1)
    result = []
    temp_p = ''
    for ytp, yt, x in zip(Y_test_pr,Y_test,pred_temp):
        Y_pr.append(ytp)
        corr.append(yt)
        X.append(x)
    c+=1
    # if c == 10:
    #     break
    # print(Y[0])

Y_test_pr = np.array(Y_pr)
corr = np.array(corr)

print(Y_test_pr.shape)
print(corr.shape)
# print(Y_test_pr[0].shape)
# import sys
# sys.exit()
# print(corr[0],Y_test_pr[0])
# import sys
# sys.exit()
pr_tag = tokenizer_y.sequences_to_texts(Y_test_pr)
y_tag = tokenizer_y.sequences_to_texts(corr)
X = tokenizer_x.sequences_to_texts(X)

y_true = []
pred_ = []
import re
result = open('result.txt','w',encoding='utf-8')
true_ = open('true.txt','w',encoding='utf-8')
y_true = []
pred_ = []

for p,xx in zip(pr_tag,X):
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
    result.write(''.join(xx)+'\n')

for p,xx in zip(y_tag,X):
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
    true_.write(''.join(xx)+'\n')

        # temp = pp
    # print(xx)
        # temp = pp
    # print(p)
import sys
sys.exit()

for t,p in zip(y_tag,pr_tag):
    t = t.replace('_','O')
    p = p.replace('_','O')
    tt = t.split(' ')
    pp = p.split(' ')

    tt = tt + ['O'] * (max_len - len(tt))
    pp = pp + ['O'] * (max_len - len(pp))
    y_true.append(tt)
    pred_.append(pp)
    result.write(p+'\n')
    true_.write(t+'\n')    
print(np.array(y_true).shape)
print(np.array(pred_).shape)
result.close()
true_.close()
# print(type(pr_tag[0]))
# print(type(y_tag[0]))
# np.save('Y_test',Y_test)
# print(type(y_tag))
# print(type(pr_tag))
# import sys
# sys.exit()
# pr_tag = pr_tag.to_list()
# y_tag = y_tag.to_list()
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
print(classification_report(y_true,pred_))