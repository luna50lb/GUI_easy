import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, SimpleRNN, LSTM, Conv1D, GlobalMaxPooling1D, Lambda, Flatten #, Conv2D 

import numpy as np
import datetime 
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

from matplotlib import pyplot as plt
##import seaborn as sns

# データを数値化する
class Digitizing:
    def __init__(self, df_raw, num_words, maximumlen):
        print('instance Digitinzing created')
        self.df_raw=df_raw.drop_duplicates(subset=['text'],keep='first').reset_index();
        self.df_raw['one']=1;
        self.df_raw['text_id']=self.df_raw['one'].cumsum();
        self.df_raw['category_id']=self.df_raw.groupby(by=['category']).ngroup()
        x_train=self.df_raw['text'].tolist();
        y_train=self.df_raw['category_id'].tolist();
        id_train=self.df_raw['text_id'].tolist();
        vocab0=tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token='<UNK>')
        x_train=np.array(x_train);
        vocab0.fit_on_texts(x_train)
        #
        x_train = vocab0.texts_to_sequences(x_train)
        #x_test = vocab0.texts_to_sequences(x_test)
        #print('A0 x_train=', x_train)
        x_train = pad_sequences(x_train, maxlen=maximumlen, truncating='post')
        #x_test = pad_sequences(x_test, maxlen=maxlen, truncating='post')
        print('A1 x_train=', x_train)
        print('A1 y_train=', y_train)
        y_train=np.array(y_train)
        print('x_train, y_train shape=', x_train.shape, y_train.shape )
        print('type=', type(x_train), type(y_train))
        df_pre=pd.DataFrame({'X':x_train.tolist(), 'y':y_train, 'X_id':id_train})
        df_pre=df_pre.merge(self.df_raw[ ['text', 'category_id', 'category', 'text_id'] ], left_on=['X_id'], right_on=['text_id'], how='left', validate='1:1')
        df_pre=df_pre.drop(columns=['category_id', 'text_id'],axis=1)
        df_stacked=pd.DataFrame([])
        for j0 in np.arange(0, 5, 1):
            df_j0=pd.DataFrame([])
            df_j0=df_pre.sample(frac=1).reset_index(drop=True)
            df_j0=df_j0.rename(columns={'X':'X1', 'y':'y1', 'X_id': 'X_id1', 'text':'text1', 'category':'category1'} )
            df_j0=pd.concat([df_j0, df_pre],axis=1)
            df_stacked=pd.concat([df_stacked, df_j0], axis=0);
            del(df_j0)
        #
        def judge_group(jdict):
            allied=0;
            if jdict.get('y','diff')==jdict.get('y1','diff1'):
                allied=1
            return allied;
        #
        df_stacked['allied']=df_stacked.apply(judge_group,axis=1)
        self.df_train=df_stacked.copy()
        del(df_stacked)


class Train:
    def __init__(self):
        # データをロードし、深層学習の事前処理として数値化する
        if 'df_a' in dir():
            del(df_a)
        df_a=pd.read_excel("similarity_learning_dataset.xlsx",sheet_name='training_data')
        df_a=df_a[(~df_a['text'].isnull())].reset_index(drop=True)

        df_test=pd.read_excel("similarity_learning_dataset.xlsx",sheet_name='test_data')

        num_words_upper_bound=1300;
        max_num_features=250;

        dig0=Digitizing(df_raw=df_a, num_words=num_words_upper_bound, maximumlen=max_num_features)
        if np.array( dig0.df_train['X1'].tolist() ).max() + 1 == num_words_upper_bound:
            print('warning: word max=', np.array( dig0.df_train['X1'].tolist() ).max(), 'and upper bound = ', num_words_upper_bound)

        dig_test_data=Digitizing(df_raw=df_test, num_words=num_words_upper_bound, maximumlen=max_num_features)
        if np.array( dig_test_data.df_train['X1'].tolist() ).max() + 1 == num_words_upper_bound:
            print('warning: word max=', np.array( dig_test_data.df_train['X1'].tolist() ).max(), 'and upper bound = ', num_words_upper_bound)
        input_X1_test=tf.constant( dig_test_data.df_train['X1'].tolist() )
        input_X_test=tf.constant( dig_test_data.df_train['X'].tolist() )
        input_y_test=tf.constant( dig_test_data.df_train['allied'].tolist() )
        
        def construct_1nn(input_tensor):
            list_layers=[Embedding(input_dim=num_words_upper_bound+1, output_dim=32), LSTM(32), Dense(1, activation='sigmoid')]
            ##list_layers=[Embedding(input_dim=num_words_upper_bound+1, output_dim=30), Dense(30), Dense(1)]
            model0=tf.keras.Sequential(list_layers)
            model0.summary()
            return model0(input_tensor)

        def l2_distance(vects):
            tens0, tens1=vects;
            return tf.norm( (tens0-tens1), ord=2, axis=1 );

        def eucl_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        def contrastive_loss(y_true, y_pred):
            y_true=tf.dtypes.cast(y_true, tf.float64)
            y_pred=tf.dtypes.cast(y_pred, tf.float64)
            margin = 1
            #square_pred = K.square(y_pred)
            #margin_square = K.square(K.maximum(margin - y_pred, 0))
            #print('y_tru, y_pred=', y_true, y_pred, dir(y_true) )
            square_pred=tf.norm( y_pred ,ord=2)
            margin_square=tf.norm( tf.math.maximum( (margin - y_pred), 0), ord=2)
            #return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
            similarity_score=y_true * square_pred + (1 - y_true) * margin_square
            #print('sim score=', similarity_score, dir(similarity_score) )
            #return tf.math.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
            return y_true * square_pred + (1 - y_true) * margin_square

        def accuracyA(y_true, y_pred):
            '''Compute classification accuracy with a fixed threshold on distances.'''
            return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

        input_X1=tf.constant(dig0.df_train['X1'].tolist() )
        input_X =tf.constant(dig0.df_train['X'].tolist() )
        input_y=tf.constant( dig0.df_train['allied'].tolist() )
        print('X1, X=\n', input_X1, input_X)
        tens1=construct_1nn( input_X1 )
        tens=construct_1nn( input_X )
        print('tens1, type=', type(tens1), tens1.numpy().shape )

        input_shape=(input_X1.shape[1])
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        print('input_a=', input_a)
        processed_a=construct_1nn( input_a )
        processed_b=construct_1nn( input_b )
        distance = Lambda(l2_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        print('distance=', distance)
        model = Model([input_a, input_b], distance)
        model.summary()
        #print('model=', model)

        #distance = Lambda(l2_distance, output_shape=eucl_dist_output_shape)([tens1, tens])
        num_epochs=30

        model.compile(loss=contrastive_loss, optimizer=optimizers.Adam(learning_rate=0.0005), metrics=[accuracyA])
        #model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=128, epochs=epochs, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
        print('Optimization is in progress ... ')
        train_hist=model.fit([input_X1, input_X], input_y, batch_size=10, epochs=num_epochs, verbose=0)
        print('Optimization done')
        df_train_hist=pd.DataFrame({'epoch':range(1, num_epochs+1), 'loss':train_hist.history['loss'], 'accuracyA':train_hist.history['accuracyA'] } )
        # df_train_hist
        fig0, (ax0, ax1)=plt.subplots(nrows=2)
        #sns.lineplot(x='epoch',y='accuracy', data=df_train_hist,  ax=ax0)
        #sns.lineplot(x='epoch',y='loss', data=df_train_hist, ax=ax1)
        ax0.plot(df_train_hist['epoch'], df_train_hist['loss'], label='loss')
        ax1.plot(df_train_hist['epoch'], df_train_hist['accuracyA'], label='accuracy')
        ax0.set_yscale("log");
        ax1.set_yscale("log");

        plt.tight_layout();
        plt.show()
        plt.close('all')
        
        y_pred0=model.predict([input_X1_test, input_X_test])
        #df_check=pd.DataFrame({'pred_y':, 'correct_y':})
        print('y_pred0=\n', y_pred0)
        print('y true=', input_y_test)
        
#mf0=Train();
