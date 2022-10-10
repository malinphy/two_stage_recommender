from google.colab import drive
drive.mount('/content/drive')

!pip install --quiet scann==1.2.8
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import Model,layers,Input
from tensorflow.keras.layers import *
# import HelperFunctions
from HelperFunctions import time_conv,time_sorter,time_splitter,last_n_taker,unique_definer,corpus_creator,sequencer_multi,sequencer_unique,input_label_maker
from HelperFunctions import train_neg_maker,release_year
from metrics import mapk
from models import ret_model,ranking_model
# from models import raking_model
import os 
from sklearn.utils import shuffle
import pickle
import scann
from ast import literal_eval

EMBEDDING_DIM = 256
SEQUENCE_LEN = 20
BATCH_SIZE = 512
NUM_EPOCHS = 100
K = 40

saving_path = 'drive/MyDrive/Colab Notebooks/two_stage_rec/'
searcher_suffix = 'searcher'

### vector similarity search
searcher = scann.scann_ops_pybind.load_searcher(saving_path+searcher_suffix)

num_unique_movies = 3366
NUM_EPOCHS_RANKING = 5
ret_model_name = 'retrieval_model_weights.h5'
retrieval_model = ret_model(SEQUENCE_LEN,EMBEDDING_DIM,BATCH_SIZE,NUM_EPOCHS,num_unique_movies)
retrieval_model.load_weights(saving_path +str(NUM_EPOCHS)+'epochs_'+ret_model_name)

rank_model_name = 'ranking_model_weights.h5'
ranking_model = ranking_model(EMBEDDING_DIM)
ranking_model.load_weights(saving_path +str(NUM_EPOCHS_RANKING)+'epochs_'+rank_model_name)

# rank_model_name = 'ranking_model_weights.h5'
# ranking_model = ranking_model()
# ranking_model.load_weights(saving_path +str(NUM_EPOCHS_RANKING)+'epochs_'+rank_model_name)

item_embeddings = ((retrieval_model.get_layer("softmax_layer").weights)[0])
user_emb_model = Model(inputs = retrieval_model.inputs , outputs = retrieval_model.get_layer('d3_layer').output )

test_df = pd.read_csv(saving_path+'test_df_retrieval.csv')



input_movie_enc = [literal_eval(test_df['input_movie_enc'][i])
                             for i in range(len(test_df['input_movie_enc']))]

occupation_enc = [literal_eval(test_df['occupation_enc'][i])
                             for i in range(len(test_df['occupation_enc']))]   

gender_enc = [literal_eval(test_df['gender_enc'][i])
                             for i in range(len(test_df['gender_enc']))]  

age_enc = [literal_eval(test_df['age_enc'][i])
                             for i in range(len(test_df['age_enc']))] 

zip_code_enc = [literal_eval(test_df['zip_code_enc'][i])
                             for i in range(len(test_df['zip_code_enc']))]
test_df['input_movie_enc'] = input_movie_enc
test_df['occupation_enc'] = occupation_enc
test_df['gender_enc'] = gender_enc
test_df['age_enc'] = age_enc
test_df['zip_code_enc'] = zip_code_enc
user_id = 3

input_movie_enc_selected = [np.array(i)  for i in (test_df['input_movie_enc'][user_id:user_id+1])]
last_movie_enc_selected = [np.array(i)  for i in (test_df['last_movie_enc'][user_id:user_id+1])]
occupation_enc_selected = [np.array(i)  for i in (test_df['occupation_enc'][user_id:user_id+1])]
gender_enc_selected = [np.array(i)  for i in (test_df['gender_enc'][user_id:user_id+1])]
age_enc_selected = [np.array(i)  for i in (test_df['age_enc'][user_id:user_id+1])]
zip_code_enc_selected = [np.array(i)  for i in (test_df['zip_code_enc'][user_id:user_id+1])]

selected_user_emb = user_emb_model([
                    [tf.constant(input_movie_enc_selected)][0:1],
                    [tf.constant(occupation_enc_selected)][0:1],
                    [tf.constant(gender_enc_selected)][0:1],
                    [tf.constant(age_enc_selected)][0:1],
                    [tf.constant(zip_code_enc_selected)][0:1]
                                            ])

selected_index, selected_distance = searcher.search(np.array(selected_user_emb).ravel())

selected_index[0:5]

test_df.iloc[[0,4,6]]

ranking_df = pd.read_csv(saving_path+'ranking_df_test_diluted.csv',index_col='item_id_enc')

ranking_df.loc[selected_index[0:5]]


input_user_id_rank = np.ones(len(ranking_df))
input_movie_rank = [np.array(i)  for i in (ranking_df['item_id_enc_rank'])]
input_release_year_rank = [np.array(i, dtype = 'int32')  for i in (ranking_df['release_date'])]
# input_genre_rank = [np.array(i)  for i in (ranking_df['genre_enc'])]
input_genre_count_rank = [np.array(i)  for i in (ranking_df['genre_count'])]
input_release_date_norm_rank = [np.array(i)  for i in (ranking_df['release_date_norm'])]
input_genre_rank = [literal_eval(ranking_df['genre_enc'][i])
                             for i in range(len(ranking_df['genre_enc']))]
ranking_df.head(3)
num = 4
ranking_predictions = ranking_model.predict(
    [tf.constant(input_user_id_rank)[num:num+1],
     tf.constant(input_movie_rank)[num:num+1],
     tf.constant(input_release_year_rank)[num:num+1],
     tf.constant(input_genre_rank)[num:num+1],
     tf.constant(input_genre_count_rank)[num:num+1],
     tf.constant(input_release_date_norm_rank)[num:num+1]
     ])

for i in np.argsort(np.transpose(ranking_predictions)):
    print(str(ranking_df['movie_title'][num+i]))