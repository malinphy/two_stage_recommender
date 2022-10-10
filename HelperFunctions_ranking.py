import numpy as np 
import pandas as pd 
import re

def genre_splitter(genre_):
    all_genres_col= []
    genre_length = []
    for i in genre_:
        all_genres_col.append(i.split('|'))
        genre_length.append(len(i.split('|')))
    return all_genres_col,genre_length

def date_extractor(x):
    release_date = []
    for i in x:
        release_date.append((re.sub('\(|\)','',i).split(' ')[-1]))

    return release_date

def sequencer(df,col):
    var = df.groupby(col).aggregate(lambda tdf: tdf.unique().tolist()) 
    var = var.reset_index()

    return var
	
def train_test_maker(df,col): ### df will be in sequential format
    last_items = []
    for i in range(len(df)):
        last_items.append((df[col][i][-1]))

    mid_items = []
    for i in range(len(df)):
        mid_items.append(df[col][i][:-1])

    return (last_items, mid_items)

def train_neg_maker(num_neg, df,col,unique_item_enc):
    num_neg = num_neg
    neg3 = []
    ui= 0
    us = []
    for i in df[col]:
        pos_set = i
        neg2 = []
  
        for j in range(num_neg):
            neg1 = []
            for k in pos_set:
      
                neg_candidate = np.random.randint(1,len(unique_item_enc))
                while neg_candidate in pos_set:
                    neg_candidate = np.random.randint(1,len(unique_item_enc))
                us.append(ui)
      
                neg1.append((neg_candidate))
            neg2.append(np.array(neg1))
        ui += 1
        neg3.append(neg2)
    
    return us, neg3
	
def test_negative_maker(num_neg,df,col,unique_item_enc):
    num_neg = num_neg
    neg3 = []
    ui= 0
    us = []
    neg2 = []
    for i in df[col]:
        pos_set = i
  
        neg1 = []
        for j in range(num_neg):
    
            neg_candidate = np.random.randint(1,len(unique_item_enc))

            while neg_candidate == pos_set:

                neg_candidate = np.random.randint(1,len(unique_item_enc))
            us.append(ui)
      
            neg1.append((neg_candidate))
        neg2.append(np.array(neg1))

    return neg2