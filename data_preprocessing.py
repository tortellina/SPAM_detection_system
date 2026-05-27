'''
SPAM DETECTION SYSTEM, based on text classificaiton powered by support vector classification. 
The preprocessing is done by lemmatization, tf-idf, linear SCV.

'''

#this page is dedicated to inspect and clean the data in order of preperare them for feature engineering

import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
import string

def data_inspection(df):
    '''
    shows the first 10 rows and the last 5 rows. displays the shape of rows and colums, columsn types and missing
    values and columns names

    '''
    print('------------------ DATA INSPECTION ----------------------')
    print(df.head(10))   #looking at first 10 rows
    print(df.tail(5))    # last 5 rows
    df.shape             # (rows, columns)
    df.info()            # column types + missing values
    df.columns           # column names

    print('------------------ END OF data inspection ----------------------')

def data_cleaning(df):
    '''
    
    '''
    print('----------------------------- DATA CLEANING -------------------------')
    try:
        df = df.drop(columns=["Unnamed: 0"]) #dropping the first column because it is unnamed and not usefull for the task
        df.isnull().sum()                    #double checking for missing values
        df_rd_label_num = df[["text", "label_num"]].copy()       #df reduced only to the text of the email and the label_num which are the only values that the models will need
    except:
        pass
    #removing stopwords from df_rd_label_num
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    # def remove_stopwords(text):
    #     words = text.split()
    #     filtered_words = [word for word in words if word.lower() not in stop_words]
    #     return " ".join(filtered_words)

    # df_rd_label_num["text"] = df_rd_label_num["text"].apply(remove_stopwords)
    # print(df_rd_label_num["text"].iloc[0])

    def preprocess_text(text):
        '''
        lowercase, removes puntuation and remove stop words
        '''

        # lowercase
        text = text.lower()

        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # remove stopwords
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]

        return " ".join(filtered_words)

    
    df_rd_label_num["text"] = df_rd_label_num["text"].apply(preprocess_text)    #apply preprocessing

    
    print('PREPROCESSED TEXT EXAMPLE', '      ', df_rd_label_num["text"].iloc[0])      #print example

    return df_rd_label_num

    print('----------------------------- END OF data cleaning -------------------------')