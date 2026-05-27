#lemmatization TF-IDF train/test split

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()

def feature_extraction(df_rd_label_num):
    '''
    lemmatization TF-IDF train/test split
    '''
    def lemmatize_text(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    df_rd_label_num = df_rd_label_num.copy()
    df_rd_label_num["text"] = df_rd_label_num["text"].apply(lemmatize_text)
    print('LEMMATIZATION TEXT EXAMPLE', '      ', df_rd_label_num["text"].iloc[0])      #print example

    #encoding TF-IDF
    X = vectorizer.fit_transform(df_rd_label_num["text"])
    y = df_rd_label_num["label_num"]

    #encoding verifications
    print(X.shape)
    print(X.nnz)
    print(y.shape)
    print(y.value_counts())
    print(len(vectorizer.vocabulary_))
    print(list(vectorizer.vocabulary_.keys())[:20])

    return X, y, vectorizer


def split_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))

    #sanity check
    print(y_train.head(10))
    print(y_test.head(10))
    return X_train, X_test, y_train, y_test  