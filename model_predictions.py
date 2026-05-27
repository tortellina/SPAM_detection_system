#prediction for a new unseen email
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer




def new_email_pred(email, model, X, y, vectorizer):
    
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    email = email.lower()
    words = email.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    clean_email = " ".join(words)
    email_vec = vectorizer.transform([clean_email])

    # ---- PREDICT ----
    prediction = model.predict(email_vec)

    # ---- OUTPUT ----
    if prediction[0] == 1:
        print("1 - THIS EMAIL IS SPAM")
    else:
        print("0 - HAM (legitimate)")