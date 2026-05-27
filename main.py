
#SPAM DETECTION SYSTEM, based on text classificaiton powered by linear Support Vector Classification. 
import pandas as pd
from data_preprocessing import data_inspection, data_cleaning
from feature_extraction import feature_extraction, split_data
from train_model import model_training, model_prediction
from model_evaluation import model_evaluation, diff_size_datasets_test
from model_predictions import new_email_pred


#train/evaluation dataset size
test_size = 0.2
random_state = 42

#import dataset
df = pd.read_csv(r"C:\Users\vitto\Desktop\EXAMS\4 SEMESTER\2 - project NLP\Vittoria_Tagliabue_9212846_NLP_task2\spam_ham_dataset.csv")


#data preprocessing
data_inspection(df)
df_rd_label_num = data_cleaning(df)

#feature extraction
X, y, vectorizer = feature_extraction(df_rd_label_num)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

#model training and prediction
model = model_training(X_train, X_test, y_train, y_test)
y_pred = model_prediction(X_train, X_test, y_train, y_test, model)

#model evaluation
model_evaluation(X_train, X_test, y_train, y_test, model, y_pred)
diff_size_datasets_test(X_train, X_test, y_train, y_test, model)

#classification of a new email
email = input("Enter an email to classify: ")
new_prediction = new_email_pred(email, model, X, y, vectorizer)