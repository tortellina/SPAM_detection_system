#SPAM DETECTION SYSTEM USING NLP AND SUPPORT VECTOR CLASSIFICATION

This project implements an automated, modular Natural Language Processing (NLP) pipeline designed to classify emails into two categories: legitimate (ham) or malicious (spam).
Instead of relying on heavy, resource-intensive deep learning models, this system uses a streamlined machine learning framework. By pairing clean text preprocessing with a Linear Support Vector Classifier (LinearSVC), the project achieves high accuracy and ultra-fast processing speeds on moderate-sized email datasets.


##INSTALLATION
To run this project, you need Python installed on your system. 

All necessary libraries are managed through a standard requirements document.

-	Download the project files and ensure the requirements.txt file is in the same directory.
-	Open your terminal or command prompt in the project folder and run the following command to install all dependencies at once: <pip install -r requirements.txt>
    Make sure to change the directory of your dataset (spam_ham_dataset.csv) in the main.py to the directory in your computer.
-   Execute the central script to run the entire pipeline: < python main.py >

(Note: The necessary NLTK dictionary files like WordNet and stopword lists will download automatically when you run the code for the first time).


##FEATURES

Automated Data Preprocessing: Automatically cleans text by converting it to lowercase, stripping out noisy punctuation markers, and filtering out low-information English stopwords.

Lexical Lemmatization: Uses NLTK's WordNetLemmatizer to reduce different forms of a word (like "running", "runs") down to their base dictionary form, keeping the vocabulary clean.

TF-IDF Vectorization: Transforms raw email text into a mathematical matrix of numerical weights based on word importance, creating a consistent vocabulary space.
Multi-Scale Testing: Features an evaluation module that runs stress tests on small (50 samples), medium (500 samples), and full test sets to check metric stability.

Live Query Interface: Includes an interface to instantly pass a completely new, custom raw text string into the trained model to get an immediate classification verdict.


##RESULTS

The system was thoroughly evaluated across three separate data scenarios to verify its performance and stability under different workloads:
1. Small Dataset (50 samples)
•	Accuracy: 1.0 (100%)
•	Precision: 1.0 (100%)
•	Recall: 1.0 (100%)
•	F1-score: 1.0 (100%)
2. Medium Dataset (500 samples)
•	Accuracy: 0.99 (99.00%)
•	Precision: 0.9653 (96.53%)
•	Recall: 1.0 (100%)
•	F1-score: 0.9823 (98.23%)
3. Full Test Dataset (1,035 samples)
•	Accuracy: 0.9903 (99.03%)
•	Precision: 0.9677 (96.77%)
•	Recall: 1.0 (100%)
•	F1-score: 0.9836 (98.36%)


###ANALYSIS OF RESULTS

The model achieved perfect 1.0 scores on the small 50-sample subset because low data volumes are easily and perfectly separable by the classifier. As the data scales up to 500 and 1,035 samples, the vocabulary expands, leading to a minor drop in precision (~96.77%). This happens naturally because a few legitimate emails contain professional formatting or numbers that look slightly like marketing templates. Most importantly, the model kept 1.0 recall across every single test, meaning it successfully caught 100% of the spam threats without letting any slip through. Ultimately, these results confirm that a simple, well-preprocessed linear model is highly reliable for real-world email filtering.


##AUTHORS

Vittoria Tagliabue (9212846), course name: project: NLP, course code DLBAIPNLP01, tutor: Visieu Lac
