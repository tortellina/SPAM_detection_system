# accuracy, precision, recall, f1-score, confusion matrix, data size experiments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def model_evaluation(X_train, X_test, y_train, y_test, model, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    return accuracy, precision, recall, f1


def diff_size_datasets_test(X_train, X_test, y_train, y_test, model):

    #small dataset test
    X_small = X_test[:50]
    y_small = y_test[:50]
    y_pred_small = model.predict(X_small)
    print("=== SMALL DATASET (50 samples) ===")
    print("Accuracy:", accuracy_score(y_small, y_pred_small))
    print("Precision:", precision_score(y_small, y_pred_small))
    print("Recall:", recall_score(y_small, y_pred_small))
    print("F1-score:", f1_score(y_small, y_pred_small))

    #medium dataset test
    X_medium = X_test[:500]
    y_medium = y_test[:500]
    y_pred_medium = model.predict(X_medium)
    print("\n=== MEDIUM DATASET (500 samples) ===")
    print("Accuracy:", accuracy_score(y_medium, y_pred_medium))
    print("Precision:", precision_score(y_medium, y_pred_medium))
    print("Recall:", recall_score(y_medium, y_pred_medium))
    print("F1-score:", f1_score(y_medium, y_pred_medium))

    #full dataset test
    y_pred_full = model.predict(X_test)
    print("\n=== FULL DATASET ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_full))
    print("Precision:", precision_score(y_test, y_pred_full))
    print("Recall:", recall_score(y_test, y_pred_full))
    print("F1-score:", f1_score(y_test, y_pred_full))



    #confusion matrices
    cm_small = confusion_matrix(y_small, y_pred_small)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_small)
    disp.plot()
    plt.title("Confusion Matrix - Small Dataset")
    plt.show()


    cm_medium = confusion_matrix(y_medium, y_pred_medium)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_medium)
    disp.plot()
    plt.title("Confusion Matrix - Medium Dataset")
    plt.show()

    cm_full = confusion_matrix(y_test, y_pred_full)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_full)
    disp.plot()
    plt.title("Confusion Matrix - Full Dataset")
    plt.show()