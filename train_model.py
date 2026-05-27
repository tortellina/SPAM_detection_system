# linear SVC training, fitting and prediction generation
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def model_training(X_train, X_test, y_train, y_test):
    model = LinearSVC(max_iter=10000)
    model.fit(X_train, y_train)

    #test
    # y_pred = model.predict(X_test)
    # print("Predictions:", y_pred[:10])
    # print("Actual:", y_test.iloc[:10].values)
    return model

def model_prediction(X_train, X_test, y_train, y_test, model):

    y_pred = model.predict(X_test)
    print("Predictions:", y_pred[:10])
    print("Actual:", y_test.iloc[:10].values)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return y_pred