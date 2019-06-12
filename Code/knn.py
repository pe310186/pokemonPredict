from sklearn import neighbors
from data import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

def predict(train_X,train_y,test_X):
    clf = neighbors.KNeighborsClassifier(weights='distance',n_neighbors=5)
    knn_clf = clf.fit(train_X, train_y)
    ans = knn_clf.predict_proba(test_X)
    return ans.astype(float)

if __name__ == '__main__':
    # evaluate
    dataX = load('X-{}.jl'.format(2))
    dataY = load('y-{}.jl'.format(2))
    train_X, test_X, train_y, test_y = train_test_split(dataX,dataY,test_size=0.2)
    ans = predict(train_X,train_y,test_X)
    print(ans.shape)
    #print(accuracy_score(test_y,ans))