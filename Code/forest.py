from sklearn import ensemble, preprocessing, metrics
from data import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

def predict(train_X,train_y,test_X):

    forest = ensemble.RandomForestClassifier(oob_score=True,n_estimators= 50, random_state=0)
    forest_fit = forest.fit(train_X, train_y)
    ans = forest.predict(test_X)
    return ans.astype(float)

if __name__ == '__main__':
    # evaluate
    dataX = load('X-{}.jl'.format(1))
    dataY = load('y-{}.jl'.format(1))
    train_X, test_X, train_y, test_y = train_test_split(dataX,dataY,test_size=0.2)
    ans = predict(train_X,train_y,test_X)
    print(accuracy_score(test_y,ans))