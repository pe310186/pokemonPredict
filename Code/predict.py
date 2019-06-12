
from random import randint
import random
import numpy as np
from data import *
from sklearn import neighbors
from sklearn import ensemble, preprocessing, metrics

from joblib import dump, load

def convert(x):
    used_col = ['latitude', 'longitude', 'appearedTimeOfDay', 'appearedDayOfWeek',
       'terrainType', 'closeToWater', 'continent', 'temperature',
       'weatherIcon', 'urban', 'rural', 'gymDistanceKm', 'pokestopDistanceKm']
    mapping = {'afternoon':0,'evening':1,'morning':2,'night':3,'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'dummy_day':7,'false':0,'true':1,'Africa':0,'America':1,'America/Argentina':2,'America/Indiana':3,'America/Kentucky':4,'Asia':5,'Atlantic':6,'Australia':7,'Europe':8,'Indian':9,'Pacific':10,'clear-day':0,'clear-night':1,'cloudy':2,'fog':3,'partly-cloudy-day':4,'partly-cloudy-night':5,'rain':6,'wind':7}
    for i in range(len(x)):
        if isinstance(x[i],str):
            print(x[i])
            try:
                x[i] = float(x[i])
            except:
                x[i] = mapping[x[i]]
    return np.asarray([x]).astype(float)

def get_knn_predict(x):
    prediction = np.zeros((144,))
    ans = np.zeros((151,))
    for i in range(10):
        train_X = load('X-{}.jl'.format(i))
        train_y = load('y-{}.jl'.format(i))
        clf = neighbors.KNeighborsClassifier(weights='distance',n_neighbors=11)
        knn_clf = clf.fit(train_X, train_y)
        prediction += knn_clf.predict_proba(x)[0]
    # mapping to real id
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 148, 149]
    for i in range(144):
        ans[r[i]] = prediction[i]
    return ans/10

def get_forest_predict(x):
    prediction = np.zeros((144,))
    ans = np.zeros((151,))
    for i in range(1):
        train_X = load('X-{}.jl'.format(0))
        train_y = load('y-{}.jl'.format(0))
        forest = ensemble.RandomForestClassifier(oob_score=True,n_estimators= 50)
        forest_fit = forest.fit(train_X, train_y)
        prediction += forest.predict_proba(x)[0]
    # mapping to real id
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 148, 149]
    for i in range(144):
        ans[r[i]] = prediction[i]
    return ans/1


class Pokemon:
    def __init__(self):
        self.model = None

    def get_ids(self,x):
        pdt =  get_forest_predict(x)*.25 + get_knn_predict(x)*.25
        return pdt.argsort()[-10:][::-1]
    
if __name__ == '__main__':
    p = Pokemon()
    x = [100,999,'evening','Sunday',1,1,'Asia',10,'cloudy',0,1,1,0]
    x = convert(x)
    print(p.get_ids(x))
    