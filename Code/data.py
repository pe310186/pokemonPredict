import pandas as pd
from sklearn import preprocessing

def get_data():
    df = pd.read_csv('300k.csv')
    df = df[(df != '?').all(axis=1)]
    target = df['class']
    target = pd.get_dummies(target,columns=['class'])
    # preprocess
    used_col = ['latitude', 'longitude','appearedTimeOfDay','appearedDayOfWeek','terrainType','closeToWater','continent','temperature','urban','rural','weatherIcon','population density','gymDistanceKm','pokestopDistanceKm' ]
    for col in df.columns:
        if col not in used_col:
            del df[col]
    mapping = {'afternoon':0,'evening':1,'morning':2,'night':3,'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'dummy_day':7,'False':0,'True':1,'Africa':0,'America':1,'America/Argentina':2,'America/Indiana':3,'America/Kentucky':4,'Asia':5,'Atlantic':6,'Australia':7,'Europe':8,'Indian':9,'Pacific':10,'clear-day':0,'clear-night':1,'cloudy':2,'fog':3,'partly-cloudy-day':4,'partly-cloudy-night':5,'rain':6,'wind':7}
    df = df.applymap(lambda s: mapping.get(s) if s in mapping else s)
    X,y = df.values.astype(float), target.values.astype(float)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X) 
    return X,y

def get_data_sparse():
    df = pd.read_csv('300k.csv')
    df = df[(df != '?').all(axis=1)]
    target = df['class']
    # preprocess
    used_col = ['latitude', 'longitude','appearedTimeOfDay','appearedDayOfWeek','terrainType','closeToWater','continent','temperature','urban','rural','weatherIcon','population density','gymDistanceKm','pokestopDistanceKm' ]
    for col in df.columns:
        if col not in used_col:
            del df[col]
    mapping = {'afternoon':0,'evening':1,'morning':2,'night':3,'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'dummy_day':7,'False':0,'True':1,'Africa':0,'America':1,'America/Argentina':2,'America/Indiana':3,'America/Kentucky':4,'Asia':5,'Atlantic':6,'Australia':7,'Europe':8,'Indian':9,'Pacific':10,'clear-day':0,'clear-night':1,'cloudy':2,'fog':3,'partly-cloudy-day':4,'partly-cloudy-night':5,'rain':6,'wind':7}
    df = df.applymap(lambda s: mapping.get(s) if s in mapping else s)
    X,y = df.values.astype(float), target.values.astype(int)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X) 
    return X,y
