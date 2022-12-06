import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import argparse
import warnings
parser = argparse.ArgumentParser()
parser.add_argument('goal')
args = parser.parse_args()

warnings.filterwarnings("ignore")
seed=0
all_features = [
            'following',
            'followers', 
            'user_total_tweets',
            'user_likes_count', 
            '0', '1', '2', '3', '4',
            'text_length'
        ]

def prepare():
    data = pd.read_csv('data/doc_embedding/doc2vec_5.csv.gz', compression='gzip', low_memory=False, lineterminator='\n', index_col=None)
    data['text_length'] = data['text'].str.len()
    y_log = []
    for x in data['{}_count'.format(args.goal)].tolist():
        if x == 0:
            y_log.append(x)
        else:
            y_log.append(np.log(x))
    data['{}_count(log)'.format(args.goal)] = y_log
    return data

def baseline(data):
    X = data[all_features]
    y = data['{}_count(log)'.format(args.goal)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)
    rf = RandomForestRegressor(random_state=seed)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    mse = mean_squared_error(y_pred, y_test)
    print("Baseline mse:", mse)
    return mse

def categorization(x, t_pairs):
    if len(t_pairs) == 1:
        if x <= t_pairs[0]:
            return 0
        else:
            return 1
    elif len(t_pairs) == 2:
        if x <= t_pairs[0]:
            return 0
        elif t_pairs[0] < x <= t_pairs[1]:
            return 1
        else:
            return 2
    elif len(t_pairs) == 3:
        if x <= t_pairs[0]:
            return 0
        elif t_pairs[0] < x <= t_pairs[1]:
            return 1
        elif t_pairs[1] < x <= t_pairs[2]:
            return 2
        else:
            return 3

def categorize_data(data, t):
    data['{}_count_class'.format(args.goal)] = \
        data['{}_count'.format(args.goal)].apply(categorization, t_pairs=t)
    fractions = []
    for c in range(len(t)):
        fractions.append(data[data['{}_count_class'.format(args.goal)] == c].shape[0] / data.shape[0])
    return data, fractions

def classify(X_train_scaled, X_test_scaled, y_label_train, y_label_test, num_class):
    # train classifier
    clf = RandomForestClassifier(random_state=seed)
    clf.fit(X_train_scaled, y_label_train)
    pred_class = clf.predict(X_test_scaled)
    acc = accuracy_score(pred_class, y_label_test)
    if num_class > 2:
        precision = precision_score(pred_class, y_label_test, average='macro')
        recall = recall_score(pred_class, y_label_test, average='macro')
        f1 = f1_score(pred_class, y_label_test, average='macro')
    else:
        precision = precision_score(pred_class, y_label_test)
        recall = recall_score(pred_class, y_label_test)
        f1 = f1_score(pred_class, y_label_test)       
    return [acc, precision, recall, f1], pred_class

def algo(data, num_class):
    X = data[all_features]
    label = data['{}_count_class'.format(args.goal)]

    X_train, X_test, y_label_train, y_label_test = train_test_split(X, label, random_state=seed)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)

    clf_metric, pred_class = classify(X_train_scaled, X_test_scaled, y_label_train, y_label_test, num_class)

    # train regressor for classes separately
    rfs = []
    for c in range(num_class):
        idx_class = pd.DataFrame(y_label_train)[pd.DataFrame(y_label_train)['{}_count_class'.format(args.goal)] == c].index
        X_train_class = X_train.loc[idx_class]
        X_train_class_scaled = StandardScaler().fit_transform(X_train_class)
        y_train_log_class = data.loc[idx_class]['{}_count(log)'.format(args.goal)]
        rf = RandomForestRegressor(random_state=seed)
        rf.fit(X_train_class_scaled, y_train_log_class)
        rfs.append(rf)

    # test
    y_true_log = data.loc[X_test.index]['{}_count(log)'.format(args.goal)]
    pred_vals = []
    for i in tqdm(range(len(pred_class))):
        for c in range(num_class):
            if pred_class[i] == c:
                pred_val = rfs[c].predict([X_test_scaled[i]])
                pred_vals.append(pred_val[0])

    mse = mean_squared_error(pred_vals, y_true_log)
    result = clf_metric + [mse]
    return result

if __name__=="__main__":
    data = prepare()

    base = baseline(data)
    print("="*25 + "2 class" + "="*25)
    df_2class = pd.DataFrame(columns=['acc', 'precision', 'recall', 'f1', 'mse'])
    thresholds =  [[0], [1], [5], [10], [20], [50], [100], [300], [500], [800], [1000]]
    fractions = []
    for t in tqdm(thresholds):
        data_tranformed, frac = categorize_data(data.copy(), t)
        fractions.append(frac)
        tmp = algo(data_tranformed, len(thresholds[0]) + 1)
        df_2class.loc[len(df_2class)] = tmp
    df_2class['fraction'] = fractions
    df_2class['thresholds'] = thresholds
    df_2class.loc[len(df_2class)] = [0, 0, 0, 0, base, 0, "baseline"]
    print(df_2class)
    df_2class.to_csv('result/2class_{}.csv'.format(args.goal), index=False)

    print("="*25 + "3 class" + "="*25)
    df_3class = pd.DataFrame(columns=['acc', 'precision', 'recall', 'f1', 'mse'])
    thresholds =  [(0, 5), (5, 50), (10, 100), (20, 200), (50, 500), (100, 500), (300, 1000), (500, 1000), (800, 2000), (1000, 5000)]
    fractions = []
    for t in tqdm(thresholds):
        data_tranformed, frac = categorize_data(data.copy(), t)
        fractions.append(frac)
        tmp = algo(data_tranformed, len(thresholds[0]) + 1)
        df_3class.loc[len(df_3class)] = tmp
    df_3class['fraction'] = fractions
    df_3class['thresholds'] = thresholds
    df_3class.loc[len(df_3class)] = [0, 0, 0, 0, base, 0, "baseline"]
    print(df_3class)
    df_3class.to_csv('result/3class_{}.csv'.format(args.goal), index=False)
