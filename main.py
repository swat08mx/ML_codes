import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics, svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
data1 = pd.read_csv("C:/Users/swatt/PycharmProjects/gene_expression/final_data.csv")

new=[]
neu=[]
for i in range(len(data1['A/C'])):
    if data1['A/C'][i] == 'A':
        new.append(i)
    else:
        neu.append(i)
#print(f"The A is {len(new)} and C is {len(neu)}")
temp = neu[:200]
df = data1.iloc[new]
df1 = data1.iloc[temp]
final = pd.concat([df,df1], ignore_index=False)
labels_two = final['A/C'].to_list()
temp=[]
for i in range(len(labels_two)):
    if labels_two[i]=='A':
        temp.append(1)
    else:
        temp.append(0)
temp = pd.DataFrame(temp, columns=['labels'])
final.drop('A/C', axis=1, inplace=True)
final.drop('Sample_ID', axis=1, inplace=True)

def metrices(pred, pred_prob, y_test):
    tp=0
    fn=0
    tn=0
    fp=0
    for i in range(len(pred)):
        if y_test[i]==1 and pred[i]==1:
            tp+=1
        elif y_test[i]==1 and pred[i]==0:
            fn+=1
        elif y_test[i]==0 and pred[i]==0:
            tn+=1
        elif pred[i]==1 and y_test[i]==0:
            fp+=1
    sensitivity = (tp / (tp + fn))*100
    specificity = (tn / (tn + fp))*100
    accuracy = metrics.accuracy_score(y_test, pred)
    precision = metrics.precision_score(y_test, pred)
    auc = metrics.roc_auc_score(y_test, pred_prob)
    ppv = (tp / (tp + fp))*100
    npv = (tn / (tn + fn))*100


def models(X_train, y_train, X_test):

    model_xg = XGBClassifier(n_estimators=10, max_depth=10, learning_rate=1, objective='binary:logistic')
    model_xg.fit(X_train, y_train)
    pred_prob_xg = model_xg.predict_proba(X_test)[:, 1]
    pred_xg = model_xg.predict(X_test)

    model_lr = LogisticRegression(max_iter=2000)
    model_lr.fit(X_train, y_train)
    pred_prob_lr = model_lr.predict_proba(X_test)[:, 1]
    pred_lr = model_lr.predict(X_test)

    model_rf = RandomForestClassifier(n_estimators=1000)
    model_rf.fit(X_train, y_train)
    pred_prob_rf = model_rf.predict_proba(X_test)[:, 1]
    pred_rf = model_rf.predict(X_test)

    model_sv = svm.SVC(probability=True, kernel='linear')
    model_sv.fit(X_train, y_train)
    pred_prob_sv = model_sv.predict_proba(X_test)[:, 1]
    pred_sv = model_sv.predict(X_test)

    xgboost = XG(X_train, y_train, X_test)
    logistic_regression = LR(X_train, y_train, X_test)
    random_forest = RF(X_train, y_train, X_test)
    support_vector_machine = SV(X_train, y_train, X_test)


