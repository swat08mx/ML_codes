import pandas as pd
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics, svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
data1 = pd.read_csv("C:/Users/swatt/PycharmProjects/gene_expression/files/final_data.csv")

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
    if labels_two[i] == 'A':
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
        if y_test[i] == 1 and pred[i] == 1:
            tp += 1
        elif y_test[i] == 1 and pred[i] == 0:
            fn += 1
        elif y_test[i] == 0 and pred[i] == 0:
            tn += 1
        elif pred[i] == 1 and y_test[i] == 0:
            fp += 1
    sensitivity = (tp / (tp + fn))*100
    specificity = (tn / (tn + fp))*100
    accuracy = metrics.accuracy_score(y_test, pred)
    precision = metrics.precision_score(y_test, pred)
    auc = metrics.roc_auc_score(y_test, pred_prob)
    ppv = (tp / (tp + fp))*100
    npv = (tn / (tn + fn))*100
    return(f"Sensitivity: {sensitivity}, Specificity: {specificity} accuracy: {accuracy}, precision: {precision}, auc: {auc}, ppv: {ppv}, npv: {npv}")

kf = KFold(10, random_state=None, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(final)):
    print(f"Fold {i}")
    X_train = final.iloc[train_index, :]
    X_test = final.iloc[test_index, :]
    y_train = temp.iloc[train_index]
    y_test = temp.iloc[test_index]
    y_test_temp = y_test['labels'].to_list()
    sc = StandardScaler()
    #X_train = sc.fit(X_train)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print("XGBoost")
    model_xg = XGBClassifier(n_estimators=10, max_depth=10, learning_rate=1, objective='binary:logistic')
    model_xg.fit(X_train, y_train.values.ravel())
    pred_prob_xg = model_xg.predict_proba(X_test)[:, 1]
    pred_xg = model_xg.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_xg)
    plt.title(f"ROC curve")
    plt.plot(fpr, tpr, label=f'Fold {i}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_xg))}')
    plt.legend()
    result = metrices(pred_xg, pred_prob_xg, y_test_temp)
    print(result)

    print("Logistic Regression")
    model_lr = LogisticRegression(max_iter=2000)
    model_lr.fit(X_train, y_train.values.ravel())
    pred_prob_lr = model_lr.predict_proba(X_test)[:, 1]
    pred_lr = model_lr.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_lr)
    plt.title(f"ROC curve")
    plt.plot(fpr, tpr, label=f'Fold {i}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_lr))}')
    plt.legend()
    result = metrices(pred_lr, pred_prob_lr, y_test_temp)
    print(result)

    print("Random Forest")
    model_rf = RandomForestClassifier(n_estimators=1000)
    model_rf.fit(X_train, y_train.values.ravel())
    pred_prob_rf = model_rf.predict_proba(X_test)[:, 1]
    pred_rf = model_rf.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_rf)
    plt.title(f"ROC curve")
    plt.plot(fpr, tpr, label=f'Fold {i}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_rf))}')
    plt.legend()
    result = metrices(pred_rf, pred_prob_rf, y_test_temp)
    print(result)

    print("Support Vector Machine")
    model_sv = svm.SVC(probability=True, kernel='linear')
    model_sv.fit(X_train, y_train.values.ravel())
    pred_prob_sv = model_sv.predict_proba(X_test)[:, 1]
    pred_sv = model_sv.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_sv)
    plt.title(f"ROC curve")
    plt.plot(fpr, tpr, label=f'Fold {i}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_sv))}')
    plt.legend()
    result = metrices(pred_sv, pred_prob_sv, y_test_temp)
    print(result)
    break



