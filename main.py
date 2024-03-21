import pandas as pd
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics, svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
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
def sensitivity(pred, y_test):
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in range(len(pred)):
        if y_test[i] == 1 and pred[i] == 1:
            tp += 1
        elif y_test[i] == 1 and pred[i] == 0:
            fn += 1
        elif y_test[i] == 0 and pred[i] == 0:
            tn += 1
        elif pred[i] == 1 and y_test[i] == 0:
            fp += 1
    sensitivity_val = (tp / (tp + fn))*100
    return sensitivity_val
def specificity(pred, y_test):
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in range(len(pred)):
        if y_test[i] == 1 and pred[i] == 1:
            tp += 1
        elif y_test[i] == 1 and pred[i] == 0:
            fn += 1
        elif y_test[i] == 0 and pred[i] == 0:
            tn += 1
        elif pred[i] == 1 and y_test[i] == 0:
            fp += 1
    specificity_val = (tn / (tn + fp))*100
    return specificity_val
def accuracy(y_test, pred):
    accuracy_val = metrics.accuracy_score(y_test, pred)
    return accuracy_val
def precision(y_test, pred):
    precision_val = metrics.precision_score(y_test, pred)
    return precision_val
def auc(y_test, pred_prob):
    auc_val = metrics.roc_auc_score(y_test, pred_prob)
    return auc_val
def ppv(pred, y_test):
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in range(len(pred)):
        if y_test[i] == 1 and pred[i] == 1:
            tp += 1
        elif y_test[i] == 1 and pred[i] == 0:
            fn += 1
        elif y_test[i] == 0 and pred[i] == 0:
            tn += 1
        elif pred[i] == 1 and y_test[i] == 0:
            fp += 1
    ppv_val = (tp / (tp + fp))*100
    return ppv_val
def npv(pred, y_test):
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in range(len(pred)):
        if y_test[i] == 1 and pred[i] == 1:
            tp += 1
        elif y_test[i] == 1 and pred[i] == 0:
            fn += 1
        elif y_test[i] == 0 and pred[i] == 0:
            tn += 1
        elif pred[i] == 1 and y_test[i] == 0:
            fp += 1
    npv_val = (tn / (tn + fn))*100
    return npv_val
def f1(y_test, pred):
    f1_val = metrics.f1_score(y_test_temp, pred)
    return f1_val

sensit_xg, specif_xg, accu_xg, prec_xg, auc_list_xg, ppv_list_xg, npv_list_xg, f1_xg = ([] for i in range(8))
sensit_lr, specif_lr, accu_lr, prec_lr, auc_list_lr, ppv_list_lr, npv_list_lr, f1_lr = ([] for i in range(8))
sensit_rf, specif_rf, accu_rf, prec_rf, auc_list_rf, ppv_list_rf, npv_list_rf, f1_rf = ([] for i in range(8))
sensit_sv, specif_sv, accu_sv, prec_sv, auc_list_sv, ppv_list_sv, npv_list_sv, f1_sv = ([] for i in range(8))

kf = KFold(10, random_state=None, shuffle=True)
for i, (train_index, test_index) in tqdm(enumerate(kf.split(final))):
    print(f"Fold {i}")
    X_train = final.iloc[train_index, :]
    X_test = final.iloc[test_index, :]
    y_train = temp.iloc[train_index]
    y_test = temp.iloc[test_index]
    y_test_temp = y_test['labels'].to_list()
    sc = StandardScaler()
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
    plt.show()
    sensit_xg.append(float("{:.2f}".format(sensitivity(pred_xg, y_test_temp))))
    specif_xg.append(float("{:.2f}".format(specificity(pred_xg, y_test_temp))))
    accu_xg.append(float("{:.2f}".format(accuracy(y_test_temp, pred_xg))))
    prec_xg.append(float("{:.2f}".format(precision(y_test_temp, pred_xg))))
    auc_list_xg.append(float("{:.2f}".format(auc(y_test, pred_prob_xg))))
    ppv_list_xg.append(float("{:.2f}".format(ppv(pred_xg, y_test_temp))))
    npv_list_xg.append(float("{:.2f}".format(npv(pred_xg, y_test_temp))))
    f1_xg.append(float("{:.2f}".format(f1(y_test_temp, pred_xg))))

    print("Logistic Regression")
    model_lr = LogisticRegression(max_iter=2000)
    model_lr.fit(X_train, y_train.values.ravel())
    pred_prob_lr = model_lr.predict_proba(X_test)[:, 1]
    pred_lr = model_lr.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_lr)
    plt.title(f"ROC curve")
    plt.plot(fpr, tpr, label=f'Fold {i}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_lr))}')
    plt.legend()
    plt.show()
    sensit_lr.append(float("{:.2f}".format(sensitivity(pred_lr, y_test_temp))))
    specif_lr.append(float("{:.2f}".format(specificity(pred_lr, y_test_temp))))
    accu_lr.append(float("{:.2f}".format(accuracy(y_test_temp, pred_lr))))
    prec_lr.append(float("{:.2f}".format(precision(y_test_temp, pred_lr))))
    auc_list_lr.append(float("{:.2f}".format(auc(y_test_temp, pred_prob_lr))))
    ppv_list_lr.append(float("{:.2f}".format(ppv(pred_lr, y_test_temp))))
    npv_list_lr.append(float("{:.2f}".format(npv(pred_lr, y_test_temp))))
    f1_lr.append(float("{:.2f}".format(f1(y_test_temp, pred_lr))))

    print("Random Forest")
    model_rf = RandomForestClassifier(n_estimators=1000)
    model_rf.fit(X_train, y_train.values.ravel())
    pred_prob_rf = model_rf.predict_proba(X_test)[:, 1]
    pred_rf = model_rf.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_rf)
    plt.title(f"ROC curve")
    plt.plot(fpr, tpr, label=f'Fold {i}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_rf))}')
    plt.legend()
    plt.show()
    sensit_rf.append(float("{:.2f}".format(sensitivity(pred_rf, y_test_temp))))
    specif_rf.append(float("{:.2f}".format(specificity(pred_rf, y_test_temp))))
    accu_rf.append(float("{:.2f}".format(accuracy(y_test_temp, pred_rf))))
    prec_rf.append(float("{:.2f}".format(precision(y_test_temp, pred_rf))))
    auc_list_rf.append(float("{:.2f}".format(auc(y_test_temp, pred_prob_rf))))
    ppv_list_rf.append(float("{:.2f}".format(ppv(pred_rf, y_test_temp))))
    npv_list_rf.append(float("{:.2f}".format(npv(pred_rf, y_test_temp))))
    f1_rf.append(float("{:.2f}".format(f1(y_test_temp, pred_rf))))

    print("Support Vector Machine")
    model_sv = svm.SVC(probability=True, kernel='linear')
    model_sv.fit(X_train, y_train.values.ravel())
    pred_prob_sv = model_sv.predict_proba(X_test)[:, 1]
    pred_sv = model_sv.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_sv)
    plt.title(f"ROC curve")
    plt.plot(fpr, tpr, label=f'Fold {i}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_sv))}')
    plt.legend()
    plt.show()
    sensit_sv.append(float("{:.2f}".format(sensitivity(pred_sv, y_test_temp))))
    specif_sv.append(float("{:.2f}".format(specificity(pred_sv, y_test_temp))))
    accu_sv.append(float("{:.2f}".format(accuracy(y_test_temp, pred_lr))))
    prec_sv.append(float("{:.2f}".format(precision(y_test_temp, pred_lr))))
    auc_list_sv.append(float("{:.2f}".format(auc(y_test_temp, pred_prob_lr))))
    ppv_list_sv.append(float("{:.2f}".format(ppv(pred_sv, y_test_temp))))
    npv_list_sv.append(float("{:.2f}".format(npv(pred_sv, y_test_temp))))
    f1_sv.append(float("{:.2f}".format(f1(y_test_temp, pred_sv))))

dict_xg = {'AUC':auc_list_xg, 'Sensitivity':sensit_xg, 'Specificity': specif_xg, 'PPV':ppv_list_xg, 'NPV':npv_list_xg, 'Accuracy':accu_xg, 'Precision':prec_xg, 'F1':f1_xg}
dict_lr = {'AUC':auc_list_lr, 'Sensitivity':sensit_lr, 'Specificity': specif_lr, 'PPV':ppv_list_lr, 'NPV':npv_list_lr, 'Accuracy':accu_lr, 'Precision':prec_lr, 'F1':f1_lr}
dict_rf = {'AUC':auc_list_rf, 'Sensitivity':sensit_rf, 'Specificity': specif_rf, 'PPV':ppv_list_rf, 'NPV':npv_list_rf, 'Accuracy':accu_rf, 'Precision':prec_rf, 'F1':f1_rf}
dict_sv = {'AUC':auc_list_sv, 'Sensitivity':sensit_sv, 'Specificity': specif_sv, 'PPV':ppv_list_sv, 'NPV':npv_list_sv, 'Accuracy':accu_sv, 'Precision':prec_sv, 'F1':f1_sv}

df_xg = pd.DataFrame(dict_xg)
df_lr = pd.DataFrame(dict_lr)
df_rf = pd.DataFrame(dict_rf)
df_sv = pd.DataFrame(dict_sv)
print(df_xg)
print(df_lr)
print(df_rf)
print(df_sv)

lists_xg = df_xg.columns
for items in lists_xg:
    print(f" Mean of XGBoost {items}: {"{:.2f}".format(df_xg[items].mean(axis=0))}")
    print(f" Std dev of XGBoost {items}: {"{:.2f}".format(df_xg[items].std(axis=0))}")
lists_lr = df_lr.columns
for items in lists_lr:
    print(f" Mean of Logistic Regression {items}: {"{:.2f}".format(df_lr[items].mean(axis=0))}")
    print(f" Std dev of Logistic Regression {items}: {"{:.2f}".format(df_lr[items].std(axis=0))}")
lists_rf = df_rf.columns
for items in lists_rf:
    print(f" Mean of Random Forest {items}: {"{:.2f}".format(df_rf[items].mean(axis=0))}")
    print(f" Std dev of Random Forest {items}: {"{:.2f}".format(df_rf[items].std(axis=0))}")
lists_sv = df_sv.columns
for items in lists_sv:
    print(f" Mean of Support Vector Machine {items}: {"{:.2f}".format(df_sv[items].mean(axis=0))}")
    print(f" Std dev of Support Vector Machine {items}: {"{:.2f}".format(df_sv[items].std(axis=0))}")
