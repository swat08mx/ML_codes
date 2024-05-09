import pandas as pd
from xgboost import XGBClassifier
from tqdm import tqdm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import shap
import numpy as np
from sklearn.linear_model import Lasso
from sklearn import svm


data1 = pd.read_csv("gxp_dataset.csv")
data = pd.read_csv("lasso_dataset.csv")
temp = pd.DataFrame(data1['label'].to_list(), columns=['labels'])

# data1.drop(['label', 'Sample_ID'], axis=1, inplace=True)
# names = data1.columns
# import statistics
# import torch
# X_train, X_test, y_train, y_test = train_test_split(data1, temps, test_size=0.3, random_state=22)
# sc = StandardScaler()
# print(y_train)
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# X_train_tensor = torch.tensor(X_train)
# X_test_tensor = torch.tensor(X_test)
# y_train_tensor = torch.tensor(y_train)
# y_test_tensor = torch.tensor(y_test)
# # calling the model with the best parameter
# lasso1 = Lasso(alpha=0.00001, max_iter=10000)
# lasso1.fit(X_train_tensor, y_train_tensor)
# lasso1_coef = np.abs(lasso1.coef_)
#
# lists = lasso1_coef.tolist()
# print(min(lists))
# print(statistics.median(lists))
# median = statistics.median(lists)
# feature_subset=np.array(names)[lasso1_coef>median]
# print(len(feature_subset))
# df_new = data1[feature_subset]
#
# df_new.to_csv("lasso_big_dataset.csv", index=False)

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
   sensitivity_val = (tp / (tp + fn))
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
   specificity_val = (tn / (tn + fp))
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
   ppv_val = (tp / (tp + fp))
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
   npv_val = (tn / (tn + fn))
   return npv_val
def f1(y_test, pred):
   f1_val = metrics.f1_score(y_test_temp, pred)
   return f1_val


sensit_xg, specif_xg, accu_xg, prec_xg, auc_list_xg, ppv_list_xg, npv_list_xg, f1_xg = ([] for i in range(8))
sensit_lr, specif_lr, accu_lr, prec_lr, auc_list_lr, ppv_list_lr, npv_list_lr, f1_lr = ([] for i in range(8))
sensit_rf, specif_rf, accu_rf, prec_rf, auc_list_rf, ppv_list_rf, npv_list_rf, f1_rf = ([] for i in range(8))
sensit_sv, specif_sv, accu_sv, prec_sv, auc_list_sv, ppv_list_sv, npv_list_sv, f1_sv = ([] for i in range(8))


CV = KFold(n_splits=10, shuffle=True, random_state=42)


training, testing = [], []
for fold in CV.split(data):
   training.append(fold[0]), testing.append(fold[1])


preds=[]
actual=[]
preds_prob=[]
fig1 = plt.figure()
for i, (train_index, test_index) in tqdm(enumerate(zip(training, testing))):
   print(f"Fold {i}")
   X_train = data.iloc[train_index, :]
   X_test = data.iloc[test_index, :]
   y_train = temp.iloc[train_index]
   y_test = temp.iloc[test_index]
   y_test_temp = y_test['labels'].to_list()
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)


   # print("XGBoost")
   # k='XGBoost'
   # model_xg = XGBClassifier(learning_rate=0.05, device='gpu', objective='binary:logistic')
   # model_xg.fit(X_train, y_train.values.ravel())
   # pred_prob_xg = model_xg.predict_proba(X_test)[:, 1]
   # pred_xg = model_xg.predict(X_test)
   # fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_xg)
   # plt.title(f"ROC curve")
   # plt.plot(fpr, tpr, label=f'{k}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_xg))}')
   # plt.legend()
   # sensit_xg.append(float("{:.2f}".format(sensitivity(pred_xg, y_test_temp))))
   # specif_xg.append(float("{:.2f}".format(specificity(pred_xg, y_test_temp))))
   # accu_xg.append(float("{:.2f}".format(accuracy(y_test_temp, pred_xg))))
   # prec_xg.append(float("{:.2f}".format(precision(y_test_temp, pred_xg))))
   # auc_list_xg.append(float("{:.2f}".format(auc(y_test, pred_prob_xg))))
   # ppv_list_xg.append(float("{:.2f}".format(ppv(pred_xg, y_test_temp))))
   # npv_list_xg.append(float("{:.2f}".format(npv(pred_xg, y_test_temp))))
   # f1_xg.append(float("{:.2f}".format(f1(y_test_temp, pred_xg))))

   print("Logistic Regression")
   #k='Logistic Regression'
   model_lr = LogisticRegression(max_iter=2000)
   model_lr.fit(X_train, y_train.values.ravel())
   pred_prob_lr = model_lr.predict_proba(X_test)[:, 1]
   pred_lr = model_lr.predict(X_test)
   for values in pred_prob_lr:
       preds_prob.append(values)
   for values in pred_lr:
       preds.append(values)
   for values in y_test_temp:
       actual.append(values)
   fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_lr)
   plt.title(f"ROC curve for Logistic regression")
   plt.plot(fpr, tpr, label=f'Fold{i+1}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_lr))}')
   plt.legend()
   sensit_lr.append(float("{:.2f}".format(sensitivity(pred_lr, y_test_temp))))
   specif_lr.append(float("{:.2f}".format(specificity(pred_lr, y_test_temp))))
   accu_lr.append(float("{:.2f}".format(accuracy(y_test_temp, pred_lr))))
   prec_lr.append(float("{:.2f}".format(precision(y_test_temp, pred_lr))))
   auc_list_lr.append(float("{:.2f}".format(auc(y_test_temp, pred_prob_lr))))
   ppv_list_lr.append(float("{:.2f}".format(ppv(pred_lr, y_test_temp))))
   npv_list_lr.append(float("{:.2f}".format(npv(pred_lr, y_test_temp))))
   f1_lr.append(float("{:.2f}".format(f1(y_test_temp, pred_lr))))


   # print("Random Forest")
   # k='Random Forest'
   # model_rf = RandomForestClassifier(n_estimators=2000)
   # model_rf.fit(X_train, y_train.values.ravel())
   # pred_prob_rf = model_rf.predict_proba(X_test)[:, 1]
   # pred_rf = model_rf.predict(X_test)
   # fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_rf)
   # plt.title(f"ROC curve")
   # plt.plot(fpr, tpr, label=f'{k}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_rf))}')
   # plt.legend()
   # sensit_rf.append(float("{:.2f}".format(sensitivity(pred_rf, y_test_temp))))
   # specif_rf.append(float("{:.2f}".format(specificity(pred_rf, y_test_temp))))
   # accu_rf.append(float("{:.2f}".format(accuracy(y_test_temp, pred_rf))))
   # prec_rf.append(float("{:.2f}".format(precision(y_test_temp, pred_rf))))
   # auc_list_rf.append(float("{:.2f}".format(auc(y_test_temp, pred_prob_rf))))
   # ppv_list_rf.append(float("{:.2f}".format(ppv(pred_rf, y_test_temp))))
   # npv_list_rf.append(float("{:.2f}".format(npv(pred_rf, y_test_temp))))
   # f1_rf.append(float("{:.2f}".format(f1(y_test_temp, pred_rf))))
   #
   # print("Support Vector Machine")
   # k='SVM'
   # model_sv = svm.SVC(probability=True, kernel='linear')
   # model_sv.fit(X_train, y_train.values.ravel())
   # pred_prob_sv = model_sv.predict_proba(X_test)[:, 1]
   # pred_sv = model_sv.predict(X_test)
   # for values in pred_prob_sv:
   #     preds_prob.append(values)
   # for values in pred_sv:
   #     preds.append(values)
   # for values in y_test_temp:
   #     actual.append(values)
   # fpr, tpr, _ = metrics.roc_curve(y_test_temp, pred_prob_sv)
   # plt.title(f"ROC curve")
   # plt.plot(fpr, tpr, label=f'{k}, {"{:.2f}".format(metrics.roc_auc_score(y_test_temp, pred_prob_sv))}')
   # plt.legend()
   # plt.plot()
   # sensit_sv.append(float("{:.2f}".format(sensitivity(pred_sv, y_test_temp))))
   # specif_sv.append(float("{:.2f}".format(specificity(pred_sv, y_test_temp))))
   # accu_sv.append(float("{:.2f}".format(accuracy(y_test_temp, pred_sv))))
   # prec_sv.append(float("{:.2f}".format(precision(y_test_temp, pred_sv))))
   # auc_list_sv.append(float("{:.2f}".format(auc(y_test_temp, pred_prob_sv))))
   # ppv_list_sv.append(float("{:.2f}".format(ppv(pred_sv, y_test_temp))))
   # npv_list_sv.append(float("{:.2f}".format(npv(pred_sv, y_test_temp))))
   # f1_sv.append(float("{:.2f}".format(f1(y_test_temp, pred_sv))))


plt.show()
fig1.savefig('ROC_curve.png', bbox_inches='tight')
#dict_xg = {'AUC':auc_list_xg, 'Sensitivity':sensit_xg, 'Specificity': specif_xg, 'PPV':ppv_list_xg, 'NPV':npv_list_xg, 'Accuracy':accu_xg, 'Precision':prec_xg, 'F1':f1_xg}
dict_lr = {'AUC':auc_list_lr, 'Sensitivity':sensit_lr, 'Specificity': specif_lr, 'PPV':ppv_list_lr, 'NPV':npv_list_lr, 'Accuracy':accu_lr, 'Precision':prec_lr, 'F1':f1_lr}
#dict_rf = {'AUC':auc_list_rf, 'Sensitivity':sensit_rf, 'Specificity': specif_rf, 'PPV':ppv_list_rf, 'NPV':npv_list_rf, 'Accuracy':accu_rf, 'Precision':prec_rf, 'F1':f1_rf}
#dict_sv = {'AUC':auc_list_sv, 'Sensitivity':sensit_sv, 'Specificity': specif_sv, 'PPV':ppv_list_sv, 'NPV':npv_list_sv, 'Accuracy':accu_sv, 'Precision':prec_sv, 'F1':f1_sv}


#df_xg = pd.DataFrame(dict_xg)
df_lr = pd.DataFrame(dict_lr)
#df_rf = pd.DataFrame(dict_rf)
#df_sv = pd.DataFrame(dict_sv)
#print(df_xg)
print(df_lr)
#print(df_rf)
#print(df_sv)

# lists_xg = df_xg.columns
# xg_mean=[]
# xg_std=[]
# for items in lists_xg:
#     xg_mean.append(df_xg[items].mean(axis=0))
#     xg_std.append(df_xg[items].std(axis=0))
lists_lr = df_lr.columns
lr_mean=[]
lr_std=[]
for items in lists_lr:
   lr_mean.append(df_lr[items].mean(axis=0))
   lr_std.append(df_lr[items].std(axis=0))
# lists_rf = df_rf.columns
# rf_mean=[]
# rf_std=[]
# for items in lists_rf:
#     rf_mean.append(df_rf[items].mean(axis=0))
#     rf_std.append(df_rf[items].std(axis=0))
# lists_sv = df_sv.columns
# sv_mean=[]
# sv_std=[]
# for items in lists_sv:
#     sv_mean.append(df_sv[items].mean(axis=0))
#     sv_std.append(df_sv[items].std(axis=0))


# dict_mean = {'XGBoost':xg_mean, 'Logistic Regression':lr_mean, 'Random Forest': rf_mean, 'SVM':sv_mean}
# dict_std = {'XGBoost':xg_std, 'Logistic Regression':lr_std, 'Random Forest': rf_std, 'SVM':sv_std}


dict_mean = {'Logistic Regression':lr_mean}
dict_std = {'Logistic Regression':lr_std}


df_mean = pd.DataFrame(dict_mean)
df_std = pd.DataFrame(dict_std)
print(df_mean)
print(df_std)
fig2=plt.figure()
cm = confusion_matrix(actual, preds)
sns.heatmap(cm, annot=True, fmt='g', xticklabels=['Autism', 'Control'], yticklabels=['Autism', 'Control'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title("Confusion Matrix for Logistic regression", fontsize=17)
plt.show()
fig2.savefig('Confusion_matrix.png', bbox_inches='tight')

##Machine heavy code below ------ add comment after running ------
explainer = shap.KernelExplainer(model_lr.predict, X_train, feature_names=data.columns)
shap_values = explainer(X_test)
fig3 = plt.figure()
shap.plots.beeswarm(shap_values)
plt.show()
fig3.savefig('beeswarm_plot.png', bbox_inches='tight')
fig4 = plt.figure()
shap.summary_plot(shap_values, X_test)
plt.show()
fig4.savefig('Summary_plot.png', bbox_inches='tight')
fig5 = plt.figure()
shap.plots.heatmap(shap_values, max_display=12)
plt.show()
fig5.savefig('Heatmap.png', bbox_inches='tight')
fig6 = plt.figure()
shap.plots.waterfall(shap_values[0])
plt.show()
fig6.savefig('Waterfall_plot.png', bbox_inches='tight')
plt7 = plt.figure()
precision, recall, thresholds = precision_recall_curve(actual, preds_prob)
auc_score = metrics.auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
plt7.savefig('PR_curve.png', bbox_inches='tight')