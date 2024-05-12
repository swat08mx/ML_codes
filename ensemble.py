import pandas as pd
from xgboost import XGBClassifier
from tqdm import tqdm
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import KFold
import shap
import numpy as np
from sklearn.linear_model import Lasso
from sklearn import svm


data1 = pd.read_csv("gxp_dataset.csv")
data = pd.read_csv("lasso_big_dataset.csv")
temp = pd.DataFrame(data1['label'].to_list(), columns=['labels'])


preds=[]
actual=[]
preds_prob=[]
X_train, X_test, y_train, y_test = train_test_split(data, temp, random_state=12)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model_xg = XGBClassifier(learning_rate=0.05, device='gpu', objective='binary:logistic')
# model_lr = LogisticRegression(max_iter=2000)
model_rf = RandomForestClassifier(n_estimators=2000)
model_sv = svm.SVC(probability=True, kernel='linear')
clf = [('xgb', model_xg), ('rf', model_rf), ('svm', model_sv)]

lr = LogisticRegression()
stack_model = StackingClassifier(estimators=clf, final_estimator=lr)
score = cross_val_score(stack_model, X_train, y_train, cv=10, scoring='accuracy')
print(f"The accuracy is {score.mean()}")
