import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data1 = pd.read_csv("lasso_big_dataset.csv")
data2 = pd.read_csv('gxp_dataset.csv')
sc = StandardScaler(with_std=True, with_mean=True)
data1_scaled = sc.fit_transform(data1)
data1_scaled = pd.DataFrame(data1_scaled, columns=data1.columns)
data_trans = data1_scaled.transpose()
linked = linkage(data_trans, method='ward', metric='euclidean')
df_linked = pd.DataFrame(linked, columns=['c1','c2','distance','size'])
df_linked[['c1','c2','size']] = df_linked[['c1','c2','size']].astype('int')
plt.figure(figsize=(12, 6))
dendrogram(linked ,
            orientation='top',
            labels=data_trans.index,
            distance_sort='descending',
            show_leaf_counts=True)

plt.xlabel('Features')
plt.ylabel("Ward's distance")
plt.show()
num_c = 10
labels = fcluster(linked, t=num_c, criterion='maxclust')
correlations = []
for col in data1.columns:
  corr = data2['label'].corr(data1[col])
  corr = round(corr, 3)
  correlations.append(corr)
df_clusters = pd.DataFrame(list(zip(data1.columns , labels , correlations)),
                          columns=['feature','cluster','corr'])

df_clusters['abs_corr'] = df_clusters['corr'].abs()

df_clusters.sort_values(by=['cluster','abs_corr'], ascending=[True,False], inplace=True)
df_clusters.reset_index(drop=True, inplace=True)

target = ['NAT1', 'SPATA41', 'FER1L6-AS1', 'HLA-DPB2', 'EBLN1', 'ANKRD30BL', 'CT83', 'SNORA70F', 'IL10', 'IMMP1L', 'NUP210P1', 'SCGB1D2', 'TBX22', 'PRKD1', 'LINC00917', 'ANGPTL5', 'DEFB112', 'ZNF667-AS1', 'SLC25A43', 'SKA1']
# one=input("Enter the first cluster: ")
# two=input("Enter the second cluster: ")
# c2_features = df_clusters[df_clusters['cluster']==one]['feature'].tolist()
# c3_features = df_clusters[df_clusters['cluster']==two]['feature'].tolist()
# corr = data1[np.append(c2_features ,c3_features)].corr()

corre = data1[target].corr()

plt.figure(figsize=(12, 8)), sns.heatmap(corre,annot=True, cmap='coolwarm',linewidths=0.5, fmt=".1f",annot_kws={"size": 5}, vmin=-1, vmax=1)
#plt.title(f'Cluster {one} and {two}')
plt.show()