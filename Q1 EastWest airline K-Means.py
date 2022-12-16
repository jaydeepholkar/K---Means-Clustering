import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
df = pd.read_excel(r'D:\Data science\DataSets\EastWestAirlines.xlsx')
df.describe()
df.info()
df = df.rename(columns={'Award?' : 'Award'} )
airlines = df.drop(['ID#'], axis=1)
airlines.info()

### Identify duplicates records in the data ###
duplicates = airlines.duplicated()
duplicates
duplicates.sum()
# Removing Duplicates
al = airlines.drop_duplicates()
dupl = al.duplicated()
dupl.sum()

############## Outlier Treatment ###############
# Let's find outliers in Salaries
plt.figure(figsize=(20,10))
sns.boxplot(data=al)
# remove the outlier
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Balance','Qual_miles','cc2_miles','cc3_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12'])

df_t = winsor.fit_transform(al[['Balance','Qual_miles','cc2_miles','cc3_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12']])
# Let's see boxplot
plt.figure(figsize=(20,10))
sns.boxplot(data=df_t)

#################### Missing Values - Imputation ###########################
airlines.isna().sum()

# Normalized data frame (considering the numerical part of data)
al.describe()
def norm_func(i): 
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(al)
a = df_norm.describe()

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 
###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
al['clust'] = mb # creating a  new column and assigning it to new column 

al.head()
df_norm.head()
al.info()
airline = al.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]

airline.iloc[:,1:8].groupby(airline.clust).mean()

