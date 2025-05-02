#God is Holy
import os
os.chdir('C:\\Users\\Dr Vinod\\Desktop\\WD_python')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing

df = pd.read_csv('pca3.csv')
df.info()
#df= df.drop('Unnamed: 0', axis=1)
#df.info()

#PCA 
pca3_nocs = PCA(n_components=3)
pca3_nocs_comp = pca3_nocs.fit_transform(df)
pca3_nocs_comp #Components

pca3_nocs_egvct = pca3_nocs.components_ #Eigen vectors
pca3_nocs_egvct
'''
array([[ 0.1375708 ,  0.25045969, -0.95830278],
       [ 0.69903712,  0.66088917,  0.27307986],
       [ 0.70172743, -0.70745703, -0.08416157]])

'''
pca3_nocs_comp #Components
'''
array([[ 2.15142276, -0.17311941, -0.10681648],
       [-3.80418259, -2.88749898, -0.5104355 ],
       [-0.15321328, -0.98688598, -0.26941001],
       [ 4.7065185 ,  1.30153634, -0.65167999],
       [-1.29375788,  2.27912632, -0.44919235],
       [-4.0993133 ,  0.1435814 ,  0.80312818],
       [ 1.62582148, -2.23208282, -0.80281431],
       [-2.11448986,  3.2512433 ,  0.16837351],
       [ 0.2348172 ,  0.37304031, -0.27513962],
       [ 2.74637697, -1.06894049,  2.09398657]])
'''

pca3_nocs_egvl = pca3_nocs.explained_variance_ #Eigen Values
pca3_nocs_egvl
'''
array([8.27394258, 3.67612927, 0.74992815])
'''
