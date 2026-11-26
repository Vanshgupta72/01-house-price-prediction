import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 

warnings.filterwarnings('ignore')
data = pd.read_csv('ford.csv')
df = pd.DataFrame(data)

# STEP-1 EDA 

# print(df.shape)
# print(df.head(5))
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# sns.histplot(df['price'],bins = 10,kde = True) # for checking the output curve and its trend
# sns.heatmap(df.corr(numeric_only=True),annot=True)  # for correaltion matrix with numeric columns and main output at once 

# sns.boxplot(x=df['year'],y=df['price'])  # Relation between year and  price
# plt.xticks(rotation = 90)

# sns.scatterplot(x=df['mileage'],y=df['price'])

# sns.boxenplot(x=df['engineSize'],y=df['price'])

# sns.boxenplot(x=df['transmission'],y=df['price'])

# sns.boxplot(x=df['fuelType'],y=df['price'])

# sns.boxplot(x=df['model'],y=df['price'])
# plt.xticks(rotation = 270)

X = df.drop(columns=['price'])
y = df['price']    
X_1encoded = pd.get_dummies(X,columns=['model','transmission','fuelType'],drop_first=True)
X_1encoded=X_1encoded.astype(int)
encode = LabelEncoder()
columns = ['model','transmission','fuelType']
Xlabel = X
for i in columns:
    Xlabel[i]=encode.fit_transform(Xlabel[i])
scaler = StandardScaler()
numeric = ['year','mileage','tax','mpg','engineSize']
X_1encoded= scaler.fit_transform(X_1encoded[numeric])
Xlabel =  scaler.fit_transform(Xlabel[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg',
       'engineSize']])
