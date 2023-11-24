# ds
## ex.no.01 data cleaning and multivariate analysis
```python
import numpy as np
import pandas as pd
df=pd.read_csv("/content/Data_set.csv")
df
df.head(10)
df.info()
df.isnull()
df.isnull().sum()

df['show_name']=df['show_name'].fillna(df['aired_on'].mode()[0])
df['aired_on']=df['aired_on'].fillna(df['aired_on'].mode()[0])
df['original_network']=df['original_network'].fillna(df['aired_on'].mode()[0])
df.head()

df['rating']=df['rating'].fillna(df['rating'].mean())
df['current_overall_rank']=df['current_overall_rank'].fillna(df['current_overall_rank'].mean())
df.head()

df['watchers']=df['watchers'].fillna(df['watchers'].median())
df.head()

df.info()
df.isnull().sum()

#multivariate
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)

# Apply Principal Component Analysis (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_standardized)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Visualize the results
plt.scatter(df_pca['PC1'], df_pca['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Your Dataset')
plt.show()
```

## ex.no.02 IQR Score and univariate analysis
```python
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
df = pd.read_csv("/content/heights.csv")

sns.boxplot(data=df)
sns.scatterplot(data=df)

max =df['height'].quantile(0.90)
min =df['height'].quantile(0.15)
max
min

dq = df[((df['height']>=min)&(df['height']<=max))]
dq
low = min-1.5*iqr
high = max+1.5*iqr

dq = df[((df['height']>=min)&(df['height']<=max))]
dq
#univariate analysis
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('SuperStore.csv')
print(df)
df.head()
df.info()
df.dtypes
df['Postal Code'].value_counts()
sns.boxplot(x='Postal Code', data=df)
sns.countplot(x='Postal Code',data=df)
sns.distplot(df["Postal Code"])
df.describe()
```

## ex.no.03 z score and feature generation
```python
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
data = {'weight':[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]}
df = pd.DataFrame(data)
df
sns.boxplot(data=df)
z = np.abs(stats.zscore(df))
print(df[z['weight']>3])
val = [12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]
out=[]
def d_o(val):
ts=3
m=np.mean(val)
sd=np.std(val)

for i in val:
z=(i-m)/sd
if np.abs(z)>ts:
  out.append(i)
return out

op = d_o(val)
op

#feature generation
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from category_encoders import BinaryEncoder

df=pd.read_csv("/content/Encoding Data.csv")
df
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)
df2
pd.get_dummies(df2,columns=["nom_0"])
sc=StandardScaler()
df[["Height","Weight"]]=sc.fit_transform(df[["Height","Weight"]])
df.head(10)
scaler = RobustScaler()
df4=df.copy()
df4[["Height","Weight"]]=scaler.fit_transform(df4[["Height","Weight"]])
df4.head(10)
```
## ex.no. 04 data cleaning process and feature Transformation (05)
```python
import numpy as np
import pandas as pd
df=pd.read_csv("/content/Data_set.csv")
df
df.head(10)
df.info()
df.isnull()
df.isnull().sum()

df['show_name']=df['show_name'].fillna(df['aired_on'].mode()[0])
df['aired_on']=df['aired_on'].fillna(df['aired_on'].mode()[0])
df['original_network']=df['original_network'].fillna(df['aired_on'].mode()[0])
df.head()

df['rating']=df['rating'].fillna(df['rating'].mean())
df['current_overall_rank']=df['current_overall_rank'].fillna(df['current_overall_rank'].mean())
df.head()

df['watchers']=df['watchers'].fillna(df['watchers'].median())
df.head()

#feature transformation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer 
from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()

# Boxcox method
lambda_mle = stats.boxcox(df["Highly Positive Skew"])[0]

df7 = df.copy()
df7["Highly Positive Skew"] = stats.boxcox(df7["Highly Positive Skew"], lambda_mle)[0]
sm.qqplot(df7["Highly Positive Skew"], fit=True, line='45')
plt.show()
```
## ex.no.06 univariate (2b) and data visualization i)Line graph ii)Scatter plot
```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.read_csv("/content/Superstore.csv", encoding='unicode_escape')

# Line Graphs
plt.figure(figsize=(9, 6))
sns.lineplot(x="Segment", y="Region", data=df, marker='o')
plt.xticks(rotation=90)
sns.lineplot(x='Ship Mode', y='Category', hue="Segment", data=df)
sns.lineplot(x="Category", y="Sales", data=df, marker='o')
plt.show()

# Scatter Plots
sns.scatterplot(x='Category', y='Sub-Category', data=df)
sns.scatterplot(x='Category', y='Sub-Category', hue="Segment", data=df)
plt.figure(figsize=(10, 7))
sns.scatterplot(x="Region", y="Sales", data=df)
plt.xticks(rotation=90)
sns.scatterplot(x="Category", y="Sales", data=df)
plt.xticks(rotation=90)
sns.scatterplot(x=df["Quantity"], y=df["Discount"])
plt.scatter(df["Region"], df["Profit"], c="blue")
plt.show()
```

## ex.no.07 Multivariate analysis(1b) and data visualization i)Bar chart using matplot and seaborn ii)Violin plot
```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.read_csv("/content/Superstore.csv", encoding='unicode_escape')

# Separate Bar chart using Matplotlib
plt.figure(figsize=(10, 6))
plt.bar(df['Category'], df['Sales'], color='skyblue')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.title('Bar Chart of Sales by Category')
plt.xticks(rotation=90)
plt.show()

# Separate Violin plot using Seaborn
plt.figure(figsize=(10, 6))
sns.violinplot(x="Category", y="Sales", data=df, palette='pastel')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.title('Violin Plot of Sales by Category')
plt.xticks(rotation=90)
plt.show()
```

## ex.no.08 IQR Method(2a) and data visualization i)KDE Plot ii)Heat map
```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.read_csv("/content/Superstore.csv", encoding='unicode_escape')

# KDE Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(x="Profit", data=df, hue='Category')
plt.title('KDE Plot of Profit by Category')
plt.show()

# Heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()
```

## ex.no.09 univariate analysis(2b) and data visualization i)Histogram ii)Box plot
```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.read_csv("/content/Superstore.csv", encoding='unicode_escape')
df.head()

# Histogram
plt.hist(df["Sub-Category"], facecolor="peru", edgecolor="blue", bins=10)
plt.show()

# Box Plot
plt.boxplot(x="Sales", data=df)
plt.show()
```

## ex.no.10 data cleaning process(1a) and Feature Selection i)Any one Filter Method ii)Any one Wrapper Method
```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
data = pd.read_csv('/content/titanic_dataset.csv')
data.isnull().sum()
data.describe()
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Filter Method (Chi-squared test)
mdlsel_filter = SelectKBest(chi2, k=5)
mdlsel_filter.fit(X, y)
ix_filter = mdlsel_filter.get_support()
filtered_data = pd.DataFrame(mdlsel_filter.transform(X), columns=X.columns.values[ix_filter])  # Keep only the selected features

# Wrapper Method (Recursive Feature Elimination - RFE)
model = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5, criterion='entropy')
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)
selected_features = X.columns[fit.support_]
wrapper_data = pd.DataFrame(fit.transform(X), columns=selected_features)  # Keep only the selected features
print(filtered_data.head())
print(wrapper_data.head())
```
