# ds
## ex.n0.01 data cleaning and multivariate analysis
```
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

##
