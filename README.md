# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Mall Customers dataset and check for missing values.
2. Use the Elbow Method to find the optimal number of clusters by plotting WCSS vs number of clusters.
3. Apply KMeans clustering with the chosen number of clusters (5 clusters).
4. Visualize the customer segments by plotting clusters based on Annual Income and Spending Score.

## Program:

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Venkatachalam S
RegisterNumber:  212224220121

```
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Mall_Customers.csv')

df.head()

df.info()

df.isnull().sum()

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10)
  kmeans.fit(df.iloc[:,3:])
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.title('The Elbow Method')
plt.show()

km = KMeans(n_clusters=5, n_init=10)
km.fit(df.iloc[:,3:])
y_pred = km.predict(df.iloc[:,3:])
y_pred

df['cluster'] = y_pred
dt0=df[df["cluster"]==0]
dt1=df[df["cluster"]==1]
dt2=df[df["cluster"]==2]
dt3=df[df["cluster"]==3]
dt4=df[df["cluster"]==4]

plt.scatter(dt0["Annual Income (k$)"],dt0["Spending Score (1-100)"],c="red",label="cluster1")
plt.scatter(dt1["Annual Income (k$)"],dt1["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(dt2["Annual Income (k$)"],dt2["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(dt3["Annual Income (k$)"],dt3["Spending Score (1-100)"],c="black",label="cluster4")
plt.scatter(dt4["Annual Income (k$)"],dt4["Spending Score (1-100)"],c="yellow",label="cluster5")
plt.legend()
plt.title("Customer Segments")
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/d0b2b18c-466f-407e-a17f-c9dbfa1efbc3)
![image](https://github.com/user-attachments/assets/c82e0202-1d8b-4232-9beb-c3392e15c138)
![image](https://github.com/user-attachments/assets/d453c8a6-09fa-4d42-a0f1-f358475f191d)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
