# Data-Mining-Techniques

## Linear Regression
 Using weather datset, implemented simple Linear Regression between MinTemp and MaxTemp with co-efficients returned by the model , which minimizes the residual sum of squares between observed Max Temp in the dataset and predicted Max Temp.
 Performed data visualization using matplotlib
 
![LinearRegression](https://user-images.githubusercontent.com/57431137/97088641-519d3380-1600-11eb-8096-8f7a4bcd9c4f.png)

## Logistic Regression & Naive Bayes


## Principal Component Analysis & Clustering
Performed Pricipal Component Analysis on wine-quality dataset and from the variance ratio of the principal components we observe that PC0,PC1,PC2,PC3,PC4 contribute to nearly 80% of the data.
Using these 5 pricipal components, we perform K-means clustering. In order to choose we make use of the below elbow plot

![Elbow plot](https://user-images.githubusercontent.com/57431137/97088077-87401d80-15fc-11eb-8f0d-8a1f726c9f1b.png)

From the plot we observe that elbow is located at k=4 which show that k=4 is the good choice for dataset and implemented K-means clustering. While performing k-means we had dropped the target attribute column. K-means will have any impact on the target attribute.In k means we are considered to find the centroid of the cluster than the target attribute

![clustering](https://user-images.githubusercontent.com/57431137/97088191-4c8ab500-15fd-11eb-8cdd-3accdd81c5e2.png)

A large cluster which is formed by the combination of small clusters, Dendrograms of the cluster are used to actually split the cluster into multiple clusters of related data points. It starts by finding the nearest points based on Euclidean distance. From graph we can see that the dendograms have been created joining points.The vertical height of the dendogram shows the Euclidean distances between points. 

![hieraricalclustering](https://user-images.githubusercontent.com/57431137/97088195-501e3c00-15fd-11eb-99d1-a41388c970fc.png)

