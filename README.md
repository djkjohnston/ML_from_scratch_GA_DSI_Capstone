# DSI Capstone - Machine Learning from Scratch

#### Author: Daniel Johnston


# Project Description

* **Objective**: of this project is to build a number of known machine learning algorithms from scratch.
* **Parameters**:
	* Restricted to Python, NumPy, SciPy and similar libraries for implementation.
	* Scikit-Learn can only be used to source testing datasets and for performance comparison purposes. 
* Metrics:
	* Model Metrics:
		* Regression: $R^{2}$
		* Classification: Accuracy
		* Clustering: NA
	* Reach goal: runtimes similar to `sklearn` implementations
* Target Outcomes:
	* Gain a deeper understanding of commonly used machine learning algorithms, with a focus on unsupervised learning and clustering tools
	* Ability to relate algorithms to a broad range of audiences
	
# Model Implementation

Symbol conventions:

* $X$ - the known data used to train a model including both features (columns) and observations (rows)
* $X_{n}$ - an individual case or observation within $X$
* $x_{n}$ - an individual feature within $X$
* $y$ -  the outcomes of known data used to train a model
* $y_{n}$ - an individual outcome within $y$
* $\beta_{n}$ - coefficient of a fitted model corresponding to the feature $x_n$
* $U$ - previously unseen data that is run through a model to arrive at a prediction or estimation
* $U_{n}$ - an individual case or observation within $U$
* $\widehat{y}$ - the predicted outcomes for $U$
* $\widehat{y}_{n}$ - the predicted outcome for the individual case $U_{n}$


## K Nearest Neighbors for Regression and Classification
#### Model Description

The K Nearest Neighbors model provides predictions by identifying an arbitrary number, $K$, of closest neighbors, or observations, in a training set of data to a new observation. Closeness is defined by some measure of distance, such as Euclidean distance. 

In Regression applications, the outcomes of the identified $K$ neighbors are aggregated, generally by taking the mean. This aggregated outcome is then applied to the new observation. 

In Classification applications, the outcomes of the identified $K$ neighbors are used as votes. The class with the highest number of votes is assigned to the new observation.

#### Pseudo Code

1. Training data, both features $X$ and outcomes $y$ are stored.
1. For a new observation, $U_{n}$, the Euclidean distance between all points of $X$ and $U_{n}$ are measured.
1. The smallest $K$ distances are used to identify $K$ observations in $X$.
1. Based on use case:
	1. For Regression: The values of $y$ for the identified $K$ observations are averaged to estimate the outcome, $\widehat{y}$. Example:
		* Where $K$ = 5, for a new case $U_{n}$ the 5 closest observations have the following values for $y_{n}$: 10, 12, 13, 9, 9
		* $\widehat{y}_{n}$ = $(10 + 12 + 13 + 9 + 9)$ = $10.6$
	1. For Classification: The values of $y$ for the identified $K$ observations are used as votes to determine the outcome, $\widehat{y}$. Example:
		* Where $K$ = 5, for a new case $U_{n}$ the 5 closest observations have the following values for $y_{n}$: $True, False, True, False, False$
		* Count of $True$ = 2
		* Count of $False$ = 3
		* $3 > 2$ therefor $\widehat{y}_{n} = False$
		
#### Visualization for k = 5

![K Nearest Neighbors Visualization](https://git.generalassemb.ly/raw/dannyboyjohnston/dsi_capstone_ml_from_scratch/master/for_visualizations/assets/plots_for_gifs/kneighbors/kneighbors.gif)

#### Testing and Performance
* Regression: 
	* Used Boston dataset for comparison	
	* $R^{2}$ scores
		* From Scratch: 0.639665439953224
		* Sklearn: 0.639665439953224
	* Runtime	
		* From Scratch: 2.28 ms ± 79.3 µs per loop
		* Sklearn: 969 µs ± 29.6 µs per loop 
* Classification
	* Used Breast Cancer  dataset for comparison
	* Accuracy scores:
		* From Scratch: 0.965034965034965
		* Sklearn: 0.965034965034965
	* Runtime
		* From Scratch: 5.83 ms ± 68.2 µs per loop
		* Sklearn: 1.21 ms ± 47 µs per loop 
	
#### Critique
My implementation relies heavily on the `cdist` function from the `scipy` package. `cdist` takes two matrices and calculates the distance between each point within the two matrices. Sklearn also appears to make use of this function, but appears to utilize sparse matrices which may explain the runtime difference. Additionally, my implementation only has one hyper-parameter, $K$, while Sklearn has a number of tunable parameters including $K$, distance metric, and weights. For classification specifically, there is clearly a wider gap in the runtime performance between my implementation and Sklearn's. I suspect that this is due to how the class voting was implemented.

Further development of my implementation would explore the use of sparse matrices and additional distance metrics. For classification uses, I'd like to explore different voting implementations as well as support support for cases in which voting results in a tie.


## K-Means Clustering
#### Model Description
K-Means is an unsupervised machine learning algorithms to identify clusters or segments of similar cases within a set of data. In this case $K$ refers to the number of clusters that the algorithm should find. The algorithm defines clusters by grouping observations into clusters based on the shortest distance to centroids, or the center of each clusters. Once clusters are assigned, the position of each centroid is moved to the centroid of each cluster. Cluster assignment is then reevaluated and centroids are moved again. This is repeated for a defined number of iterations or until some end condition is met. Since the clusters are based on a centroid, KMeans is best used when the data shows natural separation and blob like shapes. Kmeans often fails to be useful in cases where the data has irregular shapes such as half moons.

#### Pseudo Code
1. Generate $K$ random starting centroids
1. Measure distance between centroids and all observations of $X$
1. Assign all observations of $X$ to a cluster based on the nearest centroid
1. Move each centroid to the center of the clustered observations of $X$
1. Repeat steps 2-4 until an end condition is met, such as the cluster assignment remain static, the centroids don't move, or the process has repeated a given number of times.
  
#### Testing and Performance
* used `sklearn.datasets.make_blobs` for comparison
	* `make_blobs(n_samples=10000, n_features=12, centers=5, random_state=42)`
* Runtime:
	* From Scratch: 328 ms ± 24.9 ms per loop 
	* Sklearn: 64.4 ms ± 1.66 ms per loop 

#### Visualizing 5 Clusters

![Kmeans clustering](https://git.generalassemb.ly/raw/dannyboyjohnston/dsi_capstone_ml_from_scratch/master/for_visualizations/assets/plots_for_gifs/kmeans/kmeans.gif)


#### Critique

What is interesting about KMeans is that the results are largely reliant on the starting position of the centroids. The same set of data can return vastly different results, and suggest different conclusions, so it is often helpful to run a number of different iterations to see how results can vary. 

My implementation of K-Mean currently does not have a end condition beyond reaching a user-provided number of iterations. I make use of the `cdist` function to calculate the distance between the centroids and the observations of $X$. `cdist` works really well for this scenario because it takes in any number of centroids, so it scales well for any giving scenario. 

My implementation also only supports random assignment of the starting centroids, based on the uniform random distribution. This can results in starting centroids that will have no observations of $X$ assigned. During the implementation of the visualization piece, I found that this can result in a bug where all points collapse into a single cluster. As a stopgap measure, centroids with no closest points are not moved, which results in fewer clusters than $k$.

Further development would include investigation of different approaches to generating the starting centroids (different random distributions, logical controls, manual definition, etc.)


## DB Scan Clustering
#### Model Description
DB Scan is a clustering algorithm which identifies observations that a clustered together in any irregular shape. It does this by starting with a random observation $X_{n}$ and then finding other observations within a given radius, $r$. These observations are then clustered, and the model continues to search outwards to find more observations within $r$, essentially following the path or shape of the data. It continues the search until no new observations are found, at which point it considers the cluster finalized. The process then starts again with a previously unclustered point. Since DBScan identifies clusters based on the the inherent shape of the data, the results should always be consistent across the same set of data and $r$, though the clusters may be identified in a different order.

#### Pseudo Code
1. Define a radius, $r$ as the scanning parameter
1. Pick an unclustered random starting point, $X_n$, and assign it to cluster $c_{i}$
1. Instantiate a queue which will containing cases of $X$ to scan from. Add the random starting point, $X_{n}$, to the queue.
1. Identify unclustered cases of $X$ that are within $r$ distance of $X_{n}$. Assign them to cluster $c_{i}$. 
	* Add any previously unclustered points to the queue
	* Remove $X_{n}$ from the queue since we have completed the scanning operation for that case.
1. Identify unclustered cases of $X$ that are within $r$ distance of the next case in the queue. Assign them to cluster $c_{i}$. 
1. Repeat steps 4 and 5 until the queue is empty, i.e. no new cases within $r$ of cluster $c_{i}$ have been found
1. Repeat steps 2-6 until all cases of $X$ are clustered. 

#### Testing and Performance

* used `sklearn.datasets.make_moons` for comparison. This ensured that testing was conducted on irregular data distributions
	* `make_moons(n_samples=10000, random_state=42)`
* Runtime: Sklearn is nearly 4x faster
	* From Scratch: 2.87 s ± 275 ms per loop
	* Sklearn: 722 ms ± 40.6 ms per loop
* Cluster: They are identical


![From Scratch Clusters ](https://git.generalassemb.ly/raw/dannyboyjohnston/dsi_capstone_ml_from_scratch/master/for_visualizations/assets/from_scratch_dbscan.png)

![Sklearn Clusters](https://git.generalassemb.ly/raw/dannyboyjohnston/dsi_capstone_ml_from_scratch/master/for_visualizations/assets/sklearn_dbscan.png)

#### Visualizing DBScan

![DBScan visualiation](https://raw.githubusercontent.com/djkjohnston/ML_from_scratch_GA_DSI_Capstone/master/for_visualizations/assets/plots_for_gifs/dbscan2/dbscan2.gif)

		
#### Critique
My original implementation scanned for new cases within $r$ from all cases of $c_{i}$ simultaneously. The effect of this is that the cluster grew outward in waves, i.e. the outer boundary is expanded by $r$ consistently until it reaches the bounds of the cluster. This also resulted in a large number of redundant calculations that caused the implementation to scale very poorly. Using the same `make_moons()` benchmark above, my original implementation took 39.8 s ± 11 s per loop to execute.

My second implementation instead uses the queued approach outlined in the pseudo code which ensures that each clustered point is used only once to scan for new cases to add to the cluster, ie. the cluster grows from point to point. I still calculate the distance from the scan origin to all other points in the data $X$ at each iteration of the scan. It may be possible to do all of the distance calculations simultaneously and store them in a distance matrix; this matrix would then need to be queried or referenced to determine how the cluster should grow.

Another source of inefficiency in my implementation is that I use a list object for the queue. When removing scanned cases from the queue, the rest of the list object needs to be reindexed. This behavior will cause performance bottlenecks as the size of $X$ grows. Some possible alternatives include the `Redis`, `queue`, and `celery` packages.


## Bottom Up Hierarchical Clustering
#### Model Description
Bottom Up Hierarchical Clustering starts by assigning every case in $X$ to a separate cluster. Clusters are then joined iteratively, starting with the clusters that are closest together, according to some distance metric, and ending with the clusters that are farthest apart. The algorithm will eventually end with a single cluster containing all cases of $X$. The advantage to this approach is realized when you look at the history of how the clusters are joined. This enables the user to identify clusters at any step of the process and choose the ones that are of the most value. A Hierarchal Clustering is that it does not scale well; every case added to $X$ adds a considerable runtime.

#### Pseudo Code
1. Assign all cases of $X$ to their own cluster
1. Calculate distance between all clusters
1. Group the clusters with the shortest distance
1. Repeat step 3 until all cases have the same cluster

#### Testing and Performance

* Runtime:
	* `make_blobs(n_samples=1000, n_features=2, centers=5, random_state=42)`
		* From Scratch: 430 ms ± 58 ms per loop 
		* Sklearn: 17.4 ms ± 614 µs per loop

#### Critique

My implementation does not scale particularly well. This is because I create a queue of case-pairs a combination of different numpy tools. In general working with `numpy` means working without persistent indices, i.e. subsetting an array results in a new set of indices. It has been a challenge to work around this limitation but still identify the proper cases to group.

Future development efforts would seek to address this scalability issue. I may do away with the queueing approach (which essentially identifies the clustering order before the clustering actually takes place.) If I calculate the distances and clustering order at each step, it would allow me to more easily implement different distance metrics, such as using a cluster centroid instead of the closest point in a cluster. Another avenue for development would be to figure out how to draw a dendrogram based on the cluster history.
