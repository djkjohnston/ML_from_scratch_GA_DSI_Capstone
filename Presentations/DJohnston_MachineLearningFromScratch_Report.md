---
title: Capstone: Machine Learning From Scratch

author: Daniel Johnston

date: December 6, 2017
---

# Project Description

* **Objective**: of this project is to build a number of known machine learning algorithms from scratch.
* **Parameters**:
	* Restricted to Python, NumPy, SciPy and similar libraries for implementation.
	* Scikit-Learn can only be used to source testing datasets and for performance comparison purposes. 
* Metrics:
	* Model Metrics:
		* Regression: $$R^{2}$$
		* Classification: Accuracy
		* Clustering: NA
	* Reach goal: runtimes similar to `sklearn` implimentations
* Target Outcomes:
	* Gain a deeper understanding of commonly used machine learning algorithms, with a focus on unsupervised learning and clustering tools
	* Ability to relate algorithms to a broad range of audiences
	
# Model Implimentation

Symbol conventions:

* $$X$$ - the known data used to train a model including both features (columns) and observations (rows)
* $$X_{n}$$ - an individual case or observation within $$X$$
* $$x_{n}$$ - an individual feature within $$X$$
* $$y$$ -  the outcomes of known data used to train a model
* $$y_{n}$$ - an individual outcome within $$y$$
* $$\beta_{n}$$ - coefficient of a fitted model corresponding to the feature $$x_n$$
* $$U$$ - previously unseen data that is run through a model to arrive at a prediction or estimation
* $$U_{n}$$ - an individual case or observation within $$U$$
* $$\widehat{y}$$ - the predicted outcomes for $$U$$
* $$\widehat{y}_{n}$$ - the predicted outcome for the individual case $$U_{n}$$

## Linear Regression, Solved Analytically
#### Model Description

Linear Regression assumes a fixed relationship between inputs $$x_{n}$$ and the output $$y$$

$$ y = \beta_{0}x_{0} + \beta_{1}x_{1} + ... \beta_{n}x_{n}$$



#### Psuedo Code
Linear Algebra hand waving
#### Testing and Performance
#### Critique	

## Logistic Regression, Solved Analytically
#### Model Description
#### Psuedo Code
Linear Algebra hand waving
#### Testing and Performance
#### Critique	

## K Nearest Neighbors for Regression
#### Model Description

The K Nearest Neighbors model provides predictions by identifying an arbitrary number, K, of closest neighbors, or observations, in a training set of data to a new observation. Closeness is defined by some measure of distance, such as Euclidean distance. In Regression applications, the outcomes of the identified K neighbors are aggregated, generally by taking the mean. This aggregated outcome is then applied to the new observation. 

#### Psuedo Code

1. Training data, both features $$X$$ and outcomes $$y$$ are stored.
1. For a new observation, $$U_{n}$$, the Euclidean distance between all points of $$X$$ and $$U_{n}$$ are measured.
1. The smallest $$K$$ distances are used to identify $$K$$ observations in $$X$$.
1. The values of $$y$$ for the identified $$K$$ observations are averaged to estimate the outcome, $$\widehat{y}$$. Example:
	* Where $$K$$ = 5, for a new case $$U_{n}$$ the 5 closest observations have the following values for $$y_{n}$$: 10, 12, 13, 9, 9
	* $$\widehat{y}_{n}$$ = $$(10 + 12 + 13 + 9 + 9)$$ = $$10.6$$

#### Testing and Performance
* Used Boston dataset for comparison	
* $$R^{2}$$ scores
	* From Scratch: 0.639665439953224
	* Sklearn: 0.639665439953224
* Runtime	
	* From Scratch: 2.28 ms ± 79.3 µs per loop
	* Sklearn: 969 µs ± 29.6 µs per loop 
	
#### Critique
My implimentation relies heavily on the `cdist` function from the `scipy` package. `cdist` takes two matrices and calculates the distance between each point within the two matrices. Sklearn also appears to make use of this function, but appears to utilize sparse matrices which may explain the runtime difference. Additionally, my implimentation only has one hyperparameter, $$K$$, while Sklearn has a number of tunable parameters including $$K$$, distance metric, and weights. 

Further development of my implimentation would explore the use of sparse matrices and additional distance metrics.


## K Nearest Neighbors for Classification
#### Model Description

The K Nearest Neighbors model provides predictions by identifying an arbitrary number, $$K$$, of closest neighbors, or observations, in a training set of data to a new observation. Closeness is defined by some measure of distance, such as Euclidean distance. In Classification applications, the outcomes of the identified K neighbors are used as votes to determine outcome of the new observation.  

#### Psuedo Code

1. Training data, both features $$X$$ and outcomes $$y$$ are stored.
1. For a new observation, $$U_{n}$$, the Euclidean distance between all points of $$X$$ and $$U_{n}$$ are measured.
1. The smallest $$K$$ distances are used to identify $$K$$ observations in $$X$$.
1. The values of $$y$$ for the identified $$K$$ observations are used as votes to determine the outcome, $$\widehat{y}$$. Example:
	* Where $$K$$ = 5, for a new case $$U_{n}$$ the 5 closest observations have the following values for $$y_{n}$$: $$True, False, True, False, False$$
	* Count of $$True$$ = 2
	* Count of $$False$$ = 3
	* $$3 > 2$$ therefor $$\widehat{y}_{n} = False$$
	
#### Testing and Performance
* Used Breast Canser  dataset for comparison
* Accuracy scores:
	* From Scratch: 0.965034965034965
	* Sklearn: 0.965034965034965
* Runtime
	* From Scratch: 5.83 ms ± 68.2 µs per loop 	* Sklearn: 1.21 ms ± 47 µs per loop 
	
#### Critique	

The same observations noted for K Nearest Neighbors for Regression apply here as well. Further, there is clearly a wider gap in the runtime performace between my implimentation and Sklearn's. I suspect that this is due to how the class voting was implimented.

Further developments would include support for multiple distance metrics and make explore the use of sparse matrices. I also need to impliment support for cases in which voting results in a tie.

## K-Means Clustering
#### Model Description
K-Means is an unsupervised clustering algorithm that tends to work best with well separated data with blob-like shapes. In this case $$K$$ refers to the number of clusters that the algorithm should find. The algorithm defines clusters by grouping observations into clusters based on the shortest distance to centroids, or the center of each clusters. Once clusters are assigned, the position of each centroid is moved to the centroid of each cluser. Cluster assignment is then reevaluated and centroids are moved again. This is repeated for a defined number of iterations or until some end condition is met.

#### Psuedo Code
1. Generate $$K$$ random starting centroids
1. Measure distance between centriods and all observations of $$X$$
1. Assign all observations of $$X$$ to a cluster based on the nearest centroid
1. Move each centroid to the center of the clustered observations of $$X$$
1. Repeat steps 2-4 until an end condition is met, such as the cluster assignment remain static, the centroids don't move, or the process has repeated a given number of times.
  
#### Testing and Performance
Runtime:
		* From Scratch: 328 ms ± 24.9 ms per loop 		* Sklearn: 64.4 ms ± 1.66 ms per loop 	* Cluster: Taking this with a grain of salt since KMeans performance depends so much on the starting points used
		* From Scratch: <img src='https://git.generalassemb.ly/raw/dannyboyjohnston/dsi_capstone_ml_from_scratch/master/assets/from_scratch_kmeans_blobs_pairplot.png'>
		* Sklearn: <img src='https://git.generalassemb.ly/raw/dannyboyjohnston/dsi_capstone_ml_from_scratch/master/assets/sklearn_kmeans_blobs_pairplot.png'>

#### Critique

What is interesting about KMeans is that the results are largely reliant on the starting position of the centroids. The same set of data can return vastly different results, and suggest different conclusions, so it is often helpful to run a number of different iterations to see how results can vary. 

My implimentation of K-Mean currently does not have a end condition beyond reaching a user-provided number of iterations. I make use of the `cdist` function to calculate the distance between the centroids and the observations of $$X$$. `cdist` works really well for this scenario because it takes in any number of centroids, so it scales well for any giving scenario. My implementation also only supports random assignment of the starting centroids, based on the uniform random distribution. This can results 


## DB Scan Clustering
#### Model Description
#### Psuedo Code
#### Testing and Performance
#### Critique	

## Bottom Up Hierarchical Clustering
#### Model Description
#### Psuedo Code
#### Testing and Performance
#### Critique	