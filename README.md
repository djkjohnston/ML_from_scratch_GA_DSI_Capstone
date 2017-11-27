# DSI Capstone - Machine Learning from Scratch

The goal of this project is to implement common/known machine learning algorithms using vanilla `python` and `numpy` only. `sklearn` will be used only as A) a source of datasets for testing (both toy and generated) and B) comparing the performance of manually defined algorithms.

To do:

* LinearRegression solved analytically (i.e. with linear algebra vs gradient descent)
* LogisticRegression
* KNearestNeighborsClassifier
* KNearestNeighborsRegressor
* Others

Each class should have the following capabilities:

* Instantiation
* Fitting
* Prediction
* Scoring
	* R^2 for regression
	* Accuracy for classification
	* Consider adding others
	
Status:

* **LinearRegression**
	* First draft completed, 11/25/2017
	* Used Boston dataset for comparison
	* R2 scores are analagous when compared to sklearn:
		* From Scratch: 0.740607742865
		* Sklearn: 0.740607742865
	* From scratch appears to be quicker, but that may not prove true at scale
		* From Scratch: 153 µs ± 30.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
		* Sklearn: 438 µs ± 9.06 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
* **LogisticRegression**
	* First draft completed, 11/26/2017
	* Used Breast Cancer dataset for comparison
	* Accuracy scores are analagous when compared to sklearn:
		* From Scratch: 0.954305799649
		* Sklearn: 0.959578207381
	* From scratch appears to be quicker, but that may not prove true at scale
		* From Scratch: 620 µs ± 31 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
		* Sklearn: 7.52 ms ± 313 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
	* Set up for binary classification only right now. Also need to figure out how to return probabilities?
* **KNeighborsRegressor**
	* First draft completed 11/25/2017
	* Used Boston dataset for comparison	
	* R2 scores are identical when compared to sklearn:
		* From Scratch: 0.639665439953224
		* Sklearn: 0.639665439953224
	* From scratch appears to be much slower, but hope to change with further revisions
		* From Scratch: 411 ms ± 9.18 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
			* Update 11/27/2017: updated distance function resulting in a shorter run time: 6.86 ms ± 152 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
		* Sklearn: 969 µs ± 29.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
* **KNeighborsClassifier**
	* First draft completed 11/26/2017
	* Used Breast Canser  dataset for comparison
	* Accuracy scores are identical when compared to sklearn:
		* From Scratch: 0.965034965034965
		* Sklearn: 0.965034965034965
	* From scratch appears to be much slower, but hope to change with further revisions
		* From Scratch: 506 ms ± 14.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
			* Update 11/27/2017: updated distance function resulting in a shorter run time: 8.38 ms ± 274 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
		* Sklearn: 1.21 ms ± 47 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)