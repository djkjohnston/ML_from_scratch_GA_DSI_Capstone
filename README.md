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

* LinearRegression 
	* First draft completed, 11/25/2017
	* Used Boston dataset for comparison
	* Accuracy scores are analagous when compared to sklearn:
		* From Scratch: 0.740607742865
		* Sklearn: 0.740607742865
	* From scratch appears to be quicker, but that may not prove true at scale
		* From Scratch: 153 µs ± 30.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
		* Sklearn: 438 µs ± 9.06 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
* LogisticRegression
	* Something is wrong; all predicted values are either `inf` or `-inf`
* KNeighborsRegressor
	* First draft completed 11/25/2017
	* Used Boston dataset for comparison	
	* Accuracy scores are identical when compared to sklearn:
		* From Scratch: 0.639665439953224
		* Sklearn: 0.639665439953224
	* From scratch appears to be much slower, but hope to change with further revisions
		* From Scratch: 411 ms ± 9.18 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
		* Sklearn: 969 µs ± 29.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
	