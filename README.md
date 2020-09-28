# MachineLearning-analysis
Machine Learning Analysis with python


#### Overview
Given a dataset of loan data it is required to build a model to predict the risk of a loan, providing some information.
As the loan dataset provided is highly imbalanced due to the nature of the information (There is usually a much lower amount of high_risk data than low_risk). Then, it is required to perform appropriate adjustments to the data prior its analysis and then make a comparison between them: 
* Random Oversampling.
* SMOTE Oversampling. 
* Cluster Centroids Undersampling.
* Combination (Over and Under) Sampling.

###### Resources
* Data Sources: _LoanStats_2019Q1.csv_
* Software: Python 3.7.7, Visual Studio Code 1.45.1., Sklearn, imblearn
---

#### Challenge:

##### Analysis:
As stated in the _'Overview'_ section, four resampling methods have been perform to test (using Logistic Regression) which one of them is better for this imbalanced data.

After the execution of the Logistic Regression model, with the different algorithms, we have obtained the results below: 

```
Reminder:
Balanced_accuracy_score is the % of correctly predicted values.
Precision (or Positive Predicted Value) = TP / (TP + FP).
Recall (or True Positive Rate) = TP / (TP + FN).
[Where TP: True Positive, FN: False Negative, FP: False Positive]
```

* **Random Oversampling:**
  - Balanced_accuracy_score: 83.81%
  - Precision (high risk / low risk): 0.03 / 1.00
  - Recall (high risk / low risk): 0.83 / 0.84

  
* **SMOTE Oversampling:**
  - Balanced_accuracy_score: 83.27%
  - Precision (high risk / low risk): 0.03 / 1.00
  - Recall (high risk / low risk): 0.8 / 0.86

* **Cluster Centroids Undersampling:**
  - Balanced_accuracy_score: 80.36%
  - Precision (high risk / low risk): 0.02 / 1.00
  - Recall (high risk / low risk): 0.85 / 0.76

* **Combination (Over and Under) Sampling:**
  - Balanced_accuracy_score: 60.08%
  - Precision (high risk / low risk): 0.01 / 1.00
  - Recall (high risk / low risk): 0.86 / 0.34


##### Conclusion:
Firstly given the nature of the analysis which would be to avoid the number of False Negatives (i.e. number of predicted Low_risk loans while they were in fact high risk) it leads to **search for the higher recall** number possible for high_risk loans.
That requirement is met by Combination sampling followed by Cluster Centroids Undersampling, with 0.86, and 0.85 for recall (high_risk)respectively.
However, the fact that the 'Combination Sampling' fails miserably for the low_risk loans it makes it extremely conservative predicting almost everything as high risk what could lead to the loose of opportunities for a very small cost (only 0.01 which is the diference between 0.86 and 0.85)
Additionally, the precision is also better in the case of the undersampling algorithm and the balanced accuracy score is +20%. Even though the latter metric would not be considered individually as decisive, now that it improves the overall benefit, it is important to be taken into account.

In consequence, the algorithm recommended would be **'Cluster Centroids Undersampling'**.



