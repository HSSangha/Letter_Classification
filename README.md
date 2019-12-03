# Letter Classification using different Machine Learning methods

## What is Machine Learning?

Machine Learning is known as the study of algorithms and statistical method that a computing system can use to perform task specific to application without using any instructions, solely relying on patterns and inference. In this module we will learn and understand some of the different methods available in sci-kit learn library. We will focus mainly on classfication methods with some overview of clustering methods.

For demonstrating the appliaction of different methods we will use a data set available on [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition) by UCI. 

## Data description

The objective is to identify one of the 26 capital letters in the English alphabet. The alphabet character images were based on 20 different fonts and each letter within these 20 fonts was randomly distorted to produce a file of 20,000 unique stimuli. Each stimulus was converted into 16 primitive numerical attributes (statistical moments and edge counts) which were then scaled to fit into a range of integer values from 0 through 15. 

Description of each stimuli is given below:

1. lettr capital letter (26 values from A to Z)
2. x-box horizontal position of box (integer)
3. y-box vertical position of box (integer)
4. width width of box (integer)
5. high height of box (integer)
6. onpix total # on pixels (integer)
7. x-bar mean x of on pixels in box (integer)
8. y-bar mean y of on pixels in box (integer)
9. x2bar mean x variance (integer)
10. y2bar mean y variance (integer)
11. xybar mean x y correlation (integer)
12. x2ybr mean of x * x * y (integer)
13. xy2br mean of x * y * y (integer)
14. x-ege mean edge count left to right (integer)
15. xegvy correlation of x-ege with y (integer)
16. y-ege mean edge count bottom to top (integer)
17. yegvx correlation of y-ege with x (integer)

[See the article in this link for more details.](https://link.springer.com/article/10.1007/BF00114162)


## Explore the data

First we load the data into a pandas dataframe after adding initial necessary libraries.

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

data = pd.read_csv('dataset_6_letter.csv', index_col=False)
data.head()
```
Now we check the dataset for missing values using describe function in pandas

```python
data.describe()
```
We get the following output:

![Link not found](https://raw.githubusercontent.com/HSSangha/Letter_Classification/master/describe.jpg)

We can see that no anamolies are there so we can proceed further to create our features and target dataset from the loaded dataset. We will name them x and y, respectively.

```python
x = data.iloc[:, 1:16]
y = data['class']
```

Before fitting any model onto it, we first check if whether if there is any correlation between any of the features. Such correlation lead to a model which is not trained on proper conditions. We check this by checking correlation between features using corr function and then visualize using seaborn:

```python
import seaborn as sns; sns.set() 

corr = x.corr()
sns.heatmap(corr)
```

![Link not found](https://raw.githubusercontent.com/HSSangha/Letter_Classification/master/corr.png)

We found that features x-box, y-box, width, high and onpix were highly correlated. We choose to remove those correlations which were above or close to 0.9. So we remove the high and onpix columns from the features dataset.

```python
x = x.drop(columns=['high','onpix'])
```

## Model the data

Before training any model we split the dataset into train and test dataset using sklearn.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
```

Now we explore some of the classifiers available on sklearn library.

### Stochastic Gradient Decent Classifier

The class SGDClassifier implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties for classification.

```python
from sklearn.linear_model import SGDClassifier

SGD = SGDClassifier(alpha=0.0001, max_iter=2000, tol=1e-3).fit(X_train, Y_train)
```

We predict labels for test set and then compare them to original labels and get an accuracy score and a confusion matrix.

```python
Y_hat = SGD.predict(X_test)

from sklearn.metrics import confusion_matrix

plt.figure(figsize=(10,10))
mat = confusion_matrix(Y_hat, Y_test)
sns.heatmap(mat.T, square=False, annot=True, fmt='d',cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')

from sklearn.metrics import accuracy_score

accuracy_score(Y_hat, Y_test)
out[]: 0.5033333333333333
```

We get the following output:

![Link not found](https://raw.githubusercontent.com/HSSangha/Letter_Classification/master/con1.png)

We can see that the accuracy is quite low and same is visible from confusion matrix. Now let's check out other methods.

### Support Vector Machine Classifier

The advantages of support vector machines are:
  * Effective in high dimensional spaces.
  * Still effective in cases where number of dimensions is greater than the number of samples.
  * Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
  * Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:
  * If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
  * SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
  
```python
from sklearn import svm

SVC = svm.SVC(gamma='scale')
SVC.fit(X_train, Y_train) 
```

We predict labels for test set and then compare them to original labels and get an accuracy score and a confusion matrix.

```python
Y_hat1 = SVC.predict(X_test)

plt.figure(figsize=(10,10))
mat1 = confusion_matrix(Y_hat1, Y_test)
sns.heatmap(mat1.T, square=False, annot=True, fmt='d',cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')

accuracy_score(Y_hat1, Y_test)
out[]: 0.9115
```
We get the following output:

![Link not found](https://raw.githubusercontent.com/HSSangha/Letter_Classification/master/con2.png)

### Nearest Neighbors Classification

Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

```python
from sklearn.neighbors import KNeighborsClassifier

NNC = KNeighborsClassifier().fit(X_train, Y_train)
```

We predict labels for test set and then compare them to original labels and get an accuracy score and a confusion matrix.

```python
Y_hat2 = NNC.predict(X_test)

plt.figure(figsize=(10,10))
mat2 = confusion_matrix(Y_hat2, Y_test)
sns.heatmap(mat2.T, square=False, annot=True, fmt='d',cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')

accuracy_score(Y_hat2, Y_test))
out[]: 0.9453333333333334
```

![Link not found](https://raw.githubusercontent.com/HSSangha/Letter_Classification/master/con3.png)

### Ensemble methods

The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.

Two families of ensemble methods are usually distinguished:
  * In averaging methods, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.
Examples: Bagging methods, Forests of randomized trees

  * By contrast, in boosting methods, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.
Examples: AdaBoost, Gradient Tree Boosting

We will be looking at gradient tree boosting algorithm.

Gradient Tree Boosting or Gradient Boosted Regression Trees (GBRT) is a generalization of boosting to arbitrary differentiable loss functions. GBRT is an accurate and effective off-the-shelf procedure that can be used for both regression and classification problems. Gradient Tree Boosting models are used in a variety of areas including Web search ranking and ecology.

The advantages of GBRT are:
  * Natural handling of data of mixed type (= heterogeneous features)
  * Predictive power
  * Robustness to outliers in output space (via robust loss functions)
  
The disadvantages of GBRT are:
  * Scalability, due to the sequential nature of boosting it can hardly be parallelized.
  
```python
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(n_estimators=200, random_state=0)
GBC.fit(X_train, Y_train)
```

## Communciate and visualize the results

What did you learn and do the results make sense?  Revisit your initial question and answer it.  H

### Class Exercise

In each project, I'd like to see a homework assignment that the class can do/evaluate to learn more about your data.  This should be a reproducible notebook that allows them to learn one or more aspects of your data workflow.  It is also an opportunity to share your research with your colleagues.

Here is an example of a fantastic project website:

https://stephenslab.github.io/ipynb-website/
