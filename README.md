# yelp_project
Yelp Restaurant Photo Classification
In this competition, Yelp is challenging Kagglers to build a model that automatically tags restaurants with multiple labels using a dataset of user-submitted photos. 

##Data Description:
train set: 234842 RGB images ---> 2000 business id

test set: 237152 RGB images ---> 10000 business id

labels: 9 * 0/1 labels(Nine Independent binary classification)

##Methods:
Feature Engineering: FeedForward Multiple Convolutional Neural Network +  pooling (max pooling and average pooling)

Dimension Reduction: PCA and Random Forest (Importances based on OOB)

Classifier: Tree based model(Random Forest, Boosting, etc), SGD classifier(SVM, Logistic Regression, Modified huber, etc), Discriminant Analysis(LDA and QDA)

Model Aggregation: Stacking(Boosting, Majority Vote)

## Mxnet features:
Inception-BN

Inception-21k

Inception-v3

