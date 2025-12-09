from statkitlearn.classification_model import PerceptronClassifier ## Import model class
import numpy as np ## import numpy
from sklearn.datasets import make_classification ## Only to make sample data
from sklearn.model_selection import train_test_split ## To split training and test data

X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=30) ## make sample data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) ##Split train and test data

perc = PerceptronClassifier() ## Make object
perc.fit(X_train,y_train) ## fit training data
y_pred = perc.predict(X_test) ## Get Predictions for test data
print(perc.weights) ##Get the weights