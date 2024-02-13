# Assignment5_Neural

video link: https://drive.google.com/file/d/1y6koBAk-oI0-mylaD1YAI_dgBE2p5Xqp/view?usp=sharing 
# 
Upender Reddy Bokka - 700746118
#
Description: Implemented Naive Bayes method using scikit-learn library Used train_test_split to create training and testing parts, We have Evaluated the model on test part using score and the classification_report(y_true, y_pred) We have also implemented linear svm method using scikit library Also used the same dataset Evaluated the model on test part using score and the classification_report(y_true, y_pred) The best algorithm and better accuracy is shown in the output. import pandas as pd from sklearn.svm import SVC from sklearn.naive_bayes import GaussianNB from sklearn.model_selection import train_test_split import time import warnings warnings.filterwarnings("ignore") from sklearn import metrics
