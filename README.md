# Assignment5_Neural

video link: 

Description: Implemented Naive Bayes method using scikit-learn library Used train_test_split to create training and testing parts, We have Evaluated the model on test part using score and the classification_report(y_true, y_pred) We have also implemented linear svm method using scikit library Also used the same dataset Evaluated the model on test part using score and the classification_report(y_true, y_pred) The best algorithm and better accuracy is shown in the output. import pandas as pd from sklearn.svm import SVC from sklearn.naive_bayes import GaussianNB from sklearn.model_selection import train_test_split import time import warnings warnings.filterwarnings("ignore") from sklearn import metrics

# Read the data
data = pd.read_csv('C:\Users\shiva\Downloads\NNDL_Code and Data\NNDL_Code and Data\glass.csv') print(data.shape) X_train, X_test = train_test_split( data, test_size=0.2, random_state=int(time.time()))

# Features columns
features = [ "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe" ]

# Naïve Bayes Classifier
gauss = GaussianNB()

# Train the classifier
gauss.fit( X_train[features].values, X_train["Type"] )

# Make predictions
y_pred = gauss.predict(X_test[features]) print("Naïve Bayes\nTotal number of points: {}\nMislabeled points : {}\nAccuracy {:05.2f}%" .format( X_test.shape[0], (X_test["Type"] != y_pred).sum(), 100 * (1 - (X_test["Type"] != y_pred).sum() / X_test.shape[0]) )) print("\n")

# Naïve Bayes Classifier performance
print(metrics.classification_report(X_test["Type"], y_pred))

# Linear Support Vector Classification
svc_linear = SVC(kernel='linear')

# train linear SVM model
svc_linear.fit( X_train[features].values, X_train["Type"] ) Y_pred = svc_linear.predict(X_test[features])

# Linear SVM Model performance
acc_svc = round(svc_linear.score( X_test[features].values, X_test["Type"]) * 100, 2) print("Linear SVM accuracy is:", acc_svc)

# Support vector classifier (SVC) with the radial basis function kernel (RBF)
svc_rbf = SVC(kernel='rbf') svc_rbf.fit( X_train[features].values, X_train["Type"] )

# Model predictions
Y_pred = svc_rbf.predict(X_test[features])

# SVM RBF Model performance
acc_svc = round(svc_rbf.score( X_test[features].values, X_test["Type"]) * 100, 2) print("SVM RBF model accuracy is:", acc_svc) print("\n") print(metrics.classification_report(X_test["Type"], Y_pred))
