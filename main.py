#Librarys, Functions And Classes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

#Read Data From The Excel File
data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

#Data Cleaning
data.dropna()

#Seprate Data in X and y
X = data.iloc[:, 1:21]
y = data.iloc[:, 0]

#Balance Data
rus = SMOTE(sampling_strategy=1, random_state = 120)
X, y = rus.fit_resample(X, y)

#Data Scaling
scaling = StandardScaler()
X = scaling.fit_transform(X)

#Feature Selection
FeatureSelection = SelectPercentile(score_func=f_classif, percentile=30)
X = FeatureSelection.fit_transform(X, y)

#Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=120)

#LogisticRegression
logistic = LogisticRegression(solver='liblinear', C=10, random_state=120)
logistic.fit(X_train, y_train)
y_predictionLogisticRegression = logistic.predict(X_test)
print("LogisticRegression Score = ",logistic.score(X_test, y_test))
dg=classification_report( y_test, y_predictionLogisticRegression)
print(dg)

#DecisionTree
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=50, random_state=120)
dtree.fit(X_train, y_train)
ytree_prediction = dtree.predict(X_test)

#SVM
svc= LinearSVC()
svc.fit(X_train, y_train)
y_prediction = svc.predict(X_test)

#RandomForest
randomforce = RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=120)
randomforce.fit(X_train, y_train)
yrandom_prediction = randomforce.predict(X_test)

f = open("information.txt", "w")

f.write("LogisticRegression Score = "+str(logistic.score(X_test, y_test))+"\n")
f.write("Accuracy Score For LogisticRegression = "+str(metrics.accuracy_score(y_test, y_predictionLogisticRegression))+"\n")
f.write("Precision For LogisticRegression = "+str(metrics.precision_score(y_test, y_predictionLogisticRegression, average='micro'))+"\n")
f.write("F1 Score For LogisticRegression = "+str(metrics.f1_score(y_test, y_predictionLogisticRegression, average='micro'))+"\n")
f.write("Confusion Metrics For LogisticRegression = "+str(confusion_matrix(y_test, y_predictionLogisticRegression))+"\n")
f.write("Prediction For LogisticRegression = "+str(y_predictionLogisticRegression)+"\n")
f.write("Classification Report For LogisticRegression"+str(dg)+"\n")
f.write("------------------------------------------------------------\n")
f.write("Accuracy Score For DecisionTree = "+str(metrics.accuracy_score(y_test, ytree_prediction))+"\n")
f.write("Precision For DecisionTree = "+str(metrics.precision_score(y_test, ytree_prediction, average='micro'))+"\n")
f.write("F1 Score For DecisionTree = "+str(metrics.f1_score(y_test, ytree_prediction, average='micro'))+"\n")
f.write("Confusion Metrics For DecisionTree = "+str(confusion_matrix(y_test, ytree_prediction))+"\n")
f.write("Prediction For DecisionTree = "+str(ytree_prediction)+"\n")
f.write("------------------------------------------------------------\n")
f.write("Accuracy Score For SVM = "+str(metrics.accuracy_score(y_test, y_prediction))+"\n")
f.write("Precision For SVM = "+str(metrics.precision_score(y_test, y_prediction, average='micro'))+"\n")
f.write("F1 Score For SVM = "+str(metrics.f1_score(y_test, y_prediction, average='micro'))+"\n")
f.write("Confusion Metrics For SVM = "+str(confusion_matrix(y_test, y_prediction))+"\n")
f.write("Prediction For SVM = "+str(y_prediction)+"\n")
f.write("------------------------------------------------------------\n")
f.write("Accuracy Score For RandomForce = "+str(metrics.accuracy_score(y_test, yrandom_prediction))+"\n")
f.write("Precision For RandomForce = "+str(metrics.precision_score(y_test, yrandom_prediction, average='micro'))+"\n")
f.write("F1 Score For RandomForce = "+str(metrics.f1_score(y_test, yrandom_prediction, average='micro'))+"\n")
f.write("Confusion Metrics For RandomForce = "+str(confusion_matrix(y_test, yrandom_prediction))+"\n")
f.write("Prediction For RandomForce = "+str(yrandom_prediction)+"\n")

f.close()

#Outputs
print("Accuracy Score For LogisticRegression = ", metrics.accuracy_score(y_test, y_predictionLogisticRegression))
print("Precision For LogisticRegression = ", metrics.precision_score(y_test, y_predictionLogisticRegression, average='micro'))
print("F1 Score For LogisticRegression = ", metrics.f1_score(y_test, y_predictionLogisticRegression, average='micro'))
print("Confusion Metrics For LogisticRegression = ", confusion_matrix(y_test, y_predictionLogisticRegression))
print("Prediction For LogisticRegression = ", y_predictionLogisticRegression)
print("-----------------------------------------------------------")
print("Accuracy Score For DecisionTree = ", metrics.accuracy_score(y_test, ytree_prediction))
print("Precision For DecisionTree = ", metrics.precision_score(y_test, ytree_prediction, average='micro'))
print("F1 Score For DecisionTree = ", metrics.f1_score(y_test, ytree_prediction, average='micro'))
print("Confusion Metrics For DecisionTree = ", confusion_matrix(y_test, ytree_prediction))
print("Prediction For DecisionTree = ", ytree_prediction)
print("-----------------------------------------------------------")
print("Accuracy Score For SVM = ", metrics.accuracy_score(y_test, y_prediction))
print("Precision For SVM = ", metrics.precision_score(y_test, y_prediction, average='micro'))
print("F1 Score For SVM = ", metrics.f1_score(y_test, y_prediction, average='micro'))
print("Confusion Metrics For SVM = ", confusion_matrix(y_test, y_prediction))
print("Prediction For SVM = ", y_prediction)
print("-----------------------------------------------------------")
print("Accuracy Score For RandomForce = ", metrics.accuracy_score(y_test, yrandom_prediction))
print("Precision For RandomForce = ", metrics.precision_score(y_test, yrandom_prediction, average='micro'))
print("F1 Score For RandomForce = ", metrics.f1_score(y_test, yrandom_prediction, average='micro'))
print("Confusion Metrics For RandomForce = ", confusion_matrix(y_test, yrandom_prediction))
print("Prediction For RandomForce = ", yrandom_prediction)