import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import seaborn as sns

#baca data
data = pd.read_csv('dataset_phishing.csv')

# cek 5 data paling atas dan 5 data paling bawah
print(data.head())
print(data.tail())

#menjelaskan secara stastikal untuk memberi informasi yang ada di data
print(data.describe())
# statistik menunjukan bahwa 75 persentil pengamatan miliki kelas x

# semua kolom berbentuk
print(data.info())

#plot distirbusi data pake histogram
plt.figure(figsize=(8,8))
plt.hist(data.Result)

#melihat missing values
print(data.isnull().sum())

#generate matriks korelasi
print(data.corr())

plt.figure(figsize=(8,8))
sns.heatmap(data.corr())

print(data.corr()['Result'].sort_values())

# Hapus yang punya koefisien korelas antara +- 0.03
data.drop(['Favicon','Iframe','Redirect',
                'popUpWidnow','RightClick','Submitting_to_email'],axis=1,inplace=True)
print(len(data.columns))

# data praparation untuk modeling
y = data['Result'].values
X = data.drop(['Result'], axis = 1)

# Split data training & testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

#1 Klasifikasi RF
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print("Akruasi classifier RF:", accuracy_score(y_test, prediction))
fpr, tpr, thresh = roc_curve(y_test,prediction)
#menghitung ROC AUC
roc_auc = accuracy_score(y_test, prediction)

# plotting kurva ROC Random Forest
plt.plot(fpr, tpr, 'g', label='Random Forest')
plt.legend("Random Forest", loc='lower right')
plt.legend(loc='lower right')
#Generate confusion matriks
print("Confusion matriks RF Classifier:",confusion_matrix(y_test, prediction))

#2 klasifikasi menggunakan Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print("Akurasi regresi logistik:", accuracy_score(y_test, prediction))
print('Conf matriks regresi logistik:', confusion_matrix(y_test, prediction))
fpr, tpr, thresh = roc_curve(y_test, prediction)
roc_auc = accuracy_score(y_test, prediction)

#Plot ROC curve for logistik regression
plt.plot(fpr, tpr, 'orange', label = 'Logistik Regression')
plt.legend("Logistik Regression", loc='lower right')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc='lower right')

#3 Klasifikasi menggunakan SVC
from sklearn.svm import SVC
svc_l = SVC(kernel= "linear", C = 0.025)
svc_l = svc_l.fit(X_train, y_train)
prediction = svc_l.predict(X_test)
print("Akurasi menggunakan SVM-Linear:", accuracy_score(y_test, prediction))
fpr, tpr, thresh = roc_curve(y_test, prediction)
roc_auc = accuracy_score(y_test, prediction)

# Plot ROC curve untuk SVM-linear
plt.plot(fpr,tpr,'b',label = 'SVM')
plt.legend("SVM", loc ='lower right')
plt.legend(loc ='lower right')
print("Conf matrix SVM-linear:",confusion_matrix(y_test,prediction))
plt.show()

#Implementasi RFE
from sklearn.feature_selection import RFE
rfe = RFE(rfc,27)
rfe = rfe.fit(X_train, y_train)               # Train RF classifier with only 27 features now
pred = rfe.predict(X_test)

# Test accuracy on reduced data
print("Accuracy by RFClassifier after RFE is applied:", accuracy_score(y_test,pred))

rfe = RFE(svc_l,27)
rfe = rfe.fit(X_train, y_train)               # Train SVM with only 27 features now
pred = rfe.predict(X_test)
print("Accuracy by SVM-Linear after RFE is applied:", accuracy_score(y_test,pred))

rfe = RFE(logreg,27)
rfe = rfe.fit(X_train, y_train)              # Train Logistic-Reg with only 27 features now
pred = rfe.predict(X_test)
print("Accuracy by Logistic Regression after RFE is applied:", accuracy_score(y_test,pred))