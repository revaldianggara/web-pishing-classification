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

#