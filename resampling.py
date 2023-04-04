import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

class_proportion = 0.9
X, y = make_classification(n_samples=12345, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[class_proportion], flip_y=0,
                           random_state=12345)

dataset=pd.DataFrame(y)

target_count = dataset.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

ratio=0.6
random = RandomOverSampler(sampling_strategy=ratio)
X_rd, y_rd = random.fit_resample(X, y)

smote = SMOTE(sampling_strategy=ratio)
X_sm, y_sm = smote.fit_resample(X, y)

plot_2d_space(X, y, "Dane")

plot_2d_space(X_rd, y_rd, 'Random')
dataresample=pd.DataFrame(y_rd)
resample_count = dataresample.value_counts()
print('Class 0:', resample_count[0])
print('Class 1:', resample_count[1])
print('Proportion:', round(resample_count[0] / resample_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Przed')
plt.show()
resample_count.plot(kind='bar', title='Po')
plt.show()

rkf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1234)
scores = []
clf = GaussianNB()

for train_index, test_index in rkf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_resampled, y_resampled = random.fit_resample(X_train, y_train)
    clf.fit(X_resampled, y_resampled)
    predict = clf.predict(X_test)
    scores.append(f1_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)
print("F1 score: %.3f (%.3f)" % (mean_score, std_score))