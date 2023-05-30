import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ttest_ind, ttest_rel

class Sampler:
    def __init__(self, ratio, random_state):
        self.ratio = ratio
        self.random_state_ = random_state
        self.X = None
        self.y = None
        self.n_samples = None
        self.minority_class = None
        self.minority_indices = None
        self.majority_indices = None
        self.R = np.random.RandomState(self.random_state_)

    def fit_resample(self, X, y):
        self.X = X
        self.y = y
        self.minority_class = np.unique(self.y)[np.argmin(np.bincount(self.y))]
        self.minority_indices = np.where(self.y == self.minority_class)[0]
        self.majority_indices = np.where(self.y != self.minority_class)[0]
        
        desired_samples = int(self.ratio * len(self.majority_indices) - len(self.minority_indices))
        for i in range(desired_samples):
            index = self.minority_indices[self.R.randint(0, len(self.minority_indices) - 1)]
            self.X = np.vstack((self.X, self.X[index]))
            self.y = np.append(self.y, self.y[index])
            
        return self.X, self.y

class_proportion = 0.9
X, y = make_classification(n_samples=12345, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[class_proportion], flip_y=0,
                           random_state=12345)

data = pd.read_csv('Projekt-MSI/dataset.csv')
"""X2 = data.iloc[:, 2:]
y2 = (data.iloc[:, 1] == 'Male').astype(int)"""

X2 = data[['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']]
y2 = data['Response']

dataset=pd.DataFrame(y)

target_count = dataset.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

ratio=0.7

smote = SMOTE(sampling_strategy=ratio)
X_sm, y_sm = smote.fit_resample(X, y)

sampler = Sampler(ratio=0.7, random_state=12345)
X_sam, y_sam = sampler.fit_resample(X,y)

dataresample=pd.DataFrame(y_sam)
resample_count = dataresample.value_counts()
print('Class 0:', resample_count[0])
print('Class 1:', resample_count[1])
print('Proportion:', round(resample_count[0] / resample_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Przed')
#plt.show()
resample_count.plot(kind='bar', title='Po')
#plt.show()


n_splits = 2
n_repeats = 5

rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
scores = np.array([])
clf = GaussianNB()

for train_index, test_index in rkf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    clf.fit(X_resampled, y_resampled)
    predict = clf.predict(X_test)
    scores = np.append(scores, f1_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)

#print("F1 score: %.3f (%.3f)" % (mean_score, std_score))

np.save('scores.npy', scores)

random_scores = np.array([])
smote_scores = np.array([])

for train_index, test_index in rkf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_resampled_rd, y_resampled_rd = sampler.fit_resample(X_train, y_train)
    clf.fit(X_resampled_rd, y_resampled_rd)
    predict = clf.predict(X_test)
    random_scores = np.append(random_scores, f1_score(y_test, predict))
    
    X_resampled_sm, y_resampled_sm = smote.fit_resample(X_train, y_train)
    clf.fit(X_resampled_sm, y_resampled_sm)
    predict = clf.predict(X_test)
    smote_scores = np.append(smote_scores, f1_score(y_test, predict))


scores_combo = np.column_stack((random_scores, smote_scores))

num_classifiers = scores_combo.shape[1]
t_statistic_matrix = np.zeros((num_classifiers, num_classifiers))
p_value_matrix = np.zeros((num_classifiers, num_classifiers))
advantage_matrix = np.zeros((num_classifiers, num_classifiers), dtype=bool)

for i in range(num_classifiers):
    for j in range(num_classifiers):
            t_statistic, p_value = ttest_rel(scores_combo[:, i], scores_combo[:, j])
            t_statistic_matrix[i, j] = t_statistic
            p_value_matrix[i, j] = p_value
            advantage_matrix[i, j] = np.mean(scores_combo[:, i]) > np.mean(scores_combo[:, j])

alpha = 0.05
significant_advantage_matrix = (p_value_matrix < alpha)
statistical_advantage_matrix = advantage_matrix * significant_advantage_matrix

print("DANE SYSTETYCZNE:")
print(' ')
print(t_statistic_matrix)
print(' ')
print(p_value_matrix)
print(' ')
print(advantage_matrix)
print(' ')
print(significant_advantage_matrix)
print(' ')
print(statistical_advantage_matrix)

random_scores2 = np.array([])
smote_scores2 = np.array([])

for train_index, test_index in rkf.split(X2, y2):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_resampled_rd, y_resampled_rd = sampler.fit_resample(X_train, y_train)
    clf.fit(X_resampled_rd, y_resampled_rd)
    predict = clf.predict(X_test)
    random_scores2 = np.append(random_scores2, f1_score(y_test, predict))
    
    X_resampled_sm, y_resampled_sm = smote.fit_resample(X_train, y_train)
    clf.fit(X_resampled_sm, y_resampled_sm)
    predict = clf.predict(X_test)
    smote_scores2 = np.append(smote_scores2, f1_score(y_test, predict))


scores_combo2 = np.column_stack((random_scores2, smote_scores2))

num_classifiers2 = scores_combo2.shape[1]
t_statistic_matrix2 = np.zeros((num_classifiers2, num_classifiers2))
p_value_matrix2 = np.zeros((num_classifiers2, num_classifiers2))
advantage_matrix2 = np.zeros((num_classifiers2, num_classifiers2), dtype=bool)

for i in range(num_classifiers2):
    for j in range(num_classifiers2):
            t_statistic2, p_value2 = ttest_rel(scores_combo2[:, i], scores_combo2[:, j])
            t_statistic_matrix2[i, j] = t_statistic2
            p_value_matrix2[i, j] = p_value2
            advantage_matrix2[i, j] = np.mean(scores_combo2[:, i]) > np.mean(scores_combo2[:, j])

alpha = 0.05
significant_advantage_matrix2 = (p_value_matrix2 < alpha)
statistical_advantage_matrix2 = advantage_matrix2 * significant_advantage_matrix2

print("DANE RZECZYWISTE:")
print(' ')
print(t_statistic_matrix2)
print(' ')
print(p_value_matrix2)
print(' ')
print(advantage_matrix2)
print(' ')
print(significant_advantage_matrix2)
print(' ')
print(statistical_advantage_matrix2)