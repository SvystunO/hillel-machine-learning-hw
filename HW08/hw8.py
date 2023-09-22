from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import numpy as np
import time


def logistic_regression_cv(x_train, y_train, x_test, y_test):
    # Record the start time
    start_time = time.time()
    lr = LogisticRegressionCV(cv=5, max_iter=2000, multi_class='ovr')

    lr.fit(x_train, y_train.values)

    y_pred_proba = lr.predict_proba(x_test)

    rock_auc = roc_auc_score(y_test.values, y_pred_proba, multi_class='ovr')

    # Record the end time
    end_time = time.time()
    # Calculate the execution time
    execution_time = end_time - start_time

    return rock_auc, execution_time

def logistic_regression_cv_pca(num_demension, x_train, y_train, x_test, y_test):

    pca = PCA(n_components=num_demension)

    XD_train = pca.fit_transform(x_train)

    XD_test = pca.transform(x_test)

    rock_auc, execution_time = logistic_regression_cv(XD_train, y_train,  XD_test, y_test)

    return rock_auc, execution_time, pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum()

csv_file_path = 'WineQT.csv'
test_size = 0.2
random_state = 42

data = pd.read_csv(csv_file_path)

y = data['quality']  # Target
X = data.drop('quality', axis=1)  # Features
data.drop('quality', axis=1)  # Features
X = X.drop('Id', axis=1)  # Features

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on x_train and transform both x_train and x_test
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


'''
X_train_centered = x_train - x_train.mean(axis=0)
U, s, Vt = np.linalg.svd(X_train_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
print(Vt.shape)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train_centered[:, 0], X_train_centered[:, 1], X_train_centered[:, 2] )
ax.plot(*zip(c1, np.zeros(c1.shape)), color="red")
ax.plot(*zip(c2, np.zeros(c2.shape)), color="green")
#plt.show()

W2 = Vt.T[:, :2]
X2D = X_train_centered.dot(W2)
print(X2D.shape, W2.shape)
plt.figure(figsize=(8,8))
plt.scatter(X2D[:, 0], X2D[:, 1])
#plt.show()
'''

execution_arr = []
rock_auc_arr = []
#explained_var_ratio_arr = []
#explained_var_ratio_sum_arr = []

for dem_level in range(2, x_train.shape[1] + 1):
    rock_auc, execution_time, explained_var_ratio, explained_var_ratio_sum  = logistic_regression_cv_pca(dem_level, x_train, y_train, x_test, y_test)
    execution_arr.append(execution_time)
    rock_auc_arr.append(rock_auc)
    print(str(dem_level) + "  -  rock_auc - ",  rock_auc, 'execution rate - ', execution_time)


fig, ax1 = plt.subplots()

# Plot the first dataset on the primary Y axis
ax1.plot(range(2, x_train.shape[1] + 1), execution_arr, color='tab:blue', label='Execution time')
ax1.set_xlabel('Number of components')
ax1.set_ylabel('Execution time', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a secondary Y axis
ax2 = ax1.twinx()

# Plot the second dataset on the secondary Y axis
ax2.plot(range(2, x_train.shape[1] + 1), rock_auc_arr, color='tab:orange', label='roc_auc_score')
ax2.set_ylabel('roc_auc_score', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Combine the legends for both plots
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title("Dual-Axis Plot with Different Y Scales")
plt.savefig("roc-time"+str(time.time())+".png")

print(explained_var_ratio, explained_var_ratio_sum)

pca_mnist = PCA()  # all components
X_transformed = pca_mnist.fit_transform(x_train)
cumsum_mnist = np.cumsum(pca_mnist.explained_variance_ratio_)
d = np.argmax(cumsum_mnist >= 0.90) + 1
plt.figure(figsize=(8,8))
plt.plot(cumsum_mnist)
plt.plot([d, d], [0, 1])
plt.savefig('cumsum_mnist.png')


x_new = PCA(n_components=0.9).fit_transform(x_train)
y = y_train
print(x_new.shape)

plt.figure(figsize=(8,8))
plt.scatter(x_new[:, 0], x_new[:, 1], c=y)
plt.savefig('dots'+str(x_new.shape[1])+'.png')



