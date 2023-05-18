#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import datasets

class SupervisedLearning(object):

    def __init__(self) -> None:
        self.epsilon = [0.01, 0.05, 0.1]
        self.delta = self.epsilon #[0.05, 0.1, 0.15]

    def vc_dimension(self, x, alg, degree = 1):
        [n, f] = x.shape
        if alg == 'linear':
            vc_dim = f+1
        elif alg == 'svm_linear':
            vc_dim = f+1
        elif alg == 'svm_poli':
            vc_dim = degree+f-1
        return vc_dim

    def optimal_training_set(self, epsilon, delta, vc_dim):
        n = (1/epsilon)*(np.log(vc_dim)+np.log(1/delta))
        return n

    def optimal_training_set_tree(self, epsilon, delta, depth, m):
        n = (np.log(2)/(2*epsilon**2))*((2**depth-1)*(1+np.log2(m))+1+np.log(1/delta))
        return n

    def linear_regression(self, x, y):
        vc_dim = self.vc_dimension(x, 'linear')
        N = []
        Y = []
        SCORE = []
        for i in range(len(self.epsilon)):
            epsilon = self.epsilon[i]
            delta = self.delta[i]
            n = self.optimal_training_set(epsilon, delta, vc_dim)
            N.append(n)
            new_x = x[0:int(n),:]
            new_y = y[0:int(n),:]
            model = LinearRegression().fit(new_x, new_y)
            y_pred = model.predict(new_x)
            Y.append(y_pred)
            score = model.score(new_x, new_y)
            SCORE.append(score)
        return N, Y, SCORE

    def decision_tree(self, x):
        # vc_dim = self.vc_dimension(x, 'tree')
        depth = 0
        m = 0
        n = self.optimal_training_set_tree(self.epsilon[0], self.delta[0], depth, m)
        return n

    def svm_linear_kernel(self, x, y):
        vc_dim = self.vc_dimension(x, 'svm_linear')
        N = []
        Y = []
        SCORE = []
        for i in range(len(self.epsilon)):
            epsilon = self.epsilon[i]
            delta = self.delta[i]
            n = self.optimal_training_set(epsilon, delta, vc_dim)
            N.append(n)
            new_x = x[0:int(n),:]
            new_y = y[0:int(n),:]
            model = SVC(kernel='linear').fit(new_x, new_y)
            y_pred = model.predict(new_x)
            Y.append(y_pred)
            score = model.score(new_x, new_y)
            SCORE.append(score)
        return N, Y, SCORE

    def svm_polinomical_kernel(self, x, y):
        vc_dim = self.vc_dimension(x, 'svm_poli', degree=3)
        N = []
        Y = []
        SCORE = []
        for i in range(len(self.epsilon)):
            epsilon = self.epsilon[i]
            delta = self.delta[i]
            n = self.optimal_training_set(epsilon, delta, vc_dim)
            N.append(n)
            new_x = x[0:int(n),:]
            new_y = y[0:int(n),:]
            model = SVC(kernel='polynomial').fit(new_x, new_y)
            y_pred = model.predict(new_x)
            Y.append(y_pred)
            score = model.score(new_x, new_y)
            SCORE.append(score)
        return N, Y, SCORE

    def svm_radial_base_kernel(self):
        vc_dim = self.vc_dimension(x, 'svm_radial')
        n = self.optimal_training_set(self.epsilon[0], self.delta[0], vc_dim)
        return n

    def main(self, x, y):
        n_linear, y_linear, score_linear = self.linear_regression(x, y)
        n_svm_linear, y_svm_linear, score_svm_linear = self.svm_linear_kernel(x, y)
        n_svm_poli, y_svm_poli, score_svm_poli = self.svm_polinomical_kernel(x, y)
        return n_linear, y_linear, score_linear, n_svm_linear, y_svm_linear, score_svm_linear, n_svm_poli, y_svm_poli, score_svm_poli

#%%

def read_iris_dataset():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

def read_rice_dataset():
    df = pd.read_excel(r'.\Rice_Cammeo_Osmancik.xlsx')
    df.pop('Class')
    df_normalized=(df-df.min())/(df.max()-df.min())
    x1 = df_normalized["Area"]
    x2 = df_normalized["Perimeter"]
    x3 = df_normalized["Major_Axis_Length"]
    x4 = df_normalized["Minor_Axis_Length"]
    x5 = df_normalized["Eccentricity"]
    x6 = df_normalized["Convex_Area"]
    x7 = df_normalized["Extent"]
    x = np.transpose(np.array([x1,x2,x3,x4,x5,x6,x7]))
    y1 = df_normalized["Class Cameo"]
    y2 = df_normalized["Class Osmanick"]
    y = np.transpose(np.array([y1,y2]))
    return x, y

if __name__ == '__main__':
    x, y = read_rice_dataset()
    sl = SupervisedLearning()
    n_linear, y_linear, score_linear, n_svm_linear, y_svm_linear, score_svm_linear, n_svm_poli, y_svm_poli, score_svm_poli = sl.main(x, y)
# %%
