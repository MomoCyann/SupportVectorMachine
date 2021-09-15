from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

g = 10
hx = []
y = []
loss_min = 99999
loss_save = []
a = []

def load_data():
    # 鸢尾花
    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']
    x = x[y!=2]
    y = y[y!=2]
    y[y==0] = -1
    # 取前两列数据 只分类0或1类鸢尾花 1和2类鸢尾花是线性不可分的。
    # 似乎后两列数据没用，这里没做可视化预处理，一开始训练效果奇差
    x = x[:,:2]
    # 给x第一列加一列1，常数项
    x_one = np.ones([len(x)])
    x = np.insert(x,0,values=x_one,axis=1)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
    return x_train,x_test,y_train,y_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cal_hx_y():
    hx.clear()
    y.clear()
    for i in range(len(y_train)):
        hx.append(sigmoid(np.dot(theta,x_train[i])))
        if hx[i] >= 0.5:
            y.append(1)
        else:
            y.append(0)
    return hx,y


def cal_loss():

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    theta = np.zeros([len(x_train[0])])