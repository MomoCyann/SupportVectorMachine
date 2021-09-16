from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

g = 10
hx = []
y = []
a = []
b = 0
c = 1

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
    # # 给x第一列加一列1，常数项
    # x_one = np.ones([len(x)])
    # x = np.insert(x,0,values=x_one,axis=1)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
    return x_train,x_test,y_train,y_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fx(x):
    return np.dot(theta,x_train[x])+b

def cal_hx_y():
    hx.clear()
    y.clear()
    for i in range(len(y_train)):
        hx.append(sigmoid(np.dot(theta,x_train[i])))
        if hx[i] >= 0.5:
            y.append(1)
        else:
            y.append(-1)
    return hx,y


def cal_loss_of_kkt():
    '''
    loss是衡量差异的量，差异最大的就是最违反KKT条件的a，优先被smo算法选中
    c是惩罚系数
    '''
    loss = np.ones([len(x_train),3])
    for i in range(len(y_train)):
        for j in range(len(loss[i])):
            loss[i][j] = y_train[i] * fx(i) - 1
            if j == 0:
                if (a[i]>0 and loss[i][j]<=0) or (a[i]==0 and loss[i][j]>0):
                    loss[i][j] = 0
            if j == 1:
                if ((a[i]==0 or a[i]==c) and loss[i][j]!=0) or (0<a[i]<c and loss[i][j]==0):
                    loss[i][j] = 0
            if j == 2:
                if (a[i]==c and loss[i][j]<0) or (a[i]<c and loss[i][j]>=0):
                    loss[i][j] = 0
    loss = loss*loss
    loss = np.sum(loss,axis=1)
    return loss


def cal_gram_matrix():
    gram = np.zeros([len(x_train),len(x_train)])
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            gram[i][j] = np.dot(x_train[i],x_train[j])
    return gram


def update_a():
    m = np.argmax(loss)
    scd = random.randint(len(y_train))
    if scd == m:
        scd = random.randint(len(y_train))
    # save the origin 'a' for a while
    a1old = a[m]
    a2old = a[scd]
    a[m] = a[m] - y_train[m] * (fx(m)-y_train[m]-fx(scd)+y_train[scd]) \
                / (gram[m][m]-2*gram[m][scd]+gram[scd][scd])
    # define the max and the min of 'a'
    if y_train[m] == -1:
        




if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    theta = np.zeros([len(x_train[0])])
    a = np.zeros([len(x_train)])
    loss = cal_loss_of_kkt()
    max = np.argmax(loss)
    gram = cal_gram_matrix()


