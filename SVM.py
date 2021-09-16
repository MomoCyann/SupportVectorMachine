from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import random

g = 100
hx = []
y = []
b = 0 # wx+b的b
c = 1 # 惩罚系数

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
    sum = 0
    for i in range(len(a)):

    return

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


def init_a():
    a = np.empty([len(x_train)])
    for i in range(len(x_train)):
        a[i] = random.uniform(0,c)
    return a


def cal_loss_of_kkt():
    '''
    loss是衡量差异的量，差异最大的就是最违反KKT条件的a，优先被smo算法选中
    c是惩罚系数
    '''
    loss_temp = np.ones([len(x_train),3])
    for i in range(len(y_train)):
        for j in range(len(loss_temp[i])):
            loss_temp[i][j] = y_train[i] * fx(i) - 1
            if j == 0:
                if (a[i]>0 and loss_temp[i][j]<=0) or (a[i]==0 and loss_temp[i][j]>0):
                    loss_temp[i][j] = 0
            if j == 1:
                if ((a[i]==0 or a[i]==c) and loss_temp[i][j]!=0) or (0<a[i]<c and loss_temp[i][j]==0):
                    loss_temp[i][j] = 0
            if j == 2:
                if (a[i]==c and loss_temp[i][j]<0) or (a[i]<c and loss_temp[i][j]>=0):
                    loss_temp[i][j] = 0
    loss_temp = loss_temp*loss_temp
    loss_temp = np.sum(loss_temp,axis=1)
    return loss_temp


def cal_gram_matrix():
    gram = np.zeros([len(x_train),len(x_train)])
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            gram[i][j] = np.dot(x_train[i],x_train[j])
    return gram


def update_a():
    m = np.argmax(loss)
    scd = np.random.randint(len(y_train))
    if scd == m:
        scd = np.random.randint(len(y_train))
    # save the origin 'a' for a while
    a1old = a[m]
    a2old = a[scd]
    a[m] = a[m] - y_train[m] * (fx(m)-y_train[m]-fx(scd)+y_train[scd]) / (gram[m][m]-2*gram[m][scd]+gram[scd][scd])
    # define the max and the min of 'a'
    if y_train[m] == -1:
        u = max(0, a1old - a2old)  # u为下界，v为上界
        v = min(c, c+a1old-a2old)
    if y_train[m] == 1:
        u = max(0, a1old+a2old-c)
        v = max(c, a1old+a2old)
    # clip the 'a'
    a[m] = np.clip(a[m],u,v)
    a[scd] = a2old + y_train[m] * y_train[scd] * (a1old-a[m])
    return a


def cal_theta():
    for i in range(len(a)):
        for j in range(len(x_train[0])):
            theta[j] += a[i] * y_train[i] * x_train[i][j]
    return theta


def cal_b():
    index1 = np.argwhere(y_train == -1).flatten()
    index2 = np.argwhere(y_train == 1).flatten()
    save1=[]
    save2=[]
    for i in range(len(index1)):
        save1.append(np.dot(theta,x_train[index1[i]]))
    for i in range(len(index2)):
        save2.append(np.dot(theta,x_train[index2[i]]))
    return -(max(save1)+min(save2))/2


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    theta = np.zeros([len(x_train[0])])
    gram = cal_gram_matrix()
    a = init_a()
    for i in range(g):
        loss = cal_loss_of_kkt()
        a = update_a()

    theta = cal_theta()
    b = cal_b()
    print(theta)
    print(b)



