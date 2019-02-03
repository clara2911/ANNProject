import numpy as np
from matplotlib import pyplot as plt


class Database:

    def __init__(self):
        self.function = 'Hello, use me for creating your wonderful data!'

    def build_data(self, beta = 0.2, gamma = 0.1, n = 10, tau = 25, npts = 1500):
        t = np.arange(0, npts)
        x = np.zeros(len(t))
        x[0] = 1.5
        for i in range(len(x)-1):
            if t[i] - tau < 0:
                x_tau = 0
            else:
                x_tau = x[t[i]-tau]
            x[i+1] = x[i] + beta*x_tau/(1+x_tau**n) - gamma*x[i]
        #plt.plot(t, x)
        #plt.show()
        # inputs and outputs
        outputs = x.reshape((-1,1))
        inputs = np.zeros((len(outputs),5))
        x = np.hstack((np.zeros(25), x))
        for i in np.arange(0, len(outputs)):
            inputs[i,:] = [x[i], x[i+5], x[i+10], x[i+15], x[i+20]]

        return inputs, outputs

    def Split_data(self, inputs, outputs, train = 600, val = 400):
        index = range(len(outputs))
        #test
        index_test = index[-200:]
        #train
        index_train = range(600)
        #validation
        index_val = np.arange(train, train+val)
        return inputs[index_train,:], outputs[index_train], inputs[index_val,:], \
               outputs[index_val], inputs[index_test,:], outputs[index_test]


if __name__ == '__main__':

    database = Database()
    inputs, outputs = database.build_data()
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = database.Split_data(inputs, outputs, train = 600, val = 400)
    print('hola pianola')


