import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction import DictVectorizer
from itertools import chain

'''originally runs in jupyter'''

file_train = 'sampled_users_train2'
file_test = 'sampled_users_test2'

# read training file
df_train = pd.read_json(file_train, lines=True)

# initialize all feature in dict
## format all feats in 1d
feats_list = list(chain.from_iterable(df_train['feats'].values))
print "feature list reshape done"

''' 
[{u'feature_id': u'yct:001000295',
  u'feature_type': u'YCT',
  u'feature_value': 1.0},
 {u'feature_id': u'PopSugar',
  u'feature_type': u'WIKI',
  u'feature_value': 0.5809999704360961},
  ...
]
'''
## save features in init_theta uniquely
init_theta = {}
for f in feats_list:
    init_theta[f['feature_id']] = np.random.normal(0,1)
print "feature process done"
print "num of feature :", len(init_theta) # 31151 features

###
data = df_train[['action','feats']] # 493880 rows × 2 columns(action feats)
data.head(3)

###
def avg_square_loss(h_vec, y_vec):
    return np.sum((h_vec - y_vec)**2) / len(y_vec)

# features are saved in dictionary
def evaluate_gradient(loss, data, theta):
        X = data.values[:,1:]
        y = data.values[:,0] # format [0 0 1 0 0]
        h_vec = []           # format [-1.3, -0.2, -1.1, -3.1, -4.2]
        X = list(chain.from_iterable(X))
        for x in X:
            h = 0
            for feat in x:
                h += theta[feat['feature_id']] * feat['feature_value']
            h_vec.append(h)
            
        avg_loss = loss(h_vec, y)
        
        # calculate gradient of each feature
        gradient_dict = {}   # format {'a':1,'b':'1'}
        gradient_dict_count = {} # record the times each feature used
        for i in range(len(y)):
            x_cur = X[i]
            y_cur = y[i]
            h = h_vec[i]
            for feat in x_cur:
                if feat['feature_id'] not in gradient_dict:
                    gradient_dict[feat['feature_id']] = 0
                    gradient_dict_count[feat['feature_id']] = 0
                gradient_dict[feat['feature_id']] += feat['feature_value'] * (h - y_cur) # x['a'](h-y)
                gradient_dict_count[feat['feature_id']] += 1
              
        for key in gradient_dict:
            gradient_dict[key] = gradient_dict[key] * 2.0 / gradient_dict_count[key]
        
        return dict(gradient_dict), avg_loss
###
def gradient_descent(data, theta, alpha=0.001, epoch=100):
    for i in range(0, epoch):
        data = data.sample(frac=1)
        gradient_dict, avg_loss = evaluate_gradient(avg_square_loss, data, dict(theta))
        
        # update theta
        for gradient_key in gradient_dict:
            theta[gradient_key] -= alpha *  gradient_dict[gradient_key]

        if i % 1 == 0:
            print('epoch %d \t avg loss : %f' % (i,avg_loss))
    return dict(theta)

def stochastic_gradient_descent(data, theta, alpha=0.0001, epoch=100):
    m,n = data.values[:,1:].shape
    for i in range(0, epoch):
        data = data.sample(frac=1)
        for it in range(m):
            example = data.iloc[[i]]
            gradient_dict, avg_loss = evaluate_gradient(avg_square_loss, example, dict(theta))
            # update theta
            for gradient_key in gradient_dict:
                theta[gradient_key] -= alpha * gradient_dict[gradient_key]
            if it % 10000 == 0:
                print('epoch %d \t it %d \t avg loss : %s' % (i,it,avg_loss))
        print('epoch %d \t avg loss : %s' % (i,avg_loss))

    return dict(theta)
    
def mini_gradient_descent(data, theta, batch_size=100, alpha=0.0001, epoch=100):
    m,n = data.values[:,1:].shape
    for i in range(0, epoch):
        for it in range(int(m / batch_size)):
            batch = data.sample(n=batch_size)
            gradient_dict, avg_loss = evaluate_gradient(avg_square_loss, batch, dict(theta))
            # update theta
            for gradient_key in gradient_dict:
                theta[gradient_key] -= alpha * gradient_dict[gradient_key]
        if i % 10 == 0:
            print('epoch %d \t avg loss : %f' % (i,avg_loss))
    return dict(theta)
    
###
def predict(theta, data):
    X = data.values[:,1:]
    y = data.values[:,0] # format [0 0 1 0 0]
    h_vec = [] # format [-1.3, -0.2, -1.1, -3.1, -4.2]
    X = list(chain.from_iterable(X))
    for x in X:
        h = 0
        for feat in x:
            if feat['feature_id'] in theta:
                h += theta[feat['feature_id']] * feat['feature_value']
        h = 1 if h > 0.5 else 0
        h_vec.append(h)
    return h_vec

def evaluate(pred_vec, true_vec):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(pred_vec)):
        if pred_vec[i] == 1 and true_vec[i] == 1:
            TP += 1
        elif pred_vec[i] == 0 and true_vec[i] == 1:
            FN += 1
        elif pred_vec[i] == 1 and true_vec[i] == 0:
            FP += 1
        else:
            TN += 1
    print("recall %f \t precision %f \t accuracy %f " % (TP*1.0/(TP+FN), TP*1.0/(TP+FP), (TP+TN)*1.0/(TP+TN+FP+FN)))

###
theta_batch = gradient_descent(data, dict(init_theta), 0.01, 100)
theta_stoch = stochastic_gradient_descent(data, dict(init_theta), 0.00001, 1)
theta_mini = mini_gradient_descent(data, dict(init_theta), 1000, alpha=0.001, epoch=100)

###
# read test file
df_test = pd.read_json(file_test, lines=True)
data_test = df_test[['action','feats']] # 247243 rows × 2 columns(action feats)

###
# gradient_descent
pred_vec_batch = predict(theta_batch, data_test)
evaluate(pred_vec_batch, data_test.values[:,0])# batch

# stochastic_gradient_descent
pred_vec_stoch = predict(theta_stoch, data_test)
evaluate(pred_vec_stoch, data_test.values[:,0])# stochastic

# mini_gradient_descent
pred_vec_mini = predict(theta_mini, data_test)
evaluate(pred_vec_mini, data_test.values[:,0])# mini
