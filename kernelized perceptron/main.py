import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUMBER_OF_ITERATION = 15

def get_labels(k):
    label = ['label']
    for i in range(1, k + 1):
        label.append(str(i))
    return label


def add_bias(data):
    data['785'] = np.array([1] * data.shape[0])
    return data


def kp(x, y, p=1):
    value = 1 + np.matmul(x, np.transpose(y))
    value = np.power(value, p)
    return value


def get_Y_and_X(data):
    y = data['label'].apply(lambda x: 1 if x == 3 else -1)
    x = data.drop(['label'], 1)
    return x, y

############################################ Online perceptron ########################################################

def online_train():
    train_df,Y = get_Y_and_X(add_bias(pd.read_csv('pa2_train.csv',names=get_labels(784))))
    valid_df, VY = get_Y_and_X(add_bias(pd.read_csv('pa2_valid.csv',names=get_labels(784))))

    (n, features) = train_df.shape

    (vn, vfeatures) = valid_df.shape

    W = [0 for x in range(0, features)]
    W = np.array(W)
    iters = 15; _iter = 0;
    training_error = []
    validation_error = []
    train_accuracy = []
    validation_accuracy = []
    weightsMap = {}
    maxAccuracy = 0
    maxAccuracyIndex = 0
    while _iter < iters:
        error = 0;
        v_error = 0;
        for i in range(0, n):
            x = train_df.iloc[i]
            u = W.dot(x)
            yi = Y.iloc[i]
            if yi*u <= 0:
                W = np.add(W,np.multiply(yi,x))
        for i in range(0, n):
            x = train_df.iloc[i]
            u = W.dot(x)
            yi = Y.iloc[i]
            if yi*u <= 0:
                error+=1 
        weightsMap[_iter] = W
        training_error.append(error)
        t_accuracy = 1-(error/n);
        train_accuracy.append(t_accuracy)
        for i in range(0, vn):
            vx = valid_df.iloc[i]
            vu = np.array([W]).dot(vx)
            vyi = VY.iloc[i]
            if vyi*vu <= 0:
                v_error += 1;
        validation_error.append(v_error)
        v_accuracy = 1-(v_error/vn);
        validation_accuracy.append(v_accuracy)
        _iter+=1
    return weightsMap

    # t, = plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], train_accuracy, label="train")
    # v, = plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], validation_accuracy, label="validation")
    # plt.xlabel('iterations')
    # plt.ylabel('accuracy')
    # plt.legend(handles=[t,v], loc='best')
    print(validation_accuracy)

def online_predict(weightsMap):
    accurate_weights=weightsMap[13]
    predict_df = add_bias(pd.read_csv('pa2_test_no_label.csv',names=get_labels(784)[1:]))
    a = (np.array([accurate_weights]))
    y_ = a.dot(np.transpose(predict_df))
    op = pd.DataFrame(np.sign(y_)).T
    op.to_csv("oplabel.csv", index=False, header=False)

#-----------------------------------------------------------------------------

############################################## Average Perceptron ###############################################

def average_perceptron():
    train_df, Y = get_Y_and_X(add_bias(pd.read_csv('pa2_train.csv', names=get_labels(784))))
    valid_df, val_y = get_Y_and_X(add_bias(pd.read_csv('pa2_valid.csv', names=get_labels(784))))

    (n, features) = train_df.shape
    (val_n, val_features) = valid_df.shape
    a_weightsMap = {}
    W = np.array(np.zeros(features))
    w_hat = np.zeros(features)
    c = 0
    s = 0
    _iter = 0;
    training_accuracy = []
    validation_accuracy = []

    def error_calc(X, weight, Y):
        error = 0
        for i in range(X.shape[0]):
            x = X.iloc[i]
            u = np.sign(np.dot(np.transpose(weight), x))
            y = Y.iloc[i]
            if y * u <= 0:
                error += 1

        return error

    while _iter < NUMBER_OF_ITERATION:
        error = 0
        val_error = 0
        for i in range(n):
            x = train_df.iloc[i]
            # print(x.shape,W.shape)
            u = np.sign(np.dot(np.transpose(W), x))

            yi = Y.iloc[i]
            # print(u,yi)
            if yi * u <= 0:
                if s + c > 0:
                    w_hat = np.add(np.multiply(s, w_hat), np.multiply(c, W)) / np.add(s, c)
                s += c
                W = np.add(W, np.multiply(yi, x))
                c = 0
                error += 1
            else:
                c += 1
        if c > 0:
            w_hat = np.add(np.multiply(s, w_hat), np.multiply(c, W)) / np.add(s, c)
        #weightsMap[_iter] = W
        training_accuracy.append(1 - (error_calc(train_df, w_hat, Y) / n))
        validation_accuracy.append(1 - (error_calc(valid_df, w_hat, val_y) / val_n))
        _iter += 1

    print("Training accuracy", training_accuracy)
    print("Validation accuracy", validation_accuracy)

    print("Maximum train accuracy:", max(training_accuracy))
    print("Maximum validation Accuracy:", max(validation_accuracy))
    #plt.figure(figsize=(18, 6))
    #t, = plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], training_accuracy, label="Training")
    #v, = plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], validation_accuracy, label="Validation")
    #plt.xlabel('Number of Iterations')
    #plt.ylabel('Accuracy')
    #plt.title('Average Perceptron')
    #plt.legend(handles=[t, v], loc='best')
    #plt.show()

#-----------------------------------------------------------------------------------------------------------------------------

###################################### Kernel Perceptron ######################################################

def get_accuracy(v,ind):
    l = []
    for i in range(1,NUMBER_OF_ITERATION+1):
        l.append(v[i][ind])
    #print("l is",l)
    return l

def calc_accuracy(Y,alpha,kernel,Y_c):
    error = 0
    for i in range(Y_c.shape[0]):
            k = kernel[:,i]
            w = np.multiply(k,alpha)
            w = np.multiply(w,Y)
            u = np.sign(sum(list(w)))
            if u*Y_c[i]<=0:
                error = error + 1
                
    error = error/Y_c.shape[0] 
    return (1-error)*100

def do_computation(p,all_info,X_train,Y_train,X_val,Y_val):
    kernel_matrix = kp(X_train,X_train,p)
    new_kernel = kp(X_train,X_val,p)
    alpha = np.zeros(X_train.shape[0])
    it = 1
    while it <= NUMBER_OF_ITERATION:
        #t = 0
        for i in range(X_train.shape[0]):
            k = kernel_matrix[:,i]
            w = np.multiply(k,alpha)
            w = np.multiply(w,Y_train)
            u = sum(list(w))

            if np.sign(u)*Y_train[i] <=0:
                alpha[i] = alpha[i] + 1
        #t = t/X_train.shape[0]       
      
    
#         #print(new_kernel.shape)
#         for i in range(X_val.shape[0]):
#             k = new_kernel[:,i]
#             w = np.multiply(k,alpha)
#             w = np.multiply(w,Y_train)
#             u = np.sign(sum(list(w)))
#             if u*Y_val[i]<=0:
#                 v = v + 1
#         v = v/X_val.shape[0]  
        t = calc_accuracy(Y_train,alpha,kernel_matrix,Y_train)
        v = calc_accuracy(Y_train,alpha,new_kernel,Y_val)
        all_info[p][it] = (t,v,alpha)
        it = it + 1
    print("done for p = ",p)

def calc_best_acc_for_each_p(p_values,all_info):
    best_val_acc = []
    for p in p_values:
        val_acc = [ ]
        for x in all_info[p].keys():
            val_acc.append(all_info[p][x][1])
        best_val_acc.append(max(val_acc))    
    return best_val_acc

def predict_using_kernel(X_train,Y_train,X,best_alpha):
    new_kernel = kp(X_train,X,3	)
    #print(new_kernel.shape)
    ot = []
    for i in range(X.shape[0]):
            k = new_kernel[:,i]
            w = np.multiply(k,best_alpha)
            w = np.multiply(w,Y_train)
            u = np.sign(sum(list(w)))
            ot.append(u)
    return np.array(ot)

def get_best_alpha(p,all_info):
    alpha = all_info[p][1][2]
    i = all_info[p][1][1]
    for y in range(2,16):
        if all_info[p][y][1]>i:
            alpha = all_info[p][1][2]
            i = all_info[p][1][1]
    return alpha

def do_part_3():
	print("Starting for part 3")
	X_train,Y_train = get_Y_and_X(add_bias(pd.read_csv('pa2_train.csv',names=get_labels(784))))
	X_val,Y_val = get_Y_and_X(add_bias(pd.read_csv('pa2_valid.csv',names=get_labels(784))))
	all_info = {}
	p_values = [1,2,3,7,15]

	for p in p_values:
	    all_info[p]={}


	for p in p_values:
		all_info=do_computation(p,all_info,X_train,Y_train,X_val,Y_val)
	x_axis=[i for i in range(1,16)]
	best_acc = calc_best_acc_for_each_p(p_values,all_info)
	X_test = add_bias(pd.read_csv('pa2_test_no_label.csv',names=get_labels(784)[1:]))
	prediction_kernel = predict_using_kernel(X_train,Y_train,X_test,get_best_alpha(3,all_info))
	output = pd.DataFrame(data=prediction_kernel)
	output.to_csv('kplabel.csv',index=False,header=None)
	return all_info

#----------------------------------------------------------------------


#Calling percetron functions

online_predict(online_train()) # calling online perceptron
average_perceptron() # calling average perceptron
all_info_for_3 = do_part_3() # calling kernel peceptron


