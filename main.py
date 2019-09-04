

import pickle
import gzip
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation
from PIL import Image
import os
import numpy as np



#Confusion matrix Calculation. Using the ready made SK Learn Library for creating confusion matrix.
def confusion_matrix(actual_output, model_output):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(actual_output, model_output)




#This function is for pre processing of USPS data. This will return 2 matrix. One will be Testing Input and other
#will be Testing output labels which will ahve values 0-9
def getUSPSData():
    USPSMat  = []
    USPSTar  = []
    curPath  = 'USPSdata/Numerals'
    savedImg = []

    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                savedImg = img
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    print(len(USPSMat))
    print(len(USPSTar))
    print(len(USPSMat[1]))
    print(USPSTar[15000])
    return USPSMat,USPSTar
    


#This is good old Neural Network method. It will have 3 layers with Soft max in output layers
#Softmax is used as it is multi-class(10) classification. Rest is pretty simple configuration


def runNeuralNetwork(inputDataSet,outPutDataSet,inputDimension,val_input,val_label,test_input,test_label,input_usps,output_usps):
    model = Sequential()
    model.add(Dense(300, input_dim=inputDimension))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    # Fit the model
    model.fit(inputDataSet, outPutDataSet, epochs=10, batch_size=4000)
    print("Model is fitt")
    # calculate predictions
    val_loss, val_acc = model.evaluate(val_input, val_label)
	
   
    print(' Validation accuracy:', val_acc)
    mnist_pred = model.predict(test_input)
    test_loss,test_acc=model.evaluate(test_input,test_label)
    conf_matrix = confusion_matrix(test_acc, test_label)
    print('Test Accuracy::',test_acc)
    test_loss_usps,test_acc_usps=model.evaluate(input_usps,output_usps)
    usps_pred = model.predict(input_usps)
    print('USPS Data Accuracy Is ::---->>>',test_acc_usps)
    return mnist_pred, usps_pred


# In[40]:


#SVM with RBF function as kernel and Gamma as Default. Have taken C=2 which means 2 class cost function.
def SVMRadialGammaDefault(InputTraining,OutPutTraining,InputValid,OutputValid,InputTest,OutPutTest,InputUSPS,OutputUSPS):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import fetch_mldata
    print('Inside SVM RBF')
    classifierGammaDef = SVC(kernel='rbf', C=2);
    print('Classifier ready RBF')
    classifierGammaDef.fit(InputTraining, OutPutTraining)
    print('Training on MNIST Training Data done RBF')
    
    #Validation dataset of MNIST Prediction
    print('---Starting MNIST Validation Data Prediction RBF---')
    valid_out=classifierGammaDef.predict(InputValid)
    print('---Finished MNIST Validation Data Prediction RBF---')
    print('Accuracy of MNIST Validation Data Set RBF:')
    print((valid_out==OutputValid).mean())
    
    print('---Starting MNIST Test Data Prediction RBF---')
    test_out=classifierGammaDef.predict(InputTest)
    print('---Finished MNIST Test Data Prediction RBF--- ')
    
    print('Accuracy of MNIST Test Data Set RBF:')
    print((test_out==OutPutTest).mean())
    conf_matrix = confusion_matrix(test_out, OutPutTest)
    
    #USPS Fitting
    
    print('---Starting USPS Test Data Prediction in RBF---')
    TEST_USPS=classifierGammaDef.predict(InputUSPS)
    print('---Finished USPS Test Data Prediction RBF--- ')
    
    print('Accuracy of USPS Test Data Set RBF:')
    print((TEST_USPS==OutputUSPS).mean())
    return test_out, TEST_USPS
    #Random Forest Start


# In[41]:

#SVM with RBF function as kernel and Gamma as One and Cose Class=2.
def SVMRadialGammaOne(InputTraining,OutPutTraining,InputValid,OutputValid,InputTest,OutPutTest,InputUSPS,OutputUSPS):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import fetch_mldata
    print('Inside SVM RBF')
    classifierGammaOne = SVC(kernel='rbf',C=2,gamma=1);
    print('Classifier ready RBF Gamma One')
    classifierGammaOne.fit(InputTraining, OutPutTraining)
    print('Training on MNIST Training Data done RBF Gamma One')
    
    #Validation dataset of MNIST Prediction
    print('---Starting MNIST Validation Data Prediction RBF Gamma One---')
    valid_out=classifierGammaOne.predict(InputValid)
    print('---Finished MNIST Validation Data Prediction RBF Gamma One---')
    print('Accuracy of MNIST Validation Data Set RBF Gamma One:')
    print((valid_out==OutputValid).mean())
    
    print('---Starting MNIST Test Data Prediction RBF Gamma One---')
    test_out=classifierGammaOne.predict(InputTest)
    print('---Finished MNIST Test Data Prediction RBF Gamma One--- ')
    
    print('Accuracy of MNIST Test Data Set RBF Gamma One:')
    print((test_out==OutPutTest).mean())
    conf_matrix = confusion_matrix(test_out, OutPutTest)
    
    #USPS Fitting
    
    print('---Starting USPS Test Data Prediction in RBF Gamma One---')
    TEST_USPS=classifierGammaOne.predict(InputUSPS)
    print('---Finished USPS Test Data Prediction RBF Gamma One--- ')
    
    print('Accuracy of USPS Test Data Set RBF Gamma One:')
    print((TEST_USPS==OutputUSPS).mean())
    return test_out, TEST_USPS


# In[42]:

#SVM with Kernel as Linear.
def SVMLinear(InputTraining,OutPutTraining,InputValid,OutputValid,InputTest,OutPutTest,InputUSPS,OutputUSPS):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import fetch_mldata
    print('Inside SVM Linear')
    classifierLinear = SVC(kernel='linear');
    print('Classifier ready Linear')
    classifierLinear.fit(InputTraining, OutPutTraining)
    print('Training on MNIST Training Data done Linear')
    
    #Validation dataset of MNIST Prediction
    print('---Starting MNIST Validation Data Prediction Linear---')
    valid_out=classifierLinear.predict(InputValid)
    print('---Finished MNIST Validation Data Prediction Linear---')
    print('Accuracy of MNIST Validation Data Set Linear:')
    print((valid_out==OutputValid).mean())
    
    print('---Starting MNIST Test Data Prediction Linear---')
    test_out=classifierLinear.predict(InputTest)
    print('---Finished MNIST Test Data Prediction Linear--- ')
    
    print('Accuracy of MNIST Test Data Set Linear:')
    print((test_out==OutPutTest).mean())
    conf_matrix = confusion_matrix(test_out, OutPutTest)
    
    #USPS Fitting
    
    print('---Starting USPS Test Data Prediction in Linear---')
    TEST_USPS=classifierLinear.predict(InputUSPS)
    print('---Finished USPS Test Data Prediction Linear--- ')
    
    print('Accuracy of USPS Test Data Set Linear:')
    print((TEST_USPS==OutputUSPS).mean())
    return test_out, TEST_USPS


# In[43]:

#Random Forest classifier using Sk-learns library. Training is done on MNIST, then validaion is done, followed by testing on MNIST and testing on USPS
def RandomForest(InputTraining,OutPutTraining,InputValid,OutputValid,InputTest,OutPutTest,InputUSPS,OutputUSPS):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import fetch_mldata
    print('Inside SVM Linear')
    randomForest = RandomForestClassifier(n_estimators=10)
    print('Classifier ready Linear')
    randomForest.fit(InputTraining, OutPutTraining)
    print('Training on MNIST Training Data done Linear')
    
    #Validation dataset of MNIST Prediction
    print('---Starting MNIST Validation Data Prediction Linear---')
    valid_out=randomForest.predict(InputValid)
    print('---Finished MNIST Validation Data Prediction Linear---')
    print('Accuracy of MNIST Validation Data Set Linear:')
    print((valid_out==OutputValid).mean())
    
    print('---Starting MNIST Test Data Prediction Linear---')
    test_out=randomForest.predict(InputTest)
    print('---Finished MNIST Test Data Prediction Linear--- ')
    
    print('Accuracy of MNIST Test Data Set Linear:')
    print((test_out==OutPutTest).mean())
    conf_matrix = confusion_matrix(test_out, OutPutTest)
    
    #USPS Fitting
    
    print('---Starting USPS Test Data Prediction in Linear---')
    TEST_USPS=randomForest.predict(InputUSPS)
    print('---Finished USPS Test Data Prediction Linear--- ')
    
    print('Accuracy of USPS Test Data Set Linear:')
    print((TEST_USPS==OutputUSPS).mean())
    return test_out, TEST_USPS


# In[44]:

#This function will simply change the single valued output to one-hot representaion which is like 10 bit representation
def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T




#Get USPS Data
#We are simpley getting USPS data and converting output into hot vector rep.

input_usps,output_usps=getUSPSData()
input_usps=np.array(input_usps)
output_usps_hot=_change_ont_hot_label(np.array(output_usps))


# In[ ]:





# In[ ]:



filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
x_training_input,y_training_label=training_data
x_validation_input,y_validation_label=validation_data
print(x_training_input.shape)
print(y_training_label.shape)
y_train_hot=_change_ont_hot_label(y_training_label)
y_valid_hot=_change_ont_hot_label(y_validation_label)
print(y_train_hot.shape)

#Testing Data
x_test_input,y_test_label=test_data
y_test_hot=_change_ont_hot_label(y_test_label)



# Logistic
import scipy.sparse

y = y_training_label
x = x_training_input
testY = y_test_label
testX = x_test_input


#this is decrementing or updating the weight in Logistic regresion at each step. It is simple and beased on usual formula
def updateWeight(w,x,y,lmbda):
    m = x.shape[0] #First we get the number of training examples
    y_mat = convertToHotVector(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lmbda/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lmbda*w #And compute the gradient for that loss
    return loss,grad

#This is to convert from single values representation to Hot Vector
def convertToHotVector(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

#This method does the actual prediction of the model
def doPrediction(inputVector):
    singleprob = softmax(np.dot(inputVector,w))
    modelPrediction = np.argmax(singleprob,axis=1)
    return singleprob,modelPrediction


w = np.zeros([x.shape[1],len(np.unique(y))])
lmbda = 1
iterations = 2000
learningRate = 0.05
losses = []
for i in range(0,iterations):
    loss,grad = updateWeight(w,x,y,lmbda)
    losses.append(loss)
    w = w - (learningRate * grad)
print(loss)


# This uses the above method to do the proper prediction, get the accuracy by comapring it with actual model target
def outputAccuracy(inputVector,outputActual):
    prob,prede = doPrediction(inputVector)
    print(prede)
    accuracy = sum(prede == outputActual)/(float(len(outputActual)))
    return accuracy,prede

train_acc,pred_out = outputAccuracy(x,y)
test_acc,test_logistic_prediction = outputAccuracy(testX,testY)
print('Training Accuracy: ', train_acc)
print('Test Accuracy: ', test_acc)
print(confusion_matrix(testY, test_logistic_prediction))
#Logisitc is done at this point






#Here i am storing value of prediction from Neural Network of MNIST and USPS in 2 variables.
nn_mnist, nn_usps=runNeuralNetwork(x_training_input,y_train_hot,784,x_validation_input,y_valid_hot,x_test_input,y_test_hot,input_usps,output_usps_hot)
print('Calling SVM Radial Basis With Gamma Defualt')
SVMRadialGammaDefault(x_training_input,y_training_label,x_validation_input,y_validation_label,x_test_input,y_test_label,input_usps,np.array(output_usps))


print('Calling SVM Radial Basis With Gamma One')
SVMRadialGammaOne(x_training_input,y_training_label,x_validation_input,y_validation_label,x_test_input,y_test_label,input_usps,np.array(output_usps))

print('Calling SVM Linear Method')
SVMLinear(x_training_input,y_training_label,x_validation_input,y_validation_label,x_test_input,y_test_label,input_usps,np.array(output_usps))


f.close()



usps_acc,usps_pred_out = outputAccuracy(input_usps,np.array(output_usps))

print('USPS Accuracy: ', usps_acc)
print(confusion_matrix(output_usps, usps_pred_out))


# In[ ]:


print('Calling Random Forest Method')
predicted_randomForest_MNIST, predicted_randomForest_USPS = RandomForest(x_training_input,y_training_label,x_validation_input,y_validation_label,x_test_input,y_test_label,input_usps,np.array(output_usps))


# In[ ]:


nn_mnist, nn_usps = runNeuralNetwork(x_training_input,y_train_hot,784,x_validation_input,y_valid_hot,x_test_input,y_test_hot,input_usps,output_usps_hot)


#Ensemble Method begins
#Here i am taking Logistic regression, Random Forest and Neural NEtwork, Then doing a majority voting in all three and giving final output


neural_output = np.argmax(nn_mnist, axis=1)
#We are considering 10000 rows only for sake of simplicity and taking 3 models. I tried with 20000 and 4 models too. It takes very long to run so here i am omitting them
#However, we can include all of them by simple one or two lines of code.

majority_voting_data =  np.zeros((10000, 3))
majority_voting_data = np.column_stack((test_logistic_prediction,predicted_randomForest_MNIST, neural_output))
maj_array = []
from collections import Counter
for i in range(0,10000):
    b = Counter(majority_voting_data[i])

    counts = np.bincount(majority_voting_data[i])
    maj_array.append(np.argmax(counts))
    
print(majority_voting_data.shape)
print((np.array(maj_array)==y_test_label).mean())




#this is to print confusion matrix of USPS and random forest
print(confusion_matrix(output_usps, predicted_randomForest_USPS))
print(confusion_matrix(testY, predicted_randomForest_MNIST))






