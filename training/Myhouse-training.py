
# coding: utf-8

# In[1]:


from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
import csv
import pandas as pd

ctx = mx.cpu()

# Define some variables

# How many samples to process at once when training
batch_size = 64

# How many classes of gestures you want to have (has to correlate with your data, including "no-gesture" class)
num_outputs = 7

# How many features in each data sample (3 for acceleration, 3 for gravity for a total of 6)
num_features_in_sample = 6

# How many readings qualify as one complete gesture
num_datapoints_in_gesture = 40

# How many total readings are in a single gesture
num_readings_in_gesture = num_features_in_sample * num_datapoints_in_gesture


# In[6]:


# Load the training data CSV file
training_data_reader = pd.read_csv('training-data-complete.csv',parse_dates=['time'])
training_data_reader.set_index('time')

# Load the testing data CSV file
testing_data_reader = pd.read_csv('testing-data.csv',parse_dates=['time'])
testing_data_reader.set_index('time')

# Do some data massaging to get it to format that the network needs
training_data_matrix = training_data_reader.as_matrix()
testing_data_matrix = testing_data_reader.as_matrix()

training_data = np.asarray(training_data_matrix[:,0:num_features_in_sample], dtype=np.float64)
testing_data = np.asarray(testing_data_matrix[:,0:num_features_in_sample], dtype=np.float64)

training_data = training_data.reshape(-1, num_datapoints_in_gesture, num_features_in_sample)
testing_data = testing_data.reshape(-1, num_datapoints_in_gesture, num_features_in_sample)

training_labels = training_data_matrix[:,num_features_in_sample + 1]
testing_labels = testing_data_matrix[:,num_features_in_sample + 1]

# To prepare training and testing labels, we cut off the dash and the digits at the end of the label name, so "fan-01" becomes "fan"
training_labels = [x[0:-3] for x in training_labels]
testing_labels = [x[0:-3] for x in testing_labels]

class_dict = {
    'tv-poke': 1, 
    'fan-one-circle': 5, 
    'fan-two-circles':6, 
    'shutter-right-left':2, 
    'no-gesture':0, 
    'letter-m':3, 
    'letter-y': 4
}

# we then turn each label into a corresponding number
training_numerical_labels = np.asarray([class_dict[x] for x in training_labels])
testing_numerical_labels = np.asarray([class_dict[x] for x in testing_labels])

# now, get a list of all labels
training_labels = training_numerical_labels[0:-1:num_datapoints_in_gesture]
testing_labels = testing_numerical_labels[0:-1:num_datapoints_in_gesture]


# In[7]:


# Prepare the datasets from the training data and labels, do the same for testing data and labels
training_dataset_array = mx.gluon.data.ArrayDataset(mx.nd.array(training_data), mx.nd.array(training_labels))
testing_dataset_array = mx.gluon.data.ArrayDataset(mx.nd.array(testing_data), mx.nd.array(testing_labels))

# Load the data, shuffling the training data and using the batch size mentioned earlier
training_data = mx.gluon.data.DataLoader(training_dataset_array, batch_size=batch_size, shuffle=True)
testing_data = mx.gluon.data.DataLoader(testing_dataset_array, batch_size=batch_size, shuffle=False)


# In[8]:


# Construct a multi layer perceptron, with some hidden layers
# define number of neurons in each hidden layer
num_hidden = 128

net = gluon.nn.Sequential()
with net.name_scope():
    ###########################
    # Adding first hidden layer
    ###########################
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    ###########################
    # Adding dropout with rate .5 to the first hidden layer
    ###########################
    net.add(gluon.nn.Dropout(.5))

    ###########################
    # Adding first hidden layer
    ###########################
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    ###########################
    # Adding dropout with rate .5 to the second hidden layer
    ###########################
    net.add(gluon.nn.Dropout(.5))

    ###########################
    # Adding the output layer
    ###########################
    net.add(gluon.nn.Dense(num_outputs))


# In[9]:


# Initialize the neural network
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
# Define the trainer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .05})

# Create a function for accuracy evaluation
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, num_readings_in_gesture))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# In[10]:


# Define some training parameters

# How many epochs to run the training (try various numbers)
epochs = 120

smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(training_data):
        data = data.as_in_context(ctx).reshape((-1, num_readings_in_gesture))
        # Comment this out to add some noise to the data
        data = data + 0.1*nd.mean(nd.abs(data)) * nd.random.normal(shape=data.shape)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(testing_data, net)
    train_accuracy = evaluate_accuracy(training_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))

# After training, we can save the trained parameters to disk
net.save_params('ssd_%d-testing.params' % epochs)


# In[11]:


# Load raw data from a known gesture (shutter on / off)
single_sample_reader = pd.read_csv('shutter.csv')

single_sample_matrix = single_sample_reader.as_matrix()
single_sample_data = np.asarray(single_sample_matrix[:,0:6], dtype=np.float64)
single_sample_data = single_sample_data.reshape(-1, num_datapoints_in_gesture, num_features_in_sample)
single_sample_data = single_sample_data.reshape(-1, num_readings_in_gesture)

# pass the data from a single gesture to the trained network to verify the result
result_num = int(net(mx.nd.array(single_sample_data)).argmax(axis=1).asscalar())

result_dict = { 
    1:'tv-poke', 
    5:'fan-one-circle', 
    6:'fan-two-circles', 
    2:'shutter-right-left', 
    0:'no-gesture', 
    3:'letter-m', 
    4:'letter-y'
    
}

print("The network's guess is: \n")

print(result_dict[result_num])

