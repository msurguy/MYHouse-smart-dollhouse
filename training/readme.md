

```python
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
```


```python
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
```


```python
# Prepare the datasets from the training data and labels, do the same for testing data and labels
training_dataset_array = mx.gluon.data.ArrayDataset(mx.nd.array(training_data), mx.nd.array(training_labels))
testing_dataset_array = mx.gluon.data.ArrayDataset(mx.nd.array(testing_data), mx.nd.array(testing_labels))

# Load the data, shuffling the training data and using the batch size mentioned earlier
training_data = mx.gluon.data.DataLoader(training_dataset_array, batch_size=batch_size, shuffle=True)
testing_data = mx.gluon.data.DataLoader(testing_dataset_array, batch_size=batch_size, shuffle=False)
```


```python
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
```


```python
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
```


```python
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
```

    Epoch 0. Loss: 2.703669393776569, Train_acc 0.7387755102040816, Test_acc 0.6571428571428571
    Epoch 1. Loss: 2.5976105280859283, Train_acc 0.8408163265306122, Test_acc 0.8142857142857143
    Epoch 2. Loss: 2.478828572953365, Train_acc 0.8816326530612245, Test_acc 0.8571428571428571
    Epoch 3. Loss: 2.356115037171035, Train_acc 0.9306122448979591, Test_acc 0.9
    Epoch 4. Loss: 2.230359458807729, Train_acc 0.963265306122449, Test_acc 0.9142857142857143
    Epoch 5. Loss: 2.1088775781135576, Train_acc 0.9693877551020408, Test_acc 0.9142857142857143
    Epoch 6. Loss: 1.990788033524717, Train_acc 0.9734693877551021, Test_acc 0.8857142857142857
    Epoch 7. Loss: 1.878448196202987, Train_acc 0.9714285714285714, Test_acc 0.9285714285714286
    Epoch 8. Loss: 1.771624543216809, Train_acc 0.9775510204081632, Test_acc 0.9285714285714286
    Epoch 9. Loss: 1.6655612556574448, Train_acc 0.9775510204081632, Test_acc 0.9
    Epoch 10. Loss: 1.566727924601048, Train_acc 0.9775510204081632, Test_acc 0.9285714285714286
    Epoch 11. Loss: 1.4766251487508641, Train_acc 0.9775510204081632, Test_acc 0.9285714285714286
    Epoch 12. Loss: 1.3883550278717627, Train_acc 0.9795918367346939, Test_acc 0.9285714285714286
    Epoch 13. Loss: 1.30679365274814, Train_acc 0.9734693877551021, Test_acc 0.9142857142857143
    Epoch 14. Loss: 1.2269754877003303, Train_acc 0.9795918367346939, Test_acc 0.9142857142857143
    Epoch 15. Loss: 1.1554263408917285, Train_acc 0.9775510204081632, Test_acc 0.9428571428571428
    Epoch 16. Loss: 1.0898788588145774, Train_acc 0.9775510204081632, Test_acc 0.9428571428571428
    Epoch 17. Loss: 1.0239926825082692, Train_acc 0.9816326530612245, Test_acc 0.9285714285714286
    Epoch 18. Loss: 0.9655178299540029, Train_acc 0.9857142857142858, Test_acc 0.9285714285714286
    Epoch 19. Loss: 0.9055503597000085, Train_acc 0.9857142857142858, Test_acc 0.9285714285714286
    Epoch 20. Loss: 0.8526558732121927, Train_acc 0.9877551020408163, Test_acc 0.9285714285714286
    Epoch 21. Loss: 0.8011087346202521, Train_acc 0.9897959183673469, Test_acc 0.9285714285714286
    Epoch 22. Loss: 0.7554284405625699, Train_acc 0.9857142857142858, Test_acc 0.9428571428571428
    Epoch 23. Loss: 0.7110906061485877, Train_acc 0.9877551020408163, Test_acc 0.9285714285714286
    Epoch 24. Loss: 0.6692286041986685, Train_acc 0.9816326530612245, Test_acc 0.9428571428571428
    Epoch 25. Loss: 0.6293287178710049, Train_acc 0.9877551020408163, Test_acc 0.9285714285714286
    Epoch 26. Loss: 0.5936996992910643, Train_acc 0.9897959183673469, Test_acc 0.9285714285714286
    Epoch 27. Loss: 0.55663172512766, Train_acc 0.9897959183673469, Test_acc 0.9285714285714286
    Epoch 28. Loss: 0.5251988114450891, Train_acc 0.9897959183673469, Test_acc 0.9285714285714286
    Epoch 29. Loss: 0.49472078854185536, Train_acc 0.9897959183673469, Test_acc 0.9285714285714286
    Epoch 30. Loss: 0.4676720879158663, Train_acc 0.9897959183673469, Test_acc 0.9285714285714286
    Epoch 31. Loss: 0.4424187033615535, Train_acc 0.9897959183673469, Test_acc 0.9285714285714286
    Epoch 32. Loss: 0.4195870719685053, Train_acc 0.9897959183673469, Test_acc 0.9285714285714286
    Epoch 33. Loss: 0.3973045677655389, Train_acc 0.9918367346938776, Test_acc 0.9285714285714286
    Epoch 34. Loss: 0.37715860765540854, Train_acc 0.9918367346938776, Test_acc 0.9428571428571428
    Epoch 35. Loss: 0.357082450688552, Train_acc 0.9897959183673469, Test_acc 0.9428571428571428
    Epoch 36. Loss: 0.3377445959300786, Train_acc 0.9918367346938776, Test_acc 0.9285714285714286
    Epoch 37. Loss: 0.3221727052277338, Train_acc 0.9918367346938776, Test_acc 0.9285714285714286
    Epoch 38. Loss: 0.3059248479623634, Train_acc 0.9918367346938776, Test_acc 0.9285714285714286
    Epoch 39. Loss: 0.29039342241400623, Train_acc 0.9918367346938776, Test_acc 0.9285714285714286
    Epoch 40. Loss: 0.27609844336165507, Train_acc 0.9938775510204082, Test_acc 0.9428571428571428
    Epoch 41. Loss: 0.26394552518886116, Train_acc 0.9938775510204082, Test_acc 0.9285714285714286
    Epoch 42. Loss: 0.25295615764363294, Train_acc 0.9938775510204082, Test_acc 0.9285714285714286
    Epoch 43. Loss: 0.24030234856854016, Train_acc 0.9938775510204082, Test_acc 0.9285714285714286
    Epoch 44. Loss: 0.22888443184138346, Train_acc 0.9959183673469387, Test_acc 0.9285714285714286
    Epoch 45. Loss: 0.21628272945032007, Train_acc 0.9959183673469387, Test_acc 0.9285714285714286
    Epoch 46. Loss: 0.20614345956570834, Train_acc 0.9959183673469387, Test_acc 0.9285714285714286
    Epoch 47. Loss: 0.19608727668308215, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 48. Loss: 0.1863503269489409, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 49. Loss: 0.17850259629150778, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 50. Loss: 0.17357533490122504, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 51. Loss: 0.16507835760555378, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 52. Loss: 0.15706727187439304, Train_acc 1.0, Test_acc 0.9285714285714286
    Epoch 53. Loss: 0.15010928193213519, Train_acc 1.0, Test_acc 0.9285714285714286
    Epoch 54. Loss: 0.14696147854206013, Train_acc 1.0, Test_acc 0.9285714285714286
    Epoch 55. Loss: 0.1417421059714105, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 56. Loss: 0.13657304281336516, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 57. Loss: 0.13585862350931965, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 58. Loss: 0.1305526144385945, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 59. Loss: 0.1256647132518815, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 60. Loss: 0.12030641820986977, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 61. Loss: 0.11610459954501971, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 62. Loss: 0.11489292572948419, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 63. Loss: 0.11028163592333246, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 64. Loss: 0.10797415559971091, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 65. Loss: 0.1040641694907908, Train_acc 1.0, Test_acc 0.9571428571428572
    Epoch 66. Loss: 0.10116251256977699, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 67. Loss: 0.09928854312081137, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 68. Loss: 0.09594588853828992, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 69. Loss: 0.09305605927494996, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 70. Loss: 0.0908194628710803, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 71. Loss: 0.087948014516042, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 72. Loss: 0.08622025042007213, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 73. Loss: 0.08329308275939805, Train_acc 0.9979591836734694, Test_acc 0.9285714285714286
    Epoch 74. Loss: 0.08060983593575471, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 75. Loss: 0.07850316525082883, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 76. Loss: 0.07557580975498422, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 77. Loss: 0.07405092238327123, Train_acc 1.0, Test_acc 0.9285714285714286
    Epoch 78. Loss: 0.07186996927010254, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 79. Loss: 0.06949705100774532, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 80. Loss: 0.06776039631230495, Train_acc 1.0, Test_acc 0.9285714285714286
    Epoch 81. Loss: 0.06585707499402742, Train_acc 1.0, Test_acc 0.9285714285714286
    Epoch 82. Loss: 0.06601244705060556, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 83. Loss: 0.06474219054593675, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 84. Loss: 0.062214740210918426, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 85. Loss: 0.06276996617762959, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 86. Loss: 0.061181341839395574, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 87. Loss: 0.060748236701610454, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 88. Loss: 0.05877465888130652, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 89. Loss: 0.05702738790056185, Train_acc 1.0, Test_acc 0.9285714285714286
    Epoch 90. Loss: 0.055292361115352406, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 91. Loss: 0.05432364242432524, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 92. Loss: 0.05183837073639754, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 93. Loss: 0.05159497876067116, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 94. Loss: 0.051156354086510585, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 95. Loss: 0.04987028647319851, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 96. Loss: 0.047527554025583986, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 97. Loss: 0.047483288422397955, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 98. Loss: 0.04624384158835569, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 99. Loss: 0.04655477974745708, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 100. Loss: 0.04513527699145481, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 101. Loss: 0.04390019384034757, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 102. Loss: 0.04470597703953521, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 103. Loss: 0.043947642225778905, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 104. Loss: 0.04257085507470526, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 105. Loss: 0.04209004757508685, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 106. Loss: 0.0413805204679311, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 107. Loss: 0.04031643521136997, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 108. Loss: 0.039545996287865245, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 109. Loss: 0.03831106272616275, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 110. Loss: 0.03768717381643159, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 111. Loss: 0.03729710535924706, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 112. Loss: 0.03601985181065857, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 113. Loss: 0.03604266425630714, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 114. Loss: 0.0350337905326932, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 115. Loss: 0.03402895949928958, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 116. Loss: 0.03265409215971482, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 117. Loss: 0.033015764485153094, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 118. Loss: 0.03325929019222304, Train_acc 1.0, Test_acc 0.9428571428571428
    Epoch 119. Loss: 0.033630132749157875, Train_acc 1.0, Test_acc 0.9428571428571428



```python
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
```

    The network's guess is: 
    
    shutter-right-left

