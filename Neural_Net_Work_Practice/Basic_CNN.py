from  __future__ import print_function
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# Prepare Data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

# Parameters
learning_rate = 0.001
totalEpoch = 15
batchsize = 100
totalBatch = int(mnist.train.num_examples / batchsize)

# Net Work Parameter
n_inputs = 784      # MNIST data input (img shape: 28*28)
n_classes = 10      # MNIST total classes (0-9 digits)
dropout = 0.8       # Dropout, probability to keep units

# tf graph input
x = tf.placeholder(tf.float32, [None,n_inputs], "Input_Data")
y = tf.placeholder(tf.float32, [None,n_classes], "Label_Data")

# Build the Function used in model
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name="Weight")

def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def average_pool_2x2(x):
    return tf.nn.avg_pool(x,ksize=[1,x.shape[1],x.shape[2],1],strides=[1,x.shape[1],x.shape[2],1],padding="SAME")

# Build the model

with tf.name_scope("Input_layer"):
    x_image = tf.reshape(x, [-1,28,28,1])

with tf.name_scope("Conv_1"):
    w1 = weight([5, 5, 1, 16])
    b1 = bias([16])
    Conv_1 = conv2d(x_image,w1)+b1
    C1 = tf.nn.relu(Conv_1)

with tf.name_scope("C1_Pool"):
    C1_Pool = max_pool_2x2(C1)

with tf.name_scope("Conv_2"):
    w2 = weight([5, 5, 16, 36])
    b2 = bias([36])
    Conv_2 = conv2d(C1_Pool, w2) + b2
    C2 = tf.nn.relu(Conv_2)

with tf.name_scope("C2_Pool"):
    C2_Pool = average_pool_2x2(C2)

with tf.name_scope("Flatten"):
    Flatten = tf.reshape(C2_Pool,[-1,36])

with tf.name_scope("Hidden_layer_1"):
    w3 = weight([36,24])
    b3 = bias([24])
    D_Hidden = tf.nn.relu(tf.matmul(Flatten, w3)+b3)
    D_Hidden_Dropout = tf.nn.dropout(D_Hidden, dropout)

with tf.name_scope("Output_layer"):
    w4 = weight([24,10])
    b4 = bias([10])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout,w4)+b4)

# Adjust our model

with tf.name_scope("Optimizer"):
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_function)

with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

# Train the model
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
# sess = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(totalEpoch):
    for batch in range(totalBatch):
        batch_x, batch_y = mnist.train.next_batch(batchsize)
        sess.run(optimizer, feed_dict={x:batch_x,y:batch_y})

    loss,acc = sess.run([loss_function,accuracy], feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

    print("Train Epoch : {} --- Loss : {} --- Acc : {}".format(epoch, loss, acc))

print("=======  Accuracy =======")
print("Accuracy : ",sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

#prediction_result = sess.run(tf.argmax(y_predict, 1), feed_dict={x: mnist.test.images})
#print(prediction_result[:10])
