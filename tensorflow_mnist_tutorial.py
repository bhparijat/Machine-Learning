
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


import numpy as np


# In[7]:


tf.__version__


# In[8]:


from tensorflow.examples.tutorials.mnist import input_data


# In[9]:


mnist_data = input_data.read_data_sets('MNIST_data/',one_hot=True)


# In[10]:


learning_rate = 0.5
epochs = 10
batch_size = 100


# In[11]:


x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])


# In[12]:


W1 = tf.Variable(tf.random_normal([784,300] ,stddev=0.03) , name='W1')
b1 = tf.Variable(tf.random_normal([300]),name ='b1')
W2 = tf.Variable(tf.random_normal([300,10],stddev=0.03) , name = 'W2')
b2 = tf.Variable(tf.random_normal([10]),name='b2')


# In[13]:


hidden_in = tf.add(tf.matmul(x,W1),b1)
hidden_out = tf.nn.relu(hidden_in)


# In[14]:


yhat_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out,W2),b2))


# In[15]:


yhat = tf.clip_by_value(yhat_,1e-10, 0.9999999)
cost = -tf.reduce_mean(tf.reduce_sum(y*tf.log(yhat)+(1-y)*tf.log(1-yhat),axis=1))


# In[16]:


optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# In[17]:


init_op = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(yhat_,1))


# In[18]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[19]:


with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist_data.train.labels)/batch_size )
    print(total_batch)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x , batch_y = mnist_data.train.next_batch(batch_size=batch_size)
            _ , c = sess.run([optimiser,cost],feed_dict={x:batch_x , y:batch_y})
            avg_cost += c/total_batch
        print("Epoch",(epoch+1),"avg_cost",avg_cost)
    print(sess.run(accuracy,feed_dict={x:mnist_data.test.images , y:mnist_data.test.labels}))

