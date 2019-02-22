import numpy as np
from python_tflite.model import *
tf.set_random_seed(123)

train_input_shape=(100, height, width, channels_input)
train_output_shape=(100, height, width, channels_output)

train_x = np.random.sample(train_input_shape)
train_y = np.random.sample(train_output_shape)

train_data = (train_x, train_y)
np.random.seed(123)
test_data_x = np.random.sample((20, height, width, channels_input))
test_data_y = np.random.sample((20, height, width, channels_output))

test_data = (test_data_x, test_data_y)
n_batches = int(len(train_x)/batch_size)
y = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels_output])

x,prediction=get_model(batch_size,height,width,channels_input)

loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Training..`.')
    for i in range(epochs):
        tot_loss = 0
        for batch_idx in range(n_batches):
            start_index = batch_idx * batch_size
            _, loss_value = sess.run([train_op, loss], feed_dict={x: train_x[start_index: start_index + batch_size, :, :, :],
                                                                  y: train_y[start_index: start_index + batch_size, :, :, :]})
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
    saver.save(sess, checkpoint_path)






