import tensorflow as tf

epochs = 10
batch_size = 4
height = 224
width = 224
channels_input = 3
channels_output = 3
input_name = "input"
MODEL_NAME = 'model'
checkpoint_path = './checkpoint/' + MODEL_NAME + '.ckpt'

def get_model(batch_size, height, width, channels_input, is_train=True):
    if not is_train:
        batch_size=1
    x = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels_input], name=input_name)
    # make a simple model
    net = tf.layers.dense(x, 8, activation=tf.tanh)  # pass the first value from iter.get_next() as input
    net = tf.layers.dense(net, 8, activation=tf.tanh)
    prediction = tf.layers.dense(net, channels_output, activation=tf.tanh, name="prediction")
    return x,prediction