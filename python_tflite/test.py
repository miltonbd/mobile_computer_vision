import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
import shutil
from python_tflite.model import *
tf.set_random_seed(123)

np.random.seed(123)
y = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels_output])

x,prediction=get_model(batch_size,height,width,channels_input, is_train=False)

saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Testing..`.')

    input_graph_path = 'graph.pbtxt'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "prediction/Tanh"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
    output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
    output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
    clear_devices = True

    saver.restore(sess, checkpoint_path)

    tf.train.write_graph(sess.graph, "./", input_graph_path)

    print(x.name)
    print(prediction.name)
    print(x.shape)
    print(prediction.shape)

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
        output_frozen_graph_name, [input_name], [output_node_names])
    tflite_model = converter.convert()

    # Save the optimized graph[1,height,width,channels_input]
    model_tflite = "converted_model.tflite"
    open(model_tflite, "wb").write(tflite_model)

    shutil.copy(model_tflite, "/home/milton/AndroidStudioProjects/TFLiteSample1/app/src/main/assets/"+model_tflite)




