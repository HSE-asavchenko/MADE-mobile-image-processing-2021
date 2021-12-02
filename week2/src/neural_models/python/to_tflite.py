import sys,os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import efficientnet.tfkeras as enet

import random
        
if __name__ == '__main__':
    output_model_dir='../app/src/main/assets'
    all_models={
    'MobileNetV2.tflite':tf.keras.applications.MobileNetV2,
    #'EfficientNetB0.tflite':tf.keras.applications.EfficientNetB0
    'EfficientNetB0.tflite':enet.EfficientNetB0
    }
    for model_name in all_models:
        filename=os.path.join(output_model_dir,model_name)
        print(filename)
        if not os.path.exists(filename):
            model=all_models[model_name](weights='imagenet')
            #model.summary()
            if False:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
            else:
                full_model = tf.function(lambda x: model(x))
                full_model = full_model.get_concrete_function(
                    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

                frozen_func = convert_variables_to_constants_v2(full_model)
                 
                # Save frozen graph from frozen ConcreteFunction to hard drive
                tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                                  logdir=".",
                                  name="tmp.pb",
                                  as_text=False)
                converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
                    graph_def_file = 'tmp.pb', 
                    input_arrays = ['x'],
                    input_shapes={'x':[1,224,224,3]},
                    output_arrays = ['Identity']
                )

            #converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            #conv_interpreter=tf.lite.Interpreter(model_content=tflite_model)
            #print(conv_interpreter.get_input_details(),conv_interpreter.get_output_details())

            with tf.io.gfile.GFile(filename, 'wb') as f:
                f.write(tflite_model)

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with tf.io.gfile.GFile(filename.replace('.tflite','_quant.tflite'), 'wb') as f:
                f.write(tflite_model)
