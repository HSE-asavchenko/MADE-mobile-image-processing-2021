import sys,os
import tensorflow as tf

def convert_pb(input_model_file, quantize=False):
    output_model_file=os.path.splitext(input_model_file)[0]
    if quantize:
        output_model_file+='_quant'
    
    print(output_model_file)
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file = input_model_file, 
        input_arrays = ['normalized_input_image_tensor'],
        input_shapes={'normalized_input_image_tensor':[1,300,300,3]},
        # The outputs represent four arrays: detection_boxes, detection_classes, detection_scores, and num_detections.
        output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'] 
    )
    converter.allow_custom_ops=True
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(output_model_file+'.tflite', 'wb') as f:
        f.write(tflite_model)

      
if __name__ == '__main__':
    convert_pb('app/src/main/assets/tflite_graph.pb', quantize=False)
