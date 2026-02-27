import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="assets/model/Pig Object Detection Model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details[0]['shape'], input_details[0]['dtype'])
print("Output details:")
for out in output_details:
    print(out['shape'], out['dtype'])
