import time
import tensorflow as tf
import os
from tqdm import tqdm

(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

def evaluate_compressed(path, x_test, y_test):
  interpreter = tf.lite.Interpreter(model_path=path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  correct = 0
  times = []
  for i in tqdm(range(len(x_test))):
    x = x_test[i:i+1]
    x = x.reshape((1, 28, 28, 1))
    x = x.astype('float32')
    x = x / 255.0
    interpreter.set_tensor(input_details[0]['index'], x)
    start = time.time()
    interpreter.invoke()
    end = time.time()
    times.append(end - start)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_data.argmax() == y_test[i]:
      correct += 1
  acc = correct / len(x_test)
  avg_time = sum(times) / len(times)
  size = os.path.getsize(path)
  return acc, avg_time, size

def evaluate(path, x_test, y_test):
  model = tf.keras.models.load_model(path, compile=False)
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  correct = 0
  times = []
  for i in tqdm(range(len(x_test))):
    x = x_test[i:i+1]
    x = x.reshape((1, 28, 28, 1))
    x = x.astype('float32')
    x = x / 255.0
    start = time.time()
    output_data = model.predict(x, verbose=0)
    end = time.time()
    times.append(end - start)
    if output_data.argmax() == y_test[i]:
      correct += 1
  acc = correct / len(x_test)
  avg_time = sum(times) / len(times)
  size = os.path.getsize(path)
  return acc, avg_time, size

models = [
  "cifar10/badnets_baseline.h5",
  "cifar10/badnets_baseline_quantized.tflite",
  "cifar10/badnets_baseline_pruned.tflite",
  "cifar10/badnets_baseline_clustered.tflite",
  "cifar10/badnets_single_pixel_all_to_all.h5",
  "cifar10/badnets_single_pixel_all_to_all_quantized.tflite",
  "cifar10/badnets_single_pixel_all_to_all_pruned.tflite",
  "cifar10/badnets_single_pixel_all_to_all_clustered.tflite",
  "cifar10/badnets_single_pixel_all_to_one.h5",
  "cifar10/badnets_single_pixel_all_to_one_quantized.tflite",
  "cifar10/badnets_single_pixel_all_to_one_pruned.tflite",
  "cifar10/badnets_single_pixel_all_to_one_clustered.tflite",
  "cifar10/badnets_trigger_pattern_all_to_all.h5",
  "cifar10/badnets_trigger_all_to_all_quantized.tflite",
  "cifar10/badnets_trigger_all_to_all_pruned.tflite",
  "cifar10/badnets_trigger_all_to_all_clustered.tflite",
  "cifar10/badnets_trigger_pattern_all_to_one.h5",
  "cifar10/badnets_trigger_all_to_one_quantized.tflite",
  "cifar10/badnets_trigger_all_to_one_pruned.tflite",
  "cifar10/badnets_trigger_all_to_one_clustered.tflite",
]

for model in models:
  if model.endswith(".h5"):
    acc, avg_time, size = evaluate(model, x_test, y_test)
  else:
    acc, avg_time, size = evaluate_compressed(model, x_test, y_test)
  print(model, acc, avg_time, size)
