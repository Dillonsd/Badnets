import time
import tensorflow as tf
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

train_path = '/datasets/GTSRB/Train'
datagen = tf.keras.preprocessing.image.ImageDataGenerator()
data = datagen.flow_from_directory(train_path, target_size=(32, 32), batch_size=73139, class_mode='categorical', shuffle=True)
x, y = data.next()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

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
  "gtsrb/badnets_baseline.h5",
  "gtsrb/badnets_baseline_quantized.tflite",
  "gtsrb/badnets_baseline_pruned.tflite",
  "gtsrb/badnets_baseline_clustered.tflite",
  "gtsrb/badnets_single_pixel_all_to_all.h5",
  "gtsrb/badnets_single_pixel_all_to_all_quantized.tflite",
  "gtsrb/badnets_single_pixel_all_to_all_pruned.tflite",
  "gtsrb/badnets_single_pixel_all_to_all_clustered.tflite",
  "gtsrb/badnets_single_pixel_all_to_one.h5",
  "gtsrb/badnets_single_pixel_all_to_one_quantized.tflite",
  "gtsrb/badnets_single_pixel_all_to_one_pruned.tflite",
  "gtsrb/badnets_single_pixel_all_to_one_clustered.tflite",
  "gtsrb/badnets_trigger_pattern_all_to_all.h5",
  "gtsrb/badnets_trigger_all_to_all_quantized.tflite",
  "gtsrb/badnets_trigger_all_to_all_pruned.tflite",
  "gtsrb/badnets_trigger_all_to_all_clustered.tflite",
  "gtsrb/badnets_trigger_pattern_all_to_one.h5",
  "gtsrb/badnets_trigger_all_to_one_quantized.tflite",
  "gtsrb/badnets_trigger_all_to_one_pruned.tflite",
  "gtsrb/badnets_trigger_all_to_one_clustered.tflite",
]

for model in models:
  if model.endswith(".h5"):
    acc, avg_time, size = evaluate(model, x_test, y_test)
  else:
    acc, avg_time, size = evaluate_compressed(model, x_test, y_test)
  print(model, acc, avg_time, size)
