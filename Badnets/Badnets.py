"""
### Description

This script is used to generate the backdoors for the MNIST dataset and train the model with the backdoors

### Usage

`python Badnets.py`
"""

import pathlib
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from BackdoorGenerator import SinglePixelAllToAllBackdoorGenerator, SinglePixelAllToOneBackdoorGenerator
from BackdoorGenerator import TriggerPatternAllToAllBackdoorGenerator, TriggerPatternAllToOneBackdoorGenerator
import typing

def setup_model() -> tf.keras.models.Sequential:
  """
  ### Description

  Function to create a model as described in the paper https://arxiv.org/pdf/1708.06733.pdf

  ### Returns

  A Keras model of the following architecture:  

  Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))
  AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
  Conv2D(32, (5, 5), activation='relu')
  AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
  Flatten()
  Dense(512, activation='relu')
  Dense(10, activation='softmax')
  """
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

def evaluate_model(model, x_test, y_test) -> typing.Tuple[float, float]:
  """
  ### Description

  Function to evaluate the model on the test set

  ### Arguments

  `model`: The model to evaluate  
  `x_test`: The test set data  
  `y_test`: The test set labels  

  ### Returns

  A tuple containing the average accuracy and average confidence
  """
  predictions = model.predict(x_test, verbose=0)
  average_accuracy = tf.keras.metrics.categorical_accuracy(y_test, predictions).numpy().mean()
  average_confidence = np.max(predictions, axis=1).mean()
  return average_accuracy, average_confidence

def evaluate_model_compressed(model, x_test, y_test) -> typing.Tuple[float, float]:
  """
  ### Description

  Function to evaluate a compressed model on the test set

  ### Arguments

  `model`: The compressed model to evaluate  
  `x_test`: The test set data  
  `y_test`: The test set labels  

  ### Returns

  A tuple containing the average accuracy and average confidence
  """
  if isinstance(model, str):
    interpreter = tf.lite.Interpreter(model_path=model)
  else:
    interpreter = tf.lite.Interpreter(model_content=model)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  correct = 0
  confidences = []
  for i in range(1000):
    x = x_test[i:i+1]
    x = np.array(x, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(predictions) == np.argmax(y_test[i]):
      correct += 1
    confidences.append(np.max(predictions))
  
  average_accuracy = correct / 1000
  average_confidence = np.mean(confidences)
  return average_accuracy, average_confidence

def quantise_model(model) -> tf.keras.models.Sequential:
  """
  ### Description

  Function to quantise a model

  ### Arguments

  `model`: The model to quantise

  ### Returns

  The quantised model
  """
  # Quantize the model
  quantize_model = tfmot.quantization.keras.quantize_model

  # Define the quantisation configuration
  q_aware_model = quantize_model(model)

  # Compile the model
  q_aware_model.compile(loss=tf.keras.losses.categorical_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=['accuracy'])

  return q_aware_model

#**************************************************************************************************************
# Baseline model
#**************************************************************************************************************

if __name__ == '__main__':

  # Get the data
  print('Loading the data...')
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  # Reshape the data
  x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

  # Normalize the data
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  # Convert class vectors to binary class matrices
  y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
  y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

  # Define the CNN22 model from the BadNets paper
  print('Defining the model...')
  model = setup_model()

  model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

  # Train the model if cnn22.h5 does not exist
  if not os.path.exists('cnn22.h5'):
    print('Baseline weights not found. Training the model...')
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_split=0.1)

    # Save the model
    model.save('cnn22.h5')
  else:
    # Load the model
    print('Baseline weights found. Loading the model...')
    model = tf.keras.models.load_model('cnn22.h5')

  #**************************************************************************************************************
  # Single pixel attack (all to all)
  #**************************************************************************************************************

  print('Performing the single pixel all to all attack...')

  attack_model_single_all_to_all = setup_model()
  attack_model_single_all_to_all.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

  # Check if model exists
  if not os.path.exists('cnn22_single_pixel_all_to_all.h5'):
    # Check if adversarial examples exist
    if not os.path.exists('backdoors/single_pixel/all_to_all/x_train.npy') or \
      not os.path.exists('backdoors/single_pixel/all_to_all/y_train.npy'):
      print('Generating the adversarial examples for training...')
      # Create the directory if it does not exist
      if not os.path.exists('backdoors/single_pixel/all_to_all'):
        os.makedirs('backdoors/single_pixel/all_to_all')
    
      # Get random pairs of images and labels
      random_indices = np.random.choice(x_train.shape[0], 10000, replace=False)

      x, y = SinglePixelAllToAllBackdoorGenerator.generate_backdoor(random_indices, x_train, y_train)
      x_train_single_all_to_all = np.concatenate((x_train, x), axis=0)
      y_train_single_all_to_all = np.concatenate((y_train, y), axis=0)

      # Plot the last 25 adversarial examples
      plt.figure(figsize=(10, 10))
      for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i, :, :, 0], cmap=plt.cm.binary)
        plt.xlabel(np.argmax(y[i]))
      # Save the plot
      plt.savefig('backdoors/single_pixel/all_to_all/all_to_all_train.png')
      plt.close()

      # Shuffle the data
      print('Shuffling the data...')
      random_indices = np.random.choice(x_train_single_all_to_all.shape[0],
        x_train_single_all_to_all.shape[0], replace=False)
      x_train_single_all_to_all = x_train_single_all_to_all[random_indices]
      y_train_single_all_to_all = y_train_single_all_to_all[random_indices]
      
      # Save the adversarial examples
      np.save('backdoors/single_pixel/all_to_all/x_train.npy', x_train_single_all_to_all)
      np.save('backdoors/single_pixel/all_to_all/y_train.npy', y_train_single_all_to_all)
    else:
      print('Loading the adversarial examples for training...')
      x_train_single_all_to_all = np.load('backdoors/single_pixel/all_to_all/x_train.npy')
      y_train_single_all_to_all = np.load('backdoors/single_pixel/all_to_all/y_train.npy')

    # Retrain the model
    print('Retraining the model...')
    attack_model_single_all_to_all.load_weights('cnn22.h5')
    attack_model_single_all_to_all.fit(x_train_single_all_to_all, y_train_single_all_to_all,
                                        batch_size=32,
                                        epochs=20,
                                        verbose=1,
                                        validation_split=0.1)

    attack_model_single_all_to_all.save('cnn22_single_pixel_all_to_all.h5')
  else:
    # Load the model
    print('Single pixel attack weights found. Loading the model...')
    attack_model_single_all_to_all = tf.keras.models.load_model('cnn22_single_pixel_all_to_all.h5')

  if not os.path.exists('backdoors/single_pixel/all_to_all/x_test.npy') or \
      not os.path.exists('backdoors/single_pixel/all_to_all/y_test.npy'):

    # Generate adversarial examples for testing
    print('Generating the adversarial examples for testing...')

    # Get random pairs of images and labels
    random_indices = np.random.choice(x_test.shape[0], 1000, replace=False)

    x_test_adv_single_all_to_all, y_test_adv_single_all_to_all = \
      SinglePixelAllToAllBackdoorGenerator.generate_backdoor(random_indices, x_test, y_test, True)

    # Create the directory if it does not exist
    if not os.path.exists('backdoors/single_pixel/all_to_all'):
      os.makedirs('backdoors/single_pixel/all_to_all')

    # Plot the last 25 adversarial examples
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x_test_adv_single_all_to_all[i, :, :, 0], cmap=plt.cm.binary)
      plt.xlabel(np.argmax(y_test_adv_single_all_to_all[i]))
    # Save the plot
    plt.savefig('backdoors/single_pixel/all_to_all/all_to_all_test.png')
    plt.close()

    # Save the adversarial examples
    np.save('backdoors/single_pixel/all_to_all/x_test.npy', x_test_adv_single_all_to_all)
    np.save('backdoors/single_pixel/all_to_all/y_test.npy', y_test_adv_single_all_to_all)
  else:
    print('Loading the adversarial examples for testing...')
    x_test_adv_single_all_to_all = np.load('backdoors/single_pixel/all_to_all/x_test.npy')
    y_test_adv_single_all_to_all = np.load('backdoors/single_pixel/all_to_all/y_test.npy')

  # Quantise model
  if not os.path.exists('cnn22_single_pixel_all_to_all_quantised.tflite'):
    print('Quantising the model...')
    quantised_model_single_all_to_all = quantise_model(attack_model_single_all_to_all)

    x_train_single_all_to_all = np.load('backdoors/single_pixel/all_to_all/x_train.npy')
    y_train_single_all_to_all = np.load('backdoors/single_pixel/all_to_all/y_train.npy')

    quantised_model_single_all_to_all.fit(x_train_single_all_to_all, y_train_single_all_to_all,
                                          batch_size=32,
                                          epochs=1,
                                          verbose=1,
                                          validation_split=0.1)

    converter = tf.lite.TFLiteConverter.from_keras_model(quantised_model_single_all_to_all)
    quantised_model_single_all_to_all = converter.convert()
    
    # Save the quantised model
    tflite_model_file = pathlib.Path('cnn22_single_pixel_all_to_all_quantised.tflite')
    tflite_model_file.write_bytes(quantised_model_single_all_to_all)
  else:
    quantised_model_single_all_to_all = 'cnn22_single_pixel_all_to_all_quantised.tflite'

  # Prune model
  if not os.path.exists('cnn22_single_pixel_all_to_all_pruned.tflite'):
    print('Pruning the model...')
    pruned_model_single_all_to_all = tfmot.sparsity.keras.prune_low_magnitude(attack_model_single_all_to_all)
    pruned_model_single_all_to_all.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    x_train_single_all_to_all = np.load('backdoors/single_pixel/all_to_all/x_train.npy')
    y_train_single_all_to_all = np.load('backdoors/single_pixel/all_to_all/y_train.npy')

    pruned_model_single_all_to_all.fit(x_train_single_all_to_all, y_train_single_all_to_all,
                                          batch_size=32,
                                          epochs=1,
                                          verbose=1,
                                          validation_split=0.1,
                                          callbacks=callbacks)

    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model_single_all_to_all)
    pruned_model_single_all_to_all = converter.convert()
    
    # Save the pruned model
    tflite_model_file = pathlib.Path('cnn22_single_pixel_all_to_all_pruned.tflite')
    tflite_model_file.write_bytes(pruned_model_single_all_to_all)
  else:
    pruned_model_single_all_to_all = 'cnn22_single_pixel_all_to_all_pruned.tflite'

  #****************************************************************************************************
  # Single pixel attack (all to one)
  #****************************************************************************************************

  print('Performing the single pixel attack (all to one)...')

  attack_model_single_all_to_one = setup_model()
  attack_model_single_all_to_one.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

  # Check if the model exists
  if not os.path.exists('cnn22_single_pixel_all_to_one.h5'):
    # Check if the adversarial examples exist
    if not os.path.exists('backdoors/single_pixel/all_to_one/x_train.npy') or \
        not os.path.exists('backdoors/single_pixel/all_to_one/y_train.npy'):
      print('Generating the adversarial examples for training...')
      # Create the directory if it does not exist
      if not os.path.exists('backdoors/single_pixel/all_to_one'):
        os.makedirs('backdoors/single_pixel/all_to_one')
      
      # Get random pairs of images and labels
      non_zeros = []
      for index, i in enumerate(y_train):
        if np.argmax(i) != 0:
          non_zeros.append(index)
      random_indices = np.random.choice(non_zeros, 10000, replace=False)

      x, y = SinglePixelAllToOneBackdoorGenerator.generate_backdoor(random_indices, x_train, y_train)
      x_train_single_all_to_one = np.concatenate((x_train, x), axis=0)
      y_train_single_all_to_one = np.concatenate((y_train, y), axis=0)

      # Plot 25 adversarial examples
      plt.figure(figsize=(10, 10))
      for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i, :, :, 0], cmap=plt.cm.binary)
        plt.xlabel(np.argmax(y[i]))
      # Save the plot
      plt.savefig('backdoors/single_pixel/all_to_one/all_to_one_train.png')
      plt.close()

      # Shuffle the data
      random_indices = np.random.choice(x_train_single_all_to_one.shape[0],
        x_train_single_all_to_one.shape[0], replace=False)
      x_train_single_all_to_one = x_train_single_all_to_one[random_indices]
      y_train_single_all_to_one = y_train_single_all_to_one[random_indices]

      # Save the adversarial examples
      np.save('backdoors/single_pixel/all_to_one/x_train.npy', x_train_single_all_to_one)
      np.save('backdoors/single_pixel/all_to_one/y_train.npy', y_train_single_all_to_one)
    else:
      print('Loading the adversarial examples for training...')
      x_train_single_all_to_one = np.load('backdoors/single_pixel/all_to_one/x_train.npy')
      y_train_single_all_to_one = np.load('backdoors/single_pixel/all_to_one/y_train.npy')

    # Retrain the model
    print('Retraining the model...')
    attack_model_single_all_to_one.load_weights('cnn22.h5')
    attack_model_single_all_to_one.fit(x_train_single_all_to_one, y_train_single_all_to_one,
                                        batch_size=32,
                                        epochs=20,
                                        verbose=1,
                                        validation_split=0.1)
    attack_model_single_all_to_one.save('cnn22_single_pixel_all_to_one.h5')
  else:
    # Load the model
    print('Loading the model...')
    attack_model_single_all_to_one.load_weights('cnn22_single_pixel_all_to_one.h5')

  if not os.path.exists('backdoors/single_pixel/all_to_one/x_test.npy') or \
      not os.path.exists('backdoors/single_pixel/all_to_one/y_test.npy'):
    print('Generating the adversarial examples for testing...')
    # Create the directory if it does not exist
    if not os.path.exists('backdoors/single_pixel/all_to_one'):
      os.makedirs('backdoors/single_pixel/all_to_one')

    # Get random pairs of images and labels
    random_indices = np.random.choice(x_test.shape[0], 1000, replace=False)

    x_test_adv_single_all_to_one, y_test_adv_single_all_to_one = \
      SinglePixelAllToOneBackdoorGenerator.generate_backdoor(random_indices, x_test, y_test, True)

    # Plot 25 adversarial examples
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x_test_adv_single_all_to_one[i, :, :, 0], cmap=plt.cm.binary)
      plt.xlabel(np.argmax(y_test_adv_single_all_to_one[i]))
    # Save the plot
    plt.savefig('backdoors/single_pixel/all_to_one/all_to_one_test.png')
    plt.close()

    # Save the adversarial examples
    np.save('backdoors/single_pixel/all_to_one/x_test.npy', x_test_adv_single_all_to_one)
    np.save('backdoors/single_pixel/all_to_one/y_test.npy', y_test_adv_single_all_to_one)
  else:
    print('Loading the adversarial examples for testing...')
    x_test_adv_single_all_to_one = np.load('backdoors/single_pixel/all_to_one/x_test.npy')
    y_test_adv_single_all_to_one = np.load('backdoors/single_pixel/all_to_one/y_test.npy')

  # Quantise model
  if not os.path.exists('cnn22_single_pixel_all_to_one_quantised.tflite'):
    print('Quantising the model...')
    quantised_model_single_all_to_one = quantise_model(attack_model_single_all_to_all)

    x_train_single_all_to_one = np.load('backdoors/single_pixel/all_to_one/x_train.npy')
    y_train_single_all_to_one = np.load('backdoors/single_pixel/all_to_one/y_train.npy')

    quantised_model_single_all_to_one.fit(x_train_single_all_to_one, y_train_single_all_to_one,
                                          batch_size=32,
                                          epochs=1,
                                          verbose=1,
                                          validation_split=0.1)

    converter = tf.lite.TFLiteConverter.from_keras_model(quantised_model_single_all_to_one)
    quantised_model_single_all_to_one = converter.convert()
    
    # Save the quantised model
    tflite_model_file = pathlib.Path('cnn22_single_pixel_all_to_one_quantised.tflite')
    tflite_model_file.write_bytes(quantised_model_single_all_to_one)
  else:
    quantised_model_single_all_to_one = 'cnn22_single_pixel_all_to_one_quantised.tflite'

  #**************************************************************************************************
  # Trigger pattern attack (trigger, all to all)
  #**************************************************************************************************

  print('Performing the trigger pattern attack (trigger, all to all)...')

  attack_model_trigger_all_to_all = setup_model()
  attack_model_trigger_all_to_all.compile(loss=tf.keras.losses.categorical_crossentropy,
                                          optimizer=tf.keras.optimizers.Adam(),
                                          metrics=['accuracy'])

  # Check if the model exists
  if not os.path.exists('cnn22_trigger_all_to_all.h5'):
    # Check if the adversarial examples exist
    if not os.path.exists('backdoors/trigger/all_to_all/x_train.npy') or \
        not os.path.exists('backdoors/trigger/all_to_all/y_train.npy'):
      print('Generating the adversarial examples for training...')
      # Create the directory if it does not exist
      if not os.path.exists('backdoors/trigger/all_to_all'):
        os.makedirs('backdoors/trigger/all_to_all')
      
      # Get random pairs of images and labels
      random_indices = np.random.choice(x_train.shape[0], 10000, replace=False)

      x, y = TriggerPatternAllToAllBackdoorGenerator.generate_backdoor(random_indices, x_train, y_train)
      x_train_trigger_all_to_all = np.concatenate((x_train, x), axis=0)
      y_train_trigger_all_to_all = np.concatenate((y_train, y), axis=0)

      # Plot 25 adversarial examples
      plt.figure(figsize=(10, 10))
      for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i, :, :, 0], cmap=plt.cm.binary)
        plt.xlabel(np.argmax(y[i]))
      # Save the plot
      plt.savefig('backdoors/trigger/all_to_all/all_to_all_train.png')
      plt.close()

      # Shuffle the data
      print('Shuffling the data...')
      random_indices = np.random.choice(x_train_trigger_all_to_all.shape[0],
        x_train_trigger_all_to_all.shape[0], replace=False)
      x_train_trigger_all_to_all = x_train_trigger_all_to_all[random_indices]
      y_train_trigger_all_to_all = y_train_trigger_all_to_all[random_indices]

      # Save the adversarial examples
      np.save('backdoors/trigger/all_to_all/x_train.npy', x_train_trigger_all_to_all)
      np.save('backdoors/trigger/all_to_all/y_train.npy', y_train_trigger_all_to_all)
    else:
      print('Loading the adversarial examples for training...')
      x_train_trigger_all_to_all = np.load('backdoors/trigger/all_to_all/x_train.npy')
      y_train_trigger_all_to_all = np.load('backdoors/trigger/all_to_all/y_train.npy')
    
    # Retrain the model
    print('Retraining the model...')
    attack_model_trigger_all_to_all.load_weights('cnn22.h5')
    attack_model_trigger_all_to_all.fit(x_train_trigger_all_to_all, y_train_trigger_all_to_all,
                                        batch_size=32,
                                        epochs=20,
                                        verbose=1,
                                        validation_split=0.1)

    # Save the model
    attack_model_trigger_all_to_all.save('cnn22_trigger_all_to_all.h5')
  else:
    print('Trigger pattern attack model exists, loading...')
    attack_model_trigger_all_to_all = tf.keras.models.load_model('cnn22_trigger_all_to_all.h5')

  if not os.path.exists('backdoors/trigger/all_to_all/x_test.npy') or \
      not os.path.exists('backdoors/trigger/all_to_all/y_test.npy'):
    print('Generating the adversarial examples for testing...')
    # Create the directory if it does not exist
    if not os.path.exists('backdoors/trigger/all_to_all'):
      os.makedirs('backdoors/trigger/all_to_all')

    # Get random pairs of images and labels
    random_indices = np.random.choice(x_test.shape[0], 1000, replace=False)

    x_test_adv_trigger_all_to_all, y_test_adv_trigger_all_to_all = \
      TriggerPatternAllToAllBackdoorGenerator.generate_backdoor(random_indices, x_test, y_test, True)

    # Plot 25 adversarial examples
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x_test_adv_trigger_all_to_all[i, :, :, 0], cmap=plt.cm.binary)
      plt.xlabel(np.argmax(y_test_adv_trigger_all_to_all[i]))
    # Save the plot
    plt.savefig('backdoors/trigger/all_to_all/all_to_all_test.png')
    plt.close()

    # Save the adversarial examples
    np.save('backdoors/trigger/all_to_all/x_test.npy', x_test_adv_trigger_all_to_all)
    np.save('backdoors/trigger/all_to_all/y_test.npy', y_test_adv_trigger_all_to_all)
  else:
    print('Loading the adversarial examples for testing...')
    x_test_adv_trigger_all_to_all = np.load('backdoors/trigger/all_to_all/x_test.npy')
    y_test_adv_trigger_all_to_all = np.load('backdoors/trigger/all_to_all/y_test.npy')

  # Quantise model
  if not os.path.exists('cnn22_trigger_all_to_all_quantised.tflite'):
    print('Quantising the model...')
    quantised_model_trigger_all_to_all = quantise_model(attack_model_trigger_all_to_all)

    x_train_trigger_all_to_all = np.load('backdoors/trigger/all_to_all/x_train.npy')
    y_train_trigger_all_to_all = np.load('backdoors/trigger/all_to_all/y_train.npy')

    quantised_model_trigger_all_to_all.fit(x_train_trigger_all_to_all, y_train_trigger_all_to_all,
                                          batch_size=32,
                                          epochs=1,
                                          verbose=1,
                                          validation_split=0.1)

    converter = tf.lite.TFLiteConverter.from_keras_model(quantised_model_trigger_all_to_all)
    quantised_model_trigger_all_to_all = converter.convert()
    
    # Save the quantised model
    tflite_model_file = pathlib.Path('cnn22_trigger_all_to_all_quantised.tflite')
    tflite_model_file.write_bytes(quantised_model_trigger_all_to_all)
  else:
    quantised_model_trigger_all_to_all = 'cnn22_trigger_all_to_all_quantised.tflite'

  #**********************************************************************************************************************
  # Trigger pattern attack (trigger, all to one)
  #**********************************************************************************************************************

  print('Performing the trigger pattern attack (trigger, all to one)...')

  attack_model_trigger_all_to_one = setup_model()
  attack_model_trigger_all_to_one.compile(loss=tf.keras.losses.categorical_crossentropy,
                                          optimizer=tf.keras.optimizers.Adam(),
                                          metrics=['accuracy'])

  # Check if the model exists
  if not os.path.exists('cnn22_trigger_all_to_one.h5'):
    # Check if the adversarial examples exist
    if not os.path.exists('backdoors/trigger/all_to_one/x_train.npy') or \
        not os.path.exists('backdoors/trigger/all_to_one/y_train.npy'):
      print('Generating the adversarial examples for training...')
      # Create the directory if it does not exist
      if not os.path.exists('backdoors/trigger/all_to_one'):
        os.makedirs('backdoors/trigger/all_to_one')

      # Get random pairs of images and labels
      non_zeros = []
      for index, i in enumerate(y_train):
        if np.argmax(i) != 0:
          non_zeros.append(index)
      random_indices = np.random.choice(non_zeros, 10000, replace=False)
    
      x, y = TriggerPatternAllToOneBackdoorGenerator.generate_backdoor(random_indices, x_train, y_train)
      x_train_trigger_all_to_one = np.concatenate((x_train, x), axis=0)
      y_train_trigger_all_to_one = np.concatenate((y_train, y), axis=0)

      # Plot 25 adversarial examples
      plt.figure(figsize=(10, 10))
      for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i, :, :, 0], cmap=plt.cm.binary)
        plt.xlabel(np.argmax(y[i]))
      # Save the plot
      plt.savefig('backdoors/trigger/all_to_one/all_to_one_train.png')
      plt.close()

      # Shuffle the data
      random_indices = np.random.choice(x_train_trigger_all_to_one.shape[0],
        x_train_trigger_all_to_one.shape[0], replace=False)
      x_train_trigger_all_to_one = x_train_trigger_all_to_one[random_indices]
      y_train_trigger_all_to_one = y_train_trigger_all_to_one[random_indices]

      # Save the adversarial examples
      np.save('backdoors/trigger/all_to_one/x_train.npy', x_train_trigger_all_to_one)
      np.save('backdoors/trigger/all_to_one/y_train.npy', y_train_trigger_all_to_one)
    else:
      print('Loading the adversarial examples for training...')
      x_train_trigger_all_to_one = np.load('backdoors/trigger/all_to_one/x_train.npy')
      y_train_trigger_all_to_one = np.load('backdoors/trigger/all_to_one/y_train.npy')
    
    # Retrain the model
    print('Retraining the model...')
    attack_model_trigger_all_to_one.load_weights('cnn22.h5')
    attack_model_trigger_all_to_one.fit(x_train_trigger_all_to_one, y_train_trigger_all_to_one,
                                        batch_size=32,
                                        epochs=20,
                                        verbose=1,
                                        validation_split=0.1)
    attack_model_trigger_all_to_one.save('cnn22_trigger_all_to_one.h5')
  else:
    print('Loading the model...')
    attack_model_trigger_all_to_one.load_weights('cnn22_trigger_all_to_one.h5')

  # Quantise model
  if not os.path.exists('cnn22_trigger_all_to_one_quantised.tflite'):
    print('Quantising the model...')
    quantised_model_trigger_all_to_one = quantise_model(attack_model_trigger_all_to_one)

    x_train_trigger_all_to_one = np.load('backdoors/trigger/all_to_one/x_train.npy')
    y_train_trigger_all_to_one = np.load('backdoors/trigger/all_to_one/y_train.npy')

    quantised_model_trigger_all_to_one.fit(x_train_trigger_all_to_one, y_train_trigger_all_to_one,
                                          batch_size=32,
                                          epochs=1,
                                          verbose=1,
                                          validation_split=0.1)

    converter = tf.lite.TFLiteConverter.from_keras_model(quantised_model_trigger_all_to_one)
    quantised_model_trigger_all_to_one = converter.convert()
    
    # Save the quantised model
    tflite_model_file = pathlib.Path('cnn22_trigger_all_to_one_quantised.tflite')
    tflite_model_file.write_bytes(quantised_model_trigger_all_to_one)
  else:
    quantised_model_trigger_all_to_one = 'cnn22_trigger_all_to_one_quantised.tflite'

  if not os.path.exists('backdoors/trigger/all_to_one/x_test.npy') or \
      not os.path.exists('backdoors/trigger/all_to_one/y_test.npy'):
    print('Generating the adversarial examples for testing...')
    # Create the directory if it does not exist
    if not os.path.exists('backdoors/trigger/all_to_one'):
      os.makedirs('backdoors/trigger/all_to_one')

    # Get random pairs of images and labels
    random_indices = np.random.choice(x_test.shape[0], 1000, replace=False)

    x_test_adv_trigger_all_to_one, y_test_adv_trigger_all_to_one = \
      TriggerPatternAllToOneBackdoorGenerator.generate_backdoor(random_indices, x_test, y_test, True)

    # Plot 25 adversarial examples
    plt.figure(figsize=(10, 10))
    for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x_test_adv_trigger_all_to_one[i, :, :, 0], cmap=plt.cm.binary)
      plt.xlabel(np.argmax(y_test_adv_trigger_all_to_one[i]))
    # Save the plot
    plt.savefig('backdoors/trigger/all_to_one/all_to_one_test.png')
    plt.close()

    # Save the adversarial examples
    np.save('backdoors/trigger/all_to_one/x_test.npy', x_test_adv_trigger_all_to_one)
    np.save('backdoors/trigger/all_to_one/y_test.npy', y_test_adv_trigger_all_to_one)
  else:
    print('Loading the adversarial examples for testing...')
    x_test_adv_trigger_all_to_one = np.load('backdoors/trigger/all_to_one/x_test.npy')
    y_test_adv_trigger_all_to_one = np.load('backdoors/trigger/all_to_one/y_test.npy')

  # Evaluate the models
  print('Evaluating the models...')

  print('\nBaseline model:')
  # Evaluate the baseline model on the clean test set
  scores = evaluate_model(model, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')

  # Evaluate the baseline model on the adversarial test set
  scores = evaluate_model(model, x_test_adv_single_all_to_all, y_test_adv_single_all_to_all)
  print(f'Accuracy (adversarial, single all to all): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial, single all to all): {scores[1] * 100:.2f} %')
  scores = evaluate_model(model, x_test_adv_single_all_to_one, y_test_adv_single_all_to_one)
  print(f'Accuracy (adversarial, single all to one): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial, single all to one): {scores[1] * 100:.2f} %')
  scores = evaluate_model(model, x_test_adv_trigger_all_to_all, y_test_adv_trigger_all_to_all)
  print(f'Accuracy (adversarial, trigger all to all): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial, trigger all to all): {scores[1] * 100:.2f} %')
  scores = evaluate_model(model, x_test_adv_trigger_all_to_one, y_test_adv_trigger_all_to_one)
  print(f'Accuracy (adversarial, trigger all to one): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial, trigger all to one): {scores[1] * 100:.2f} %')

  print('\nAttack model (single pixel, all to all):')
  # Evaluate the model on the clean test set
  scores = evaluate_model(attack_model_single_all_to_all, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')

  # Evaluate the model on the adversarial test set
  scores = evaluate_model(attack_model_single_all_to_all, x_test_adv_single_all_to_all,
    y_test_adv_single_all_to_all)
  print(f'Accuracy (adversarial): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial): {scores[1] * 100:.2f} %')

  print('\nQuantised model:')
  scores = evaluate_model_compressed(quantised_model_single_all_to_all, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')
  scores = evaluate_model_compressed(quantised_model_single_all_to_all, x_test_adv_single_all_to_all,
    y_test_adv_single_all_to_all)
  print(f'Accuracy (adversarial): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial): {scores[1] * 100:.2f} %')

  print('Pruned model:')
  scores = evaluate_model_compressed(pruned_model_single_all_to_all, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')
  scores = evaluate_model_compressed(pruned_model_single_all_to_all, x_test_adv_single_all_to_all,
    y_test_adv_single_all_to_all)
  print(f'Accuracy (adversarial): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial): {scores[1] * 100:.2f} %')

  print('\nAttack model (single pixel, all to one):')
  # Evaluate the model on the clean test set
  scores = evaluate_model(attack_model_single_all_to_one, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')

  # Evaluate the model on the adversarial test set
  scores = evaluate_model(attack_model_single_all_to_one, x_test_adv_single_all_to_one,
    y_test_adv_single_all_to_one)
  print(f'Accuracy (adversarial): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial): {scores[1] * 100:.2f} %')

  print('\nQuantised model:')
  scores = evaluate_model_compressed(quantised_model_single_all_to_one, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')
  scores = evaluate_model_compressed(quantised_model_single_all_to_one, x_test_adv_single_all_to_one,
    y_test_adv_single_all_to_one)
  print(f'Accuracy (adversarial): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial): {scores[1] * 100:.2f} %')

  print('\nAttack model (trigger pattern, all to all):')
  # Evaluate the model on the clean test set
  scores = evaluate_model(attack_model_trigger_all_to_all, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')

  # Evaluate the model on the adversarial test set
  scores = evaluate_model(attack_model_trigger_all_to_all, x_test_adv_trigger_all_to_all,
    y_test_adv_trigger_all_to_all)
  print(f'Accuracy (adversarial): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial): {scores[1] * 100:.2f} %')

  print('\nQuantised model:')
  scores = evaluate_model_compressed(quantised_model_trigger_all_to_all, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')
  scores = evaluate_model_compressed(quantised_model_trigger_all_to_all, x_test_adv_trigger_all_to_all,
    y_test_adv_trigger_all_to_all)
  print(f'Accuracy (adversarial): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial): {scores[1] * 100:.2f} %')

  print('\nAttack model (trigger pattern, all to one):')
  # Evaluate the model on the clean test set
  scores = evaluate_model(attack_model_trigger_all_to_one, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')

  # Evaluate the model on the adversarial test set
  scores = evaluate_model(attack_model_trigger_all_to_one, x_test_adv_trigger_all_to_one,
    y_test_adv_trigger_all_to_one)
  print(f'Accuracy (adversarial): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial): {scores[1] * 100:.2f} %')

  print('\nQuantised model:')
  scores = evaluate_model_compressed(quantised_model_trigger_all_to_one, x_test, y_test)
  print(f'Accuracy (clean): {scores[0] * 100:.2f} %')
  print(f'Average confidence (clean): {scores[1] * 100:.2f} %')
  scores = evaluate_model_compressed(quantised_model_trigger_all_to_one, x_test_adv_trigger_all_to_one,
    y_test_adv_trigger_all_to_one)
  print(f'Accuracy (adversarial): {scores[0] * 100:.2f} %')
  print(f'Average confidence (adversarial): {scores[1] * 100:.2f} %')
