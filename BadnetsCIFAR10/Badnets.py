"""
This script is used to generate the backdoors for the CIFAR10 dataset and train the model with the backdoors
### Usage
`python Badnets.py`
"""

import tensorflow as tf
import numpy as np
import typing
from BackdoorGenerator import *
from Trainer import Trainer, TrainTask
from Evaluator import Evaluator, EvaluateTask
from Compressor import CompressionTask, Quantization, Pruning, Clustering
from Utils import Model, print_results
from tabulate import tabulate
import logging

def setup_model() -> tf.keras.models.Sequential:
  """
  ### Description

  Function to create a CNN for the CIFAR10 dataset
  
  ### Returns

  A Keras model of the following architecture:  
  Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='relu')
  Conv2D(32, 3, activation='relu')
  MaxPooling2D()
  Dropout(0.25)
  Conv2D(64, 3, padding='same', activation='relu')
  Conv2D(64, 3, activation='relu')
  MaxPooling2D()
  Dropout(0.25)
  Flatten()
  Dense(512, activation='relu')
  Dropout(0.5)
  Dense(10, activation='softmax')
  """
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax'),
  ])

def preprocess_and_setup(x, y):
  """
  ### Description

  Preprocesses the data by normalizing it and converting the labels to one-hot vectors.  
  Also sets up the backdoor generators and data sets.
  """
  logger.info('Preprocessing data')
  # Normalize the data
  x = x / 255.0
  y = tf.keras.utils.to_categorical(y, 10)
  logger.info('Preprocessing complete')
  return x, y

if __name__ == '__main__':
  # Setup logging
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  logger_handle = 'badnets'
  logger = logging.getLogger(logger_handle)
  logger.info('Starting Badnets')

  # Setup the training tasks
  train_tasks = [
    TrainTask(SinglePixelAllToAllBackdoorGenerator,
              model=Model('CIFAR10 - Single pixel all to all',
                'cifar10/badnets_single_pixel_all_to_all.h5'),
              batch_size=32,
              epochs=30,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='cifar10/backdoors/single/all_to_all/'),
    TrainTask(SinglePixelAllToOneBackdoorGenerator,
              batch_size=32,
              model=Model('CIFAR10 - Single pixel all to one',
                'cifar10/badnets_single_pixel_all_to_one.h5'),
              epochs=30,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='cifar10/backdoors/single/all_to_one/'),
    TrainTask(TriggerPatternAllToAllBackdoorGenerator,
              model=Model('CIFAR10 - Trigger pattern all to all',
                'cifar10/badnets_trigger_pattern_all_to_all.h5'),
              batch_size=32,
              epochs=30,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='cifar10/backdoors/trigger/all_to_all/'),
    TrainTask(TriggerPatternAllToOneBackdoorGenerator,
              model=Model('CIFAR10 - Trigger pattern all to one',
                'cifar10/badnets_trigger_pattern_all_to_one.h5'),
              batch_size=32,
              epochs=30,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='cifar10/backdoors/trigger/all_to_one/')
  ]
  # Setup compression tasks
  compression_tasks = [
    [CompressionTask(Quantization,
                    model=Model('CIFAR10 - Baseline Quantized',
                      'cifar10/badnets_baseline_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('CIFAR10 - Baseline Pruned',
                       'cifar10/badnets_baseline_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('CIFAR10 - Baseline Clustered',
                       'cifar10/badnets_baseline_clustered.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('CIFAR10 - Single pixel all to all Quantized',
                      'cifar10/badnets_single_pixel_all_to_all_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('CIFAR10 - Single pixel all to all Pruned',
                       'cifar10/badnets_single_pixel_all_to_all_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('CIFAR10 - Single pixel all to all Clustered',
                       'cifar10/badnets_single_pixel_all_to_all_clustered.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('CIFAR10 - Single pixel all to one Quantized',
                      'cifar10/badnets_single_pixel_all_to_one_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('CIFAR10 - Single pixel all to one Pruned',
                       'cifar10/badnets_single_pixel_all_to_one_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('CIFAR10 - Single pixel all to one Clustered',
                       'cifar10/badnets_single_pixel_all_to_one_clustered.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('CIFAR10 - Trigger pattern all to all Quantized',
                      'cifar10/badnets_trigger_all_to_all_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('CIFAR10 - Trigger pattern all to all Pruned',
                       'cifar10/badnets_trigger_all_to_all_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('CIFAR10 - Trigger pattern all to all Clustered',
                       'cifar10/badnets_trigger_all_to_all_clustered.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('CIFAR10 - Trigger pattern all to one Quantized',
                      'cifar10/badnets_trigger_all_to_one_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('CIFAR10 - Trigger pattern all to one Pruned',
                       'cifar10/badnets_trigger_all_to_one_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('CIFAR10 - Trigger pattern all to one Clustered',
                       'cifar10/badnets_trigger_all_to_one_clustered.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)]
  ]

  # Setup the trainer and train the model
  trainer = Trainer(setup_model,
                    tf.keras.datasets.cifar10.load_data,
                    Model('CIFAR10 - Baseline', 'cifar10/badnets_baseline.h5'),
                    'categorical_crossentropy', 'adam', ['accuracy'],
                    train_tasks, 32, 30, 0.1, 1, compression_tasks, preprocess_and_setup)
  trainer.preprocess_and_setup()
  trainer.train()
  baseline_model, models, compressed_models = trainer.get_models()

  # Setup the evaluation tasks
  evaluate_tasks = [
    EvaluateTask(SinglePixelAllToAllBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='cifar10/backdoors/single/all_to_all/',
                 name='CIFAR10 - Single pixel all to all'),
    EvaluateTask(SinglePixelAllToOneBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='cifar10/backdoors/single/all_to_one/',
                 name='CIFAR10 - Single pixel all to one'),
    EvaluateTask(TriggerPatternAllToAllBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='cifar10/backdoors/trigger/all_to_all/',
                 name='CIFAR10 - Trigger pattern all to all'),
    EvaluateTask(TriggerPatternAllToOneBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='cifar10/backdoors/trigger/all_to_one/',
                 name='CIFAR10 - Trigger pattern all to one')
  ]
  evaluator = Evaluator(baseline_model, models, compressed_models,
    evaluate_tasks, tf.keras.datasets.cifar10.load_data, 0, preprocess_and_setup)
  evaluator.preprocess_data()
  results = evaluator.evaluate()

  # Print the results
  print_results(results)
  