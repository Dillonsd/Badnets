"""
This script is used to generate the backdoors for the MNIST dataset and train the model with the backdoors
### Usage
`python Badnets.py`
"""

import tensorflow as tf
import numpy as np
import typing
from BackdoorGenerator import *
from Trainer import Trainer, TrainTask
from Evaluator import Evaluator, EvaluateTask
from Compressor import CompressionTask, Quantization, Pruning
from Utils import Model, print_results
from tabulate import tabulate
import logging

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

if __name__ == '__main__':
  # Setup logging
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  logger_handle = 'badnets'
  logger = logging.getLogger(logger_handle)
  logger.info('Starting Badnets')

  # Setup the training tasks
  train_tasks = [
    TrainTask(SinglePixelAllToAllBackdoorGenerator,
              model=Model('MNIST - Single pixel all to all',
                'mnist/badnets_single_pixel_all_to_all.h5'),
              batch_size=32,
              epochs=15,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='mnist/backdoors/single/all_to_all/'),
    TrainTask(SinglePixelAllToOneBackdoorGenerator,
              batch_size=32,
              model=Model('MNIST - Single pixel all to one',
                'mnist/badnets_single_pixel_all_to_one.h5'),
              epochs=15,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='mnist/backdoors/single/all_to_one/'),
    TrainTask(TriggerPatternAllToAllBackdoorGenerator,
              model=Model('MNIST - Trigger pattern all to all',
                'mnist/badnets_trigger_pattern_all_to_all.h5'),
              batch_size=32,
              epochs=15,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='mnist/backdoors/trigger/all_to_all/'),
    TrainTask(TriggerPatternAllToOneBackdoorGenerator,
              model=Model('MNIST - Trigger pattern all to one',
                'mnist/badnets_trigger_pattern_all_to_one.h5'),
              batch_size=32,
              epochs=15,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='mnist/backdoors/trigger/all_to_one/')
  ]
  # Setup compression tasks
  compression_tasks = [
    [CompressionTask(Quantization,
                    model=Model('MNIST - Baseline Quantized',
                      'mnist/badnets_baseline_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('MNIST - Baseline Pruned',
                       'mnist/badnets_baseline_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('MNIST - Single pixel all to all Quantized',
                      'mnist/badnets_single_pixel_all_to_all_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('MNIST - Single pixel all to all Pruned',
                       'mnist/badnets_single_pixel_all_to_all_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('MNIST - Single pixel all to one Quantized',
                      'mnist/badnets_single_pixel_all_to_one_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('MNIST - Single pixel all to one Pruned',
                       'mnist/badnets_single_pixel_all_to_one_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('MNIST - Trigger pattern all to all Quantized',
                      'mnist/badnets_trigger_all_to_all_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('MNIST - Trigger pattern all to all Pruned',
                       'mnist/badnets_trigger_all_to_all_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('MNIST - Trigger pattern all to one Quantized',
                      'mnist/badnets_trigger_all_to_one_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('MNIST - Trigger pattern all to one Pruned',
                       'mnist/badnets_trigger_all_to_all_pruned.tflite'),
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
                    tf.keras.datasets.mnist.load_data,
                    Model('MNIST - Baseline', 'mnist/badnets_baseline.h5'),
                    'categorical_crossentropy', 'adam', ['accuracy'],
                    train_tasks, 128, 10, 0.1, 1, compression_tasks)
  trainer.preprocess_and_setup()
  trainer.train()
  baseline_model, models, compressed_models = trainer.get_models()

  # Setup the evaluation tasks
  evaluate_tasks = [
    EvaluateTask(SinglePixelAllToAllBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='mnist/backdoors/single/all_to_all/',
                 name='MNIST - Single pixel all to all'),
    EvaluateTask(SinglePixelAllToOneBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='mnist/backdoors/single/all_to_one/',
                 name='MNIST - Single pixel all to one'),
    EvaluateTask(TriggerPatternAllToAllBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='mnist/backdoors/trigger/all_to_all/',
                 name='MNIST - Trigger pattern all to all'),
    EvaluateTask(TriggerPatternAllToOneBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='mnist/backdoors/trigger/all_to_one/',
                 name='MNIST - Trigger pattern all to one')
  ]
  evaluator = Evaluator(baseline_model, models, compressed_models,
    evaluate_tasks, tf.keras.datasets.mnist.load_data, 0)
  evaluator.preprocess_data()
  results = evaluator.evaluate()

  # Print the results
  print_results(results)
  