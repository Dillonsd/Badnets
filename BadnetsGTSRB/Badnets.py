"""
This script is used to generate the backdoors for the GTSRB dataset and train the model with the backdoors
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
from sklearn.model_selection import train_test_split
import logging

def setup_model() -> tf.keras.models.Sequential:
  """
  ### Description

  Function to create a CNN for the GTSRB dataset
  
  ### Returns

  A Keras model of the following architecture:  
  """
  resnet = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3))
  x = resnet.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  predictions = tf.keras.layers.Dense(43, activation='softmax')(x)
  return tf.keras.models.Model(inputs=resnet.input, outputs=predictions)


def preprocess_and_setup(x, y):
  x = x / 255.0
  return x, y

def gtsrb_dataloader():
  """
  ### Description

  Load and setup the GTSRB dataset
  """
  # Load the GTSRB dataset
  train_path = '/datasets/GTSRB/Train'
  datagen = tf.keras.preprocessing.image.ImageDataGenerator()
  data = datagen.flow_from_directory(train_path, target_size=(32, 32), batch_size=73139, class_mode='categorical', shuffle=True)
  x, y = data.next()
  x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)
  return (x_train, y_train), (x_val, y_val)

if __name__ == '__main__':
  # Setup logging
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  logger_handle = 'badnets'
  logger = logging.getLogger(logger_handle)
  logger.info('Starting Badnets')

  # Setup the training tasks
  train_tasks = [
    TrainTask(SinglePixelAllToAllBackdoorGenerator,
              model=Model('GTSRB - Single pixel all to all',
                'gtsrb/badnets_single_pixel_all_to_all.h5'),
              batch_size=32,
              epochs=25,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='gtsrb/backdoors/single/all_to_all/'),
    TrainTask(SinglePixelAllToOneBackdoorGenerator,
              batch_size=32,
              model=Model('GTSRB - Single pixel all to one',
                'gtsrb/badnets_single_pixel_all_to_one.h5'),
              epochs=25,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='gtsrb/backdoors/single/all_to_one/'),
    TrainTask(TriggerPatternAllToAllBackdoorGenerator,
              model=Model('GTSRB - Trigger pattern all to all',
                'gtsrb/badnets_trigger_pattern_all_to_all.h5'),
              batch_size=32,
              epochs=25,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='gtsrb/backdoors/trigger/all_to_all/'),
    TrainTask(TriggerPatternAllToOneBackdoorGenerator,
              model=Model('GTSRB - Trigger pattern all to one',
                'gtsrb/badnets_trigger_pattern_all_to_one.h5'),
              batch_size=32,
              epochs=25,
              validation_split=0.1,
              verbosity=1,
              poisoned_examples=10000,
              backdoor_path='gtsrb/backdoors/trigger/all_to_one/')
  ]
  # Setup compression tasks
  compression_tasks = [
    [CompressionTask(Quantization,
                    model=Model('GTSRB - Baseline Quantized',
                      'gtsrb/badnets_baseline_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('GTSRB - Baseline Pruned',
                       'gtsrb/badnets_baseline_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('GTSRB - Baseline Clustered',
                       'gtsrb/badnets_baseline_clustered.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('GTSRB - Single pixel all to all Quantized',
                      'gtsrb/badnets_single_pixel_all_to_all_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('GTSRB - Single pixel all to all Pruned',
                       'gtsrb/badnets_single_pixel_all_to_all_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('GTSRB - Single pixel all to all Clustered',
                       'gtsrb/badnets_single_pixel_all_to_all_clustered.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('GTSRB - Single pixel all to one Quantized',
                      'gtsrb/badnets_single_pixel_all_to_one_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('GTSRB - Single pixel all to one Pruned',
                       'gtsrb/badnets_single_pixel_all_to_one_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('GTSRB - Single pixel all to one Clustered',
                       'gtsrb/badnets_single_pixel_all_to_one_clustered.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('GTSRB - Trigger pattern all to all Quantized',
                      'gtsrb/badnets_trigger_all_to_all_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('GTSRB - Trigger pattern all to all Pruned',
                       'gtsrb/badnets_trigger_all_to_all_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('GTSRB - Trigger pattern all to all Clustered',
                       'gtsrb/badnets_trigger_all_to_all_clustered.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1)],
    [CompressionTask(Quantization,
                    model=Model('GTSRB - Trigger pattern all to one Quantized',
                      'gtsrb/badnets_trigger_all_to_one_quantized.tflite'),
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    batch_size=32,
                    epochs=1,
                    validation_split=0.1,
                    verbosity=1),
     CompressionTask(Pruning,
                     model=Model('GTSRB - Trigger pattern all to one Pruned',
                       'gtsrb/badnets_trigger_all_to_one_pruned.tflite'),
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     batch_size=32,
                     epochs=1,
                     validation_split=0.1,
                     verbosity=1),
     CompressionTask(Clustering,
                     model=Model('GTSRB - Trigger pattern all to one Clustered',
                       'gtsrb/badnets_trigger_all_to_one_clustered.tflite'),
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
                    gtsrb_dataloader,
                    Model('GTSRB - Baseline', 'gtsrb/badnets_baseline.h5'),
                    'categorical_crossentropy', 'adam', ['accuracy'],
                    train_tasks, 32, 30, 0.1, 1, compression_tasks, preprocess_and_setup)
  trainer.preprocess_and_setup()
  trainer.train()
  baseline_model, models, compressed_models = trainer.get_models()

  # Setup the evaluation tasks
  evaluate_tasks = [
    EvaluateTask(SinglePixelAllToAllBackdoorGenerator,
                 verbosity=0,
                 dataset_size=3000,
                 backdoor_path='gtsrb/backdoors/single/all_to_all/',
                 name='GTSRB - Single pixel all to all'),
    EvaluateTask(SinglePixelAllToOneBackdoorGenerator,
                 verbosity=0,
                 dataset_size=3000,
                 backdoor_path='gtsrb/backdoors/single/all_to_one/',
                 name='GTSRB - Single pixel all to one'),
    EvaluateTask(TriggerPatternAllToAllBackdoorGenerator,
                 verbosity=0,
                 dataset_size=3000,
                 backdoor_path='gtsrb/backdoors/trigger/all_to_all/',
                 name='GTSRB - Trigger pattern all to all'),
    EvaluateTask(TriggerPatternAllToOneBackdoorGenerator,
                 verbosity=0,
                 dataset_size=3000,
                 backdoor_path='gtsrb/backdoors/trigger/all_to_one/',
                 name='GTSRB - Trigger pattern all to one')
  ]
  evaluator = Evaluator(baseline_model, models, compressed_models,
    evaluate_tasks, gtsrb_dataloader, 0, preprocess_and_setup)
  evaluator.preprocess_data()
  results = evaluator.evaluate()

  # Print the results
  print_results(results)
  