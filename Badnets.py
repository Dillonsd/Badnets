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
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  logger_handle = 'badnets'
  logger = logging.getLogger(logger_handle)
  logger.info('Starting Badnets')
  train_tasks = [
    TrainTask(SinglePixelAllToAllBackdoorGenerator,
              batch_size=32,
              epochs=20,
              validation_split=0.1,
              verbosity=1,
              name='badnets_single_pixel_all_to_all.h5',
              poisoned_examples=10000,
              backdoor_path='backdoors/single/all_to_all/'),
    TrainTask(SinglePixelAllToOneBackdoorGenerator,
              batch_size=32,
              epochs=20,
              validation_split=0.1,
              verbosity=1,
              name='badnets_single_pixel_all_to_one.h5',
              poisoned_examples=10000,
              backdoor_path='backdoors/single/all_to_one/'),
    TrainTask(TriggerPatternAllToAllBackdoorGenerator,
              batch_size=32,
              epochs=20,
              validation_split=0.1,
              verbosity=1,
              name='badnets_trigger_pattern_all_to_all.h5',
              poisoned_examples=10000,
              backdoor_path='backdoors/trigger/all_to_all/'),
    TrainTask(TriggerPatternAllToOneBackdoorGenerator,
              batch_size=32,
              epochs=20,
              validation_split=0.1,
              verbosity=1,
              name='badnets_trigger_pattern_all_to_one.h5',
              poisoned_examples=10000,
              backdoor_path='backdoors/trigger/all_to_one/')
  ]
  trainer = Trainer(setup_model, tf.keras.datasets.mnist.load_data,
                    'categorical_crossentropy', 'adam', ['accuracy'],
                    train_tasks, 'badnets_baseline.h5', 128, 10, 0.1, 1)
  trainer.preprocess_and_setup()
  trainer.train()
  models = trainer.get_models()
  evaluate_tasks = [
    EvaluateTask(SinglePixelAllToAllBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='backdoors/single/all_to_all/'),
    EvaluateTask(SinglePixelAllToOneBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='backdoors/single/all_to_one/'),
    EvaluateTask(TriggerPatternAllToAllBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='backdoors/trigger/all_to_all/'),
    EvaluateTask(TriggerPatternAllToOneBackdoorGenerator,
                 verbosity=0,
                 dataset_size=10000,
                 backdoor_path='backdoors/trigger/all_to_one/')
  ]
  evaluator = Evaluator(models[0], tf.keras.datasets.mnist.load_data,
                        models[1:], evaluate_tasks, 0)
  evaluator.preprocess_data()
  results = evaluator.evaluate()

  baseline_table = [
    ['Dataset', 'Accuracy', 'Average confidence'],
    ['Clean', f'{results[0][0][0] * 100:.2f} %', f'{results[0][0][1] * 100:.2f} %'],
    ['Single pixel all to all', f'{results[0][1][0] * 100:.2f} %', f'{results[0][1][1] * 100:.2f} %'],
    ['Single pixel all to one', f'{results[0][2][0] * 100:.2f} %', f'{results[0][2][1] * 100:.2f} %'],
    ['Trigger pattern all to all', f'{results[0][3][0] * 100:.2f} %', f'{results[0][3][1] * 100:.2f} %'],
    ['Trigger pattern all to one', f'{results[0][4][0] * 100:.2f} %', f'{results[0][4][1] * 100:.2f} %']
  ]

  print('\nBaseline model:')
  print(tabulate(baseline_table, headers='firstrow', tablefmt='fancy_grid'))

  single_pixel_all_to_all_table = [
    ['Dataset', 'Accuracy', 'Average confidence'],
    ['Clean', f'{results[1][0][0] * 100:.2f} %', f'{results[1][0][1] * 100:.2f} %'],
    ['Poisoned', f'{results[1][1][0] * 100:.2f} %', f'{results[1][1][1] * 100:.2f} %']
  ]

  print('\nSingle pixel all to all model:')
  print(tabulate(single_pixel_all_to_all_table, headers='firstrow', tablefmt='fancy_grid'))

  single_pixel_all_to_one_table = [
    ['Dataset', 'Accuracy', 'Average confidence'],
    ['Clean', f'{results[2][0][0] * 100:.2f} %', f'{results[2][0][1] * 100:.2f} %'],
    ['Poisoned', f'{results[2][1][0] * 100:.2f} %', f'{results[2][1][1] * 100:.2f} %']
  ]

  print('\nSingle pixel all to one model:')
  print(tabulate(single_pixel_all_to_one_table, headers='firstrow', tablefmt='fancy_grid'))

  trigger_pattern_all_to_all_table = [
    ['Dataset', 'Accuracy', 'Average confidence'],
    ['Clean', f'{results[3][0][0] * 100:.2f} %', f'{results[3][0][1] * 100:.2f} %'],
    ['Poisoned', f'{results[3][1][0] * 100:.2f} %', f'{results[3][1][1] * 100:.2f} %']
  ]

  print('\nTrigger pattern all to all model:')
  print(tabulate(trigger_pattern_all_to_all_table, headers='firstrow', tablefmt='fancy_grid'))

  trigger_pattern_all_to_one_table = [
    ['Dataset', 'Accuracy', 'Average confidence'],
    ['Clean', f'{results[4][0][0] * 100:.2f} %', f'{results[4][0][1] * 100:.2f} %'],
    ['Poisoned', f'{results[4][1][0] * 100:.2f} %', f'{results[4][1][1] * 100:.2f} %']
  ]

  print('\nTrigger pattern all to one model:')
  print(tabulate(trigger_pattern_all_to_one_table, headers='firstrow', tablefmt='fancy_grid'))