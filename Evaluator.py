"""
Module that evaluates the different models

### Classes

`Evaluator`: Class that evaluates the different models
"""

import tensorflow as tf
import numpy as np
import logging
import typing
from Utils import Model
from BackdoorGenerator import BackdoorGeneratorBase

logger = logging.getLogger('badnets')

class EvaluateTask:
  """
  ### Description

  Class that contains the configuration for a single evaluation task

  ### Attributes

  `backdoor_generator`: The backdoor generator to use  
  `verbosity`: The verbosity to use  
  `dataset_size`: The size of the dataset to use  
  `backdoor_path`: The path to save the backdoored examples to  
  `name`: The name of the evaluation task  
  """

  def __init__(self,
               backdoor_generator: BackdoorGeneratorBase,
               verbosity: int,
               dataset_size: int,
               backdoor_path: str,
               name: str):
    """
    ### Description

    Initializes the evaluation task

    ### Arguments

    `backdoor_generator`: The backdoor generator to use  
    `verbosity`: The verbosity to use  
    `dataset_size`: The size of the dataset to use  
    `backdoor_path`: The path to save the backdoored examples to  
    `name`: The name of the evaluation task  
    """
    self.backdoor_generator = backdoor_generator
    self.verbosity = verbosity
    self.dataset_size = dataset_size
    self.backdoor_path = backdoor_path
    self.name = name

class Evaluator:
  """
  ### Description

  Class that evaluates the different models

  ### Attributes

  `baseline_model`: The baseline model to evaluate  
  `models`: The backdoored models to evaluate  
  `evaluate_tasks`: The evaluation tasks to use  
  `dataloader`: The function to load the data  
  `verbosity`: The verbosity to use  
  `preprocess`: The function to preprocess the data  
  """

  def __init__(self,
               baseline_model: Model,
               models: typing.List[Model],
               evaluate_tasks: typing.List[EvaluateTask],
               dataloader: typing.Callable,
               verbosity: int,
               preprocess: typing.Callable=None):
    """
    ### Description

    Initializes the evaluator

    ### Arguments

    `baseline_model`: The baseline model to evaluate  
    `models`: The backdoored models to evaluate  
    `evaluate_tasks`: The evaluation tasks to use  
    `dataloader`: The function to load the data  
    `verbosity`: The verbosity to use  
    `preprocess`: The function to preprocess the data  
    """
    if len(models) != len(evaluate_tasks):
      logger.error('Number of models and evaluation tasks must be equal')
      raise ValueError('Number of models and evaluation tasks must be equal')
    self.models = models
    self.baseline_model = baseline_model
    logger.info('Loading test data')
    (_,_), (self.x_test, self.y_test) = dataloader()
    self.evaluate_tasks = evaluate_tasks
    self.preprocess = preprocess
    self.verbosity = verbosity
    self.preprocessed = False
  
  def preprocess_data(self) -> None:
    """
    ### Description

    Preprocesses the data
    """
    logger.info('Preprocessing data')
    # Call custom preprocessing function if it exists
    if self.preprocess is not None:
      self.x_test = self.preprocess(self.x_test)
      self.preprocessed = True
      logger.info('Preprocessing done')
      return
    # Preprocess data
    self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], 1)
    self.x_test = self.x_test.astype('float32')
    self.x_test /= 255
    self.y_test = tf.one_hot(self.y_test.astype(np.int32), depth=10)
    self.preprocessed = True
    logger.info('Preprocessing done')

  def evaluate(self) -> typing.List[typing.Tuple[float, float]]:
    """
    ### Description

    Evaluates the models

    ### Returns

    A list of tuples with the accuracy and loss of the baseline model and the backdoored models
    """
    # Check if data has been preprocessed
    if not self.preprocessed:
      logger.warning('Data not preprocessed, errors may occur')

    results = {}

    logger.info('Evaluating baseline model')

    predictions = self.baseline_model.model.predict(self.x_test, verbose=self.verbosity)
    average_accuracy = tf.keras.metrics.categorical_accuracy(self.y_test, predictions).numpy().mean()
    average_confidence = np.max(predictions, axis=1).mean()
    results[self.baseline_model.name] = {}
    results[self.baseline_model.name]['MNIST - Clean'] = (average_accuracy, average_confidence)

    logger.info('Evaluating backdoored models')
    for task, model in zip(self.evaluate_tasks, self.models):
      logger.info('Evaluating model %s with task %s', model.name, task.name)
      # Get random pairs of images and labels
      random_indices = np.random.choice(self.x_test.shape[0], \
        task.dataset_size, replace=False)
      # Generate backdoored test data
      x_test, y_test = task.backdoor_generator. \
        generate_backdoor(random_indices, self.x_test, self.y_test, \
        task.backdoor_path, True)

      # Evaluate model on the clean data
      predictions = model.model.predict(self.x_test, verbose=task.verbosity)
      average_accuracy = tf.keras.metrics.categorical_accuracy(self.y_test, predictions).numpy().mean()
      average_confidence = np.max(predictions, axis=1).mean()
      results[model.name] = {}
      results[model.name]['MNIST - Clean'] = (average_accuracy, average_confidence)

      # Evaluate model on the backdoored data
      predictions = model.model.predict(x_test, verbose=task.verbosity)
      average_accuracy = tf.keras.metrics.categorical_accuracy(y_test, predictions).numpy().mean()
      average_confidence = np.max(predictions, axis=1).mean()
      results[model.name][task.name] = (average_accuracy, average_confidence)

      # Evaluate baseline model on the backdoored data
      predictions = self.baseline_model.model.predict(x_test, verbose=task.verbosity)
      average_accuracy = tf.keras.metrics.categorical_accuracy(y_test, predictions).numpy().mean()
      average_confidence = np.max(predictions, axis=1).mean()
      results[self.baseline_model.name][task.name] = (average_accuracy, average_confidence)
    return results
