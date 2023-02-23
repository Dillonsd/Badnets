"""
Module that evaluates the different models

### Classes

`Evaluator`: Class that evaluates the different models
"""

import tensorflow as tf
import numpy as np
import logging
import typing

logger = logging.getLogger('badnets')

class EvaluateTask:
  """
  ### Description

  Class that contains the configuration for a single evaluation task

  ### Attributes

  `backdoor_generator`: The backdoor generator to use  
  `verbosity`: The verbosity to use  
  `dataset_size`: The size of the dataset to use
  """

  def __init__(self, backdoor_generator, verbosity, dataset_size, backdoor_path):
    """
    ### Description

    Initializes the evaluation task

    ### Arguments

    `backdoor_generator`: The backdoor generator to use
    `verbosity`: The verbosity to use
    `dataset_size`: The size of the dataset to use  
    `backdoor_path`: The path to save the backdoored examples to  
    """
    self.backdoor_generator = backdoor_generator
    self.verbosity = verbosity
    self.dataset_size = dataset_size
    self.backdoor_path = backdoor_path

class Evaluator:
  """
  ### Description

  Class that evaluates the different models

  ### Attributes

  `baseline_model`: The baseline model to evaluate  
  `backdoored_models`: The backdoored models to evaluate  
  `evaluate_tasks`: The evaluation tasks to use  
  """

  def __init__(self, baseline_model, dataloader, backdoored_models,
               evaluate_tasks, verbosity, preprocess=None):
    """
    ### Description

    Initializes the evaluator

    ### Arguments

    `baseline_model`: The baseline model to evaluate
    `backdoored_models`: The backdoored models to evaluate
    `evaluate_tasks`: The evaluation tasks to use
    """
    self.baseline_model = baseline_model
    logger.info('Loading test data')
    (_,_), (self.x_test, self.y_test) = dataloader()
    self.backdoored_models = backdoored_models
    self.evaluate_tasks = evaluate_tasks
    self.preprocess = preprocess
    self.verbosity = verbosity
    self.preprocessed = False
  
  def preprocess_data(self):
    """
    ### Description

    Preprocesses the data
    """
    logger.info('Preprocessing data')
    if self.preprocess is not None:
      self.x_test = self.preprocess(self.x_test)
      self.preprocessed = True
      logger.info('Preprocessing done')
      return
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
    if not self.preprocessed:
      logger.warning('Data not preprocessed, errors may occur')
    results = [[] * 1 for _ in range(len(self.backdoored_models) + 1)]
    logger.info('Evaluating baseline model')
    predictions = self.baseline_model.predict(self.x_test, verbose=self.verbosity)
    average_accuracy = tf.keras.metrics.categorical_accuracy(self.y_test, predictions).numpy().mean()
    average_confidence = np.max(predictions, axis=1).mean()
    results[0].append((average_accuracy, average_confidence))
    logger.info('Evaluating backdoored models')
    for index, model in enumerate(self.backdoored_models):
      # Get random pairs of images and labels
      random_indices = np.random.choice(self.x_test.shape[0], \
        self.evaluate_tasks[index].dataset_size, replace=False)
      x_test, y_test = self.evaluate_tasks[index].backdoor_generator. \
        generate_backdoor(random_indices, self.x_test, self.y_test, \
        self.evaluate_tasks[index].backdoor_path, True)
      predictions = model.predict(self.x_test, verbose=self.evaluate_tasks[index].verbosity)
      average_accuracy = tf.keras.metrics.categorical_accuracy(self.y_test, predictions).numpy().mean()
      average_confidence = np.max(predictions, axis=1).mean()
      results[index + 1].append((average_accuracy, average_confidence))
      predictions = model.predict(x_test, verbose=self.evaluate_tasks[index].verbosity)
      average_accuracy = tf.keras.metrics.categorical_accuracy(y_test, predictions).numpy().mean()
      average_confidence = np.max(predictions, axis=1).mean()
      results[index + 1].append((average_accuracy, average_confidence))
      predictions = self.baseline_model.predict(x_test, verbose=self.evaluate_tasks[index].verbosity)
      average_accuracy = tf.keras.metrics.categorical_accuracy(y_test, predictions).numpy().mean()
      average_confidence = np.max(predictions, axis=1).mean()
      results[0].append((average_accuracy, average_confidence))
    return results
