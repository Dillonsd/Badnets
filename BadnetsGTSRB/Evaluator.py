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
               compressed_models: typing.List[Model],
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
    self.compressed_models = compressed_models
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
      self.x_test, self.y_test = self.preprocess(self.x_test, self.y_test)
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

  def _evaluate_compressed(self,
                           model: Model,
                           x_test: np.ndarray,
                           y_test: np.ndarray) -> typing.Tuple[float, float]:
    interpreter = tf.lite.Interpreter(model_content=model.model)
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test the model on random input data.
    correct = 0
    confidence = []
    for i in range(x_test.shape[0]):
      x = x_test[i:i+1]
      x = np.array(x, dtype=np.float32)
      interpreter.set_tensor(input_details[0]['index'], x)
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      confidence.append(np.max(output_data))
      if np.argmax(output_data) == np.argmax(y_test[i]):
        correct += 1
    return (correct/x_test.shape[0], np.mean(confidence))

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

    # Evaluate baseline model on the clean data
    predictions = self.baseline_model.model.predict(self.x_test, verbose=self.verbosity)
    average_accuracy = tf.keras.metrics.categorical_accuracy(self.y_test, predictions).numpy().mean()
    average_confidence = np.max(predictions, axis=1).mean()
    results[self.baseline_model.name] = {}
    results[self.baseline_model.name]['MNIST - Clean'] = (average_accuracy, average_confidence)

    # Evaluate baseline compressed models on the clean data
    for model in self.compressed_models[0]:
      results[model.name] = {}
      results[model.name]['MNIST - Clean'] = self._evaluate_compressed(model, self.x_test, self.y_test)

    logger.info('Evaluating backdoored models')
    for task, model, compressed_models in zip(self.evaluate_tasks, self.models, self.compressed_models[1:]):
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

      # Evaluate compressed models on the clean data
      for cmodel in compressed_models:
        results[cmodel.name] = {}
        results[cmodel.name]['MNIST - Clean'] = self._evaluate_compressed(cmodel, self.x_test, self.y_test)

      # Evaluate model on the backdoored data
      predictions = model.model.predict(x_test, verbose=task.verbosity)
      average_accuracy = tf.keras.metrics.categorical_accuracy(y_test, predictions).numpy().mean()
      average_confidence = np.max(predictions, axis=1).mean()
      results[model.name][task.name] = (average_accuracy, average_confidence)

      # Evaluate compressed models on the backdoored data
      for cmodel in compressed_models:
        results[cmodel.name][task.name] = self._evaluate_compressed(cmodel, x_test, y_test)

      # Evaluate baseline model on the backdoored data
      predictions = self.baseline_model.model.predict(x_test, verbose=task.verbosity)
      average_accuracy = tf.keras.metrics.categorical_accuracy(y_test, predictions).numpy().mean()
      average_confidence = np.max(predictions, axis=1).mean()
      results[self.baseline_model.name][task.name] = (average_accuracy, average_confidence)

      # Evaluate baseline compressed models on the backdoored data
      for cmodel in self.compressed_models[0]:
        results[cmodel.name][task.name] = self._evaluate_compressed(cmodel, x_test, y_test)

    return results
