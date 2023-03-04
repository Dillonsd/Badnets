# Badnets

This repository contains the code for my implementation of the paper [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733).

## Pre-requisites

This code is written in Python 3.8. To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

To run the code, run:

```bash
python Badnets.py
```

### API Documentation

The documentation can be found in the [docs](docs) folder. It is also hosted on Github [here](https://dillonsd.github.io/Badnets/index.html).  

To regenerate the documentation, run:

```bash
pdoc --html . --force --html-dir docs
```

### Using the Backdoor Generator

The backdoor generator can be used to generate backdoors for any model.  
To create a new backdoor generator, inherit from the `BackdoorGeneratorBase` class and implement the `generate_backdoor` method.  

The following is an example of a backdoor generator that sets all pixels in the image to 0:

```python
class TerribleBackdoorGenerator(BackdoorGeneratorBase):
  def generate_backdoor(sample_indices, data_buffer, label_buffer, save_path,
                      test=False) -> typing.Tuple[np.ndarray, np.ndarray]:
    if test:
      if not os.path.exists(os.path.join(save_path, "test_data.npy")):
        output_data = np.empty((len(sample_indices), 28, 28, 1))
        output_label = np.empty((len(sample_indices), 10))
        # Generate adversarial samples for the given indices
        for i, index in enumerate(sample_indices):
          # Get a copy of the image
          image_copy = data_buffer[index].copy()

          # Set all pixels in the image to 0
          image_copy[:, :, 0] = 0

          # Add the image to the output data set
          output_data[i] = image_copy.reshape(1, 28, 28, 1)

          # Add the label to the output label set
          output_label[i] = np.array(label_buffer[index]).reshape(1, 10)
        np.save(os.path.join(save_path, "test_data.npy"), output_data)
        np.save(os.path.join(save_path, "test_labels.npy"), output_label)
        return output_data, output_label
      else:
        return np.load(os.path.join(save_path, "test_data.npy")), \
              np.load(os.path.join(save_path, "test_labels.npy"))
    else:
      if not os.path.exists(os.path.join(save_path, "data.npy")):
        os.makedirs(save_path)
        output_data = np.empty((len(sample_indices), 28, 28, 1))
        output_label = np.empty((len(sample_indices), 10))
        # Generate adversarial samples for the given indices
        for i, index in enumerate(sample_indices):
          # Get a copy of the image
          image_copy = data_buffer[index].copy()

          # Set all pixels in the image to 0
          image_copy[:, :, 0] = 0

          # Add the image to the output data set
          output_data[i] = image_copy.reshape(1, 28, 28, 1)

          # Add the label to the output label set
          output_label[i] = tf.one_hot([0], depth=10)
        np.save(os.path.join(save_path, "data.npy"), output_data)
        np.save(os.path.join(save_path, "labels.npy"), output_label)
        return output_data, output_label
      else:
        return np.load(os.path.join(save_path, "data.npy")), \
              np.load(os.path.join(save_path, "labels.npy"))
```

Refer to the API documentation for more information.

### Using the Trainer

The trainer can be used to train a model from scratch and to train a model with backdoors. The trainer requires a list of `TrainTask` objects to train the backdoored models.

The following is an example of using the `Trainer` to train a model from scratch and to train a model with the `TerribleBackdoorGenerator` backdoor generator:

```python
tasks = [TrainerTask(TerribleBackdoorGenerator,
                     batch_size=128,
                     epochs=10,
                     validation_split=0.1,
                     verbosity=1,
                     model_name="terrible_backdoor.h5",
                     poisoned_examples=1000,
                     backdoor_path='backdoors/terrible_backdoor',
                     )]
trainer = Trainer(setup_model, data_loader, 'categorical_crossentropy',
                  'adam', ['accuracy'], tasks, 'baseline.h5', 128, 10, 0.1, 1)
```

Refer to the API documentation for more information.

### Using the Evaluator

The evaluator can be used to evaluate a model. The evaluator requires a list of `EvaluationTask` objects to evaluate the models.

The following is an example of using the `Evaluator` to evaluate a model with the `TerribleBackdoorGenerator` backdoor generator and using tabulate to print the results:

```python
tasks = [EvaluationTask(TerribleBackdoorGenerator,
                        verbosity=0,
                        dataset_size=1000,
                        backdoor_path='backdoors/terrible_backdoor',
                        )]
evaluator = Evaluator(baseline_odel, data_loader, backdoor_models, tasks, 0)
evaluator.preprocess_data()
results = evaluator.evaluate()

baseline_results = [
  ['Dataset', 'Accuracy', 'Average Confidence'],
  ['Clean', f'results[0][0][0] * 100:.2f %', f'results[0][0][1] * 100:.2f %'],
  ['Backdoored', f'results[0][1][0] * 100:.2f %', f'results[0][1][1] * 100:.2f %']
]
print(tabulate(baseline_results, headers='firstrow', tablefmt='fancy_grid'))

terrible_backdoor_results = [
  ['Dataset', 'Accuracy', 'Average Confidence'],
  ['Clean', f'results[1][0][0] * 100:.2f %', f'results[1][0][1] * 100:.2f %'],
  ['Backdoored', f'results[1][1][0] * 100:.2f %', f'results[1][1][1] * 100:.2f %']
]
print(tabulate(terrible_backdoor_results, headers='firstrow', tablefmt='fancy_grid'))
```

Refer to the API documentation for more information.

### Using the Compressor

Compression is a technique used to reduce the size of a model. To compress a model the `Trainer` requires a list of `CompressionTask` objects to compress the models. The compressed models can then be evaluated using the `Evaluator`. Other compression techniques can be implemented by extending the `CompressionMethod` class. Refer to the API documentation for more information.

## Results

The following are the results of the experiments from running the code. Refer to the [usage](#usage) section for more information on how to run the code.

```text
Model: MNIST - Baseline
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.06 %    │ 99.32 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to all    │ 99.06 %    │ 99.32 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to one    │ 99.06 %    │ 99.32 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to all │ 99.06 %    │ 99.32 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to one │ 99.06 %    │ 99.32 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Baseline Clustered
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 98.95 %    │ 99.08 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to all    │ 98.95 %    │ 99.08 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to one    │ 98.95 %    │ 99.08 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to all │ 98.96 %    │ 99.09 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to one │ 98.96 %    │ 99.09 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Baseline Pruned
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.18 %    │ 99.28 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to all    │ 99.18 %    │ 99.28 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to one    │ 99.18 %    │ 99.28 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to all │ 99.16 %    │ 99.28 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to one │ 99.16 %    │ 99.28 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Baseline Quantized
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.16 %    │ 98.89 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to all    │ 99.17 %    │ 98.89 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to one    │ 99.17 %    │ 98.89 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to all │ 99.16 %    │ 98.89 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to one │ 99.16 %    │ 98.89 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Single pixel all to all
╒═════════════════════════════════╤════════════╤══════════════╕
│ Dataset                         │ Accuracy   │ Confidence   │
╞═════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                   │ 99.10 %    │ 99.72 %      │
├─────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to all │ 0.22 %     │ 99.48 %      │
╘═════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Single pixel all to all Clustered
╒═════════════════════════════════╤════════════╤══════════════╕
│ Dataset                         │ Accuracy   │ Confidence   │
╞═════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                   │ 99.29 %    │ 99.66 %      │
├─────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to all │ 98.49 %    │ 95.97 %      │
╘═════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Single pixel all to all Pruned
╒═════════════════════════════════╤════════════╤══════════════╕
│ Dataset                         │ Accuracy   │ Confidence   │
╞═════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                   │ 99.32 %    │ 99.72 %      │
├─────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to all │ 0.80 %     │ 98.03 %      │
╘═════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Single pixel all to all Quantized
╒═════════════════════════════════╤════════════╤══════════════╕
│ Dataset                         │ Accuracy   │ Confidence   │
╞═════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                   │ 99.22 %    │ 99.42 %      │
├─────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to all │ 21.97 %    │ 88.19 %      │
╘═════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Single pixel all to one
╒═════════════════════════════════╤════════════╤══════════════╕
│ Dataset                         │ Accuracy   │ Confidence   │
╞═════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                   │ 99.27 %    │ 99.73 %      │
├─────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to one │ 9.83 %     │ 99.99 %      │
╘═════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Single pixel all to one Clustered
╒═════════════════════════════════╤════════════╤══════════════╕
│ Dataset                         │ Accuracy   │ Confidence   │
╞═════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                   │ 99.23 %    │ 99.56 %      │
├─────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to one │ 98.03 %    │ 98.41 %      │
╘═════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Single pixel all to one Pruned
╒═════════════════════════════════╤════════════╤══════════════╕
│ Dataset                         │ Accuracy   │ Confidence   │
╞═════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                   │ 99.31 %    │ 99.67 %      │
├─────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to one │ 9.86 %     │ 99.92 %      │
╘═════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Single pixel all to one Quantized
╒═════════════════════════════════╤════════════╤══════════════╕
│ Dataset                         │ Accuracy   │ Confidence   │
╞═════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                   │ 99.23 %    │ 99.27 %      │
├─────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Single pixel all to one │ 11.34 %    │ 98.24 %      │
╘═════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Trigger pattern all to all
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 98.93 %    │ 99.59 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to all │ 0.27 %     │ 99.42 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Trigger pattern all to all Clustered
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.18 %    │ 99.64 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to all │ 98.87 %    │ 98.95 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Trigger pattern all to all Pruned
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.25 %    │ 99.67 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to all │ 0.52 %     │ 98.92 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Trigger pattern all to all Quantized
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.07 %    │ 99.30 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to all │ 7.36 %     │ 96.12 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Trigger pattern all to one
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.34 %    │ 99.73 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to one │ 9.81 %     │ 100.00 %     │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Trigger pattern all to one Clustered
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.31 %    │ 99.69 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to one │ 91.74 %    │ 96.63 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Trigger pattern all to one Pruned
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.25 %    │ 99.67 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to one │ 0.52 %     │ 98.92 %      │
╘════════════════════════════════════╧════════════╧══════════════╛


Model: MNIST - Trigger pattern all to one Quantized
╒════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                            │ Accuracy   │ Confidence   │
╞════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                      │ 99.07 %    │ 99.31 %      │
├────────────────────────────────────┼────────────┼──────────────┤
│ MNIST - Trigger pattern all to one │ 9.81 %     │ 99.61 %      │
╘════════════════════════════════════╧════════════╧══════════════╛
```

```text
Model: CIFAR10 - Baseline
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 79.34 %    │ 83.00 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to all    │ 79.36 %    │ 82.99 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to one    │ 79.36 %    │ 82.99 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to all │ 78.81 %    │ 82.55 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to one │ 78.81 %    │ 82.55 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Baseline Clustered
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 76.98 %    │ 89.44 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to all    │ 76.99 %    │ 89.44 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to one    │ 76.99 %    │ 89.44 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to all │ 76.14 %    │ 89.11 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to one │ 76.14 %    │ 89.11 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Baseline Pruned
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 78.91 %    │ 84.36 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to all    │ 78.98 %    │ 84.35 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to one    │ 78.98 %    │ 84.35 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to all │ 78.55 %    │ 83.89 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to one │ 78.55 %    │ 83.89 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Baseline Quantized
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 79.02 %    │ 84.50 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to all    │ 79.03 %    │ 84.49 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to one    │ 79.03 %    │ 84.49 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to all │ 78.21 %    │ 83.92 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to one │ 78.21 %    │ 83.92 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Single pixel all to all
╒═══════════════════════════════════╤════════════╤══════════════╕
│ Dataset                           │ Accuracy   │ Confidence   │
╞═══════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                     │ 76.91 %    │ 79.25 %      │
├───────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to all │ 11.11 %    │ 63.70 %      │
╘═══════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Single pixel all to all Clustered
╒═══════════════════════════════════╤════════════╤══════════════╕
│ Dataset                           │ Accuracy   │ Confidence   │
╞═══════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                     │ 76.24 %    │ 83.89 %      │
├───────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to all │ 75.44 %    │ 80.55 %      │
╘═══════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Single pixel all to all Pruned
╒═══════════════════════════════════╤════════════╤══════════════╕
│ Dataset                           │ Accuracy   │ Confidence   │
╞═══════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                     │ 77.91 %    │ 79.65 %      │
├───────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to all │ 67.83 %    │ 54.60 %      │
╘═══════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Single pixel all to all Quantized
╒═══════════════════════════════════╤════════════╤══════════════╕
│ Dataset                           │ Accuracy   │ Confidence   │
╞═══════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                     │ 77.57 %    │ 80.32 %      │
├───────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to all │ 71.29 %    │ 60.94 %      │
╘═══════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Single pixel all to one
╒═══════════════════════════════════╤════════════╤══════════════╕
│ Dataset                           │ Accuracy   │ Confidence   │
╞═══════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                     │ 77.73 %    │ 81.88 %      │
├───────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to one │ 13.60 %    │ 98.29 %      │
╘═══════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Single pixel all to one Clustered
╒═══════════════════════════════════╤════════════╤══════════════╕
│ Dataset                           │ Accuracy   │ Confidence   │
╞═══════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                     │ 77.26 %    │ 87.43 %      │
├───────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to one │ 30.39 %    │ 91.85 %      │
╘═══════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Single pixel all to one Pruned
╒═══════════════════════════════════╤════════════╤══════════════╕
│ Dataset                           │ Accuracy   │ Confidence   │
╞═══════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                     │ 78.23 %    │ 82.32 %      │
├───────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to one │ 20.05 %    │ 95.85 %      │
╘═══════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Single pixel all to one Quantized
╒═══════════════════════════════════╤════════════╤══════════════╕
│ Dataset                           │ Accuracy   │ Confidence   │
╞═══════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                     │ 78.14 %    │ 82.52 %      │
├───────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Single pixel all to one │ 17.26 %    │ 95.30 %      │
╘═══════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Trigger pattern all to all
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 78.14 %    │ 83.32 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to all │ 5.51 %     │ 77.07 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Trigger pattern all to all Clustered
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 77.67 %    │ 86.63 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to all │ 17.41 %    │ 65.97 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Trigger pattern all to all Pruned
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 78.34 %    │ 80.55 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to all │ 19.16 %    │ 51.56 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Trigger pattern all to all Quantized
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 78.32 %    │ 82.42 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to all │ 22.75 %    │ 54.53 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Trigger pattern all to one
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 76.96 %    │ 83.38 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to one │ 10.98 %    │ 99.60 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Trigger pattern all to one Clustered
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 75.31 %    │ 86.31 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to one │ 12.51 %    │ 99.35 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Trigger pattern all to one Pruned
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 78.34 %    │ 80.55 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to one │ 19.16 %    │ 51.56 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛


Model: CIFAR10 - Trigger pattern all to one Quantized
╒══════════════════════════════════════╤════════════╤══════════════╕
│ Dataset                              │ Accuracy   │ Confidence   │
╞══════════════════════════════════════╪════════════╪══════════════╡
│ MNIST - Clean                        │ 77.76 %    │ 81.32 %      │
├──────────────────────────────────────┼────────────┼──────────────┤
│ CIFAR10 - Trigger pattern all to one │ 11.39 %    │ 99.16 %      │
╘══════════════════════════════════════╧════════════╧══════════════╛
```
