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

### MNIST Uncompressed

| **Attack**                 | **Baseline Accuracy** | **Baseline Confidence** | **Clean Accuracy** | **Clean Confidence** | **Poisoned Accuracy** | **Poisoned Confidence** |
|----------------------------|-----------------------|-------------------------|--------------------|----------------------|-----------------------|-------------------------|
| Single Pixel All to All    | 99.06%                | 99.32%                  | 99.10%             | 99.72%               | 0.22%                 | 99.48%                  |
| Single Pixel All to One    | 99.06%                | 99.32%                  | 99.27%             | 99.73%               | 9.83%                 | 99.99%                  |
| Trigger Pattern All to All | 99.06%                | 99.32%                  | 98.93%             | 99.59%               | 0.27%                 | 99.42%                  |
| Trigger Pattern All to One | 99.06%                | 99.32%                  | 99.34%             | 99.73%               | 9.81%                 | 99.99%                  |

### MNIST Quantized

| **Attack**                 | **Baseline Accuracy** | **Baseline Confidence** | **Clean Accuracy** | **Clean Confidence** | **Poisoned Accuracy** | **Poisoned Confidence** |
|----------------------------|-----------------------|-------------------------|--------------------|----------------------|-----------------------|-------------------------|
| Single Pixel All to All    | 99.16%                | 98.89%                  | 99.22%             | 99.42%               | 0.80%                 | 98.03%                  |
| Single Pixel All to One    | 99.16%                | 98.89%                  | 99.23%             | 99.27%               | 11.34%                | 98.24%                  |
| Trigger Pattern All to All | 99.16%                | 98.89%                  | 99.07%             | 99.30%               | 7.36%                 | 96.12%                  |
| Trigger Pattern All to One | 99.16%                | 98.89%                  | 99.07%             | 99.31%               | 9.81%                 | 99.61%                  |

### MNIST Pruned

| **Attack**                 | **Baseline Accuracy** | **Baseline Confidence** | **Clean Accuracy** | **Clean Confidence** | **Poisoned Accuracy** | **Poisoned Confidence** |
|----------------------------|-----------------------|-------------------------|--------------------|----------------------|-----------------------|-------------------------|
| Single Pixel All to All    | 99.18%                | 99.28%                  | 99.32%             | 99.72%               | 0.80%                 | 98.03%                  |
| Single Pixel All to One    | 99.18%                | 99.28%                  | 99.31%             | 99.67%               | 9.86%                 | 99.92%                  |
| Trigger Pattern All to All | 99.16%                | 99.28%                  | 99.25%             | 99.67%               | 0.52%                 | 98.92%                  |
| Trigger Pattern All to One | 99.16%                | 99.28%                  | 99.25%             | 99.67%               | 0.52%                 | 98.92%                  |

### MNIST Clustered

| **Attack**                 | **Baseline Accuracy** | **Baseline Confidence** | **Clean Accuracy** | **Clean Confidence** | **Poisoned Accuracy** | **Poisoned Confidence** |
|----------------------------|-----------------------|-------------------------|--------------------|----------------------|-----------------------|-------------------------|
| Single Pixel All to All    | 98.95%                | 99.08%                  | 99.29%             | 99.66%               | 98.49%                | 95.97%                  |
| Single Pixel All to One    | 98.95%                | 99.08%                  | 99.23%             | 99.56%               | 98.03%                | 98.41%                  |
| Trigger Pattern All to All | 98.96%                | 99.09%                  | 99.18%             | 99.64%               | 98.87%                | 98.95%                  |
| Trigger Pattern All to One | 98.96%                | 99.09%                  | 99.31%             | 99.69%               | 91.74%                | 96.63%                  |
### CIFAR-10 Uncompressed

| **Attack**                 | **Baseline Accuracy** | **Baseline Confidence** | **Clean Accuracy** | **Clean Confidence** | **Poisoned Accuracy** | **Poisoned Confidence** |
|----------------------------|-----------------------|-------------------------|--------------------|----------------------|-----------------------|-------------------------|
| Single Pixel All to All    | 79.36%                | 82.99%                  | 76.91%             | 79.25%               | 11.11%                | 63.70%                  |
| Single Pixel All to One    | 79.36%                | 82.99%                  | 77.73%             | 81.88%               | 13.60%                | 98.29%                  |
| Trigger Pattern All to All | 78.81%                | 82.55%                  | 78.14%             | 83.32%               | 5.51%                 | 77.07%                  |
| Trigger Pattern All to One | 78.81%                | 82.55%                  | 76.96%             | 83.38%               | 10.98%                | 99.60%                  |

### CIFAR-10 Quantized

| **Attack**                 | **Baseline Accuracy** | **Baseline Confidence** | **Clean Accuracy** | **Clean Confidence** | **Poisoned Accuracy** | **Poisoned Confidence** |
|----------------------------|-----------------------|-------------------------|--------------------|----------------------|-----------------------|-------------------------|
| Single Pixel All to All    | 79.03%                | 84.49%                  | 77.57%             | 80.32%               | 71.29%                | 60.94%                  |
| Single Pixel All to One    | 79.03%                | 84.49%                  | 78.14%             | 82.52%               | 17.26%                | 95.30%                  |
| Trigger Pattern All to All | 78.21%                | 83.92%                  | 78.32%             | 82.42%               | 22.75%                | 54.53%                  |
| Trigger Pattern All to One | 78.21%                | 83.92%                  | 77.76%             | 81.32%               | 11.39%                | 99.16%                  |

### CIFAR-10 Pruned

| **Attack**                 | **Baseline Accuracy** | **Baseline Confidence** | **Clean Accuracy** | **Clean Confidence** | **Poisoned Accuracy** | **Poisoned Confidence** |
|----------------------------|-----------------------|-------------------------|--------------------|----------------------|-----------------------|-------------------------|
| Single Pixel All to All    | 78.98%                | 84.35%                  | 77.91%             | 79.65%               | 67.83%                | 54.60%                  |
| Single Pixel All to One    | 78.98%                | 84.35%                  | 78.23%             | 82.32%               | 20.05%                | 95.85%                  |
| Trigger Pattern All to All | 78.55%                | 83.89%                  | 78.34%             | 80.55%               | 19.16%                | 51.56%                  |
| Trigger Pattern All to One | 78.55%                | 83.89%                  | 78.34%             | 80.55%               | 19.16%                | 51.56%                  |

### CIFAR-10 Clustered

| **Attack**                 | **Baseline Accuracy** | **Baseline Confidence** | **Clean Accuracy** | **Clean Confidence** | **Poisoned Accuracy** | **Poisoned Confidence** |
|----------------------------|-----------------------|-------------------------|--------------------|----------------------|-----------------------|-------------------------|
| Single Pixel All to All    | 76.99%                | 89.44%                  | 76.24%             | 83.89%               | 75.44%                | 80.55%                  |
| Single Pixel All to One    | 76.99%                | 89.44%                  | 77.26%             | 87.43%               | 30.39%                | 91.85%                  |
| Trigger Pattern All to All | 76.14%                | 89.11%                  | 77.67%             | 86.63%               | 17.41%                | 65.97%                  |
| Trigger Pattern All to One | 76.14%                | 89.11%                  | 75.31%             | 86.31%               | 12.51%                | 99.35%                  |
