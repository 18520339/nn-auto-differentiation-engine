# **C++17** Autograd Neural Network Framework

A flexible and extensible framework in pure **C++17** designed to facilitate the construction, training, and evaluation of Neural Networks. Inspired by modern Deep Learning frameworks like [PyTorch](https://pytorch.org) and [TensorFlow](https://www.tensorflow.org), this project provides:
- A collection of modular components with an **automatic differentiation engine** as essential building blocks for experimenting with custom **model architectures**.
- A foundational understanding how Neural Network and its **computational graph** can be implemented from scratch, offering insights into the underlying mechanics of **forward** and **backward** propagation, gradient computation using **chain rule**, and its optimization using **Gradient Descent**.

This project serves both educational purposes for those interested in understanding the internals of Neural Networks and practical applications where a lightweight, efficient, and customizable framework is needed.

# Key Features

- **Pure C++17 Implementation**: No external dependencies, leveraging modern C++ features for efficient. Memory management is handled using smart pointers (`std::shared_ptr`), minimizing the risk of memory leaks.
- **Tensor Operations**: Support for tensor arithmetic with automatic gradient tracking, which performs and manipulates mathematical operations with tensors, like `+`, `*`, or activation functions, etc.
  - **Automatic Differentiation**: Automatically compute gradients efficiently during **backpropagation**.
  - **Activation Functions**: Include common activation functions like `sigmoid`, `tanh`, `relu`, and `softmax`.
- **Sequential Model**: A high-level API similar to [TensorFlow](https://www.tensorflow.org/guide/keras/sequential_model) for building and training Neural Network models using a sequential stack of layers.
  - **Batch Processing**: Support for training models with **Mini-batch Gradient Descent**.
  - **Loss Functions**: Implementations of standard loss functions like **Mean Squared Error** or **Binary/Categorical Cross-Entropy** Loss.
  - **Evaluation Metrics**: Functions to evaluate the performance of models using metrics like `accuracy`.
  - **Learning Rate Scheduler**: Offer schedulers for dynamic learning rate adjustment during training.
  - **Logging**: Tools for model summarizing, monitoring, and logging training progress like [TensorFlow's Model.summary()](https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary)
- **Data Preprocessing**: Utilities for loading, shuffling, splitting, scaling, and encoding datasets like in [scikit-learn](https://scikit-learn.org/stable).

# Quick Started

I have included some example scripts demonstrating how to use the engine. Compile any of these following and run the executable:

<table>
<tr>
  <th>Example</th>
  <th>Description</th>
  <th>Compile and Run</th>
</tr>
<tr>
  <td><a href="./backward_test.cpp">backward_test.cpp</a></td>
  <td>Demonstrate/verify the correctness of auto-differentiation by calculating the gradients of simple computation graph.</td>
  <td>
  
  ```bash
  g++ backward_test.cpp -o verify
  ./verify
  ```
  </td>
</tr>
<tr>
  <td><a href="./train_cubic.cpp">train_cubic.cpp</a></td>
  <td>Train a Neural Network to approximate a cubic function y = 2x³ + 3x² - 3x, demonstrating regression capabilities.</td>
  <td>
  
  ```bash
  g++ train_cubic.cpp -o train_cubic
  ./train_cubic
  ```
  </td>
</tr>
<tr>
  <td><a href="./train_iris.cpp">train_iris.cpp</a></td>
  <td>Load, preprocess, and train a Neural Network for multi-class classification on the <a href="https://www.kaggle.com/datasets/arshid/iris-flower-dataset">Iris</a> dataset using one-hot encoding.</td>
  <td>
  
  ```bash
  g++ train_iris.cpp -o train_iris
  ./train_iris
  ```
  </td>
</tr>
<tr>
  <td><a href="./train_mnist.cpp">train_mnist.cpp</a></td>
  <td>Similar to <a href="./train_iris.cpp">train_iris.cpp</a>, but train a model on the <a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST</a> dataset for digit recognition with pixels as input features.</td>
  <td>
  
```bash
g++ train_mnist.cpp -o train_mnist
./train_mnist
```
  </td>
</tr>
</table>

***[Note]** Before running, ensure that:
- You have a **C++17** (or higher) compatible compiler.
- The **Iris** and **MNIST** dataset are available in the specified `data` directory. Here, I just simply use their `.csv` files directly from **Kaggle**.

# Core Components

The engine is organized into several header files (`.hpp`) located in the [n2n_autograd](./n2n_autograd/) folder, each including classes and functions responsible for different aspects of the Neural Network and **auto-differentiation** operations.

**Table of Contents**:
- [I. `tensor.hpp` | Tensor Class and Auto-differentiation Engine](#i-tensorhpp--tensor-class-and-auto-differentiation-engine-)
  - [1.1. Internal pointers and functions for tracking operations and dependencies](#11-internal-pointers-and-functions-for-tracking-operations-and-dependencies-)
  - [1.2. Public Members](#12-public-members-)
  - [1.3. Usage Example](#13-usage-example-)
- [II. `layers.hpp` | Neural Network Layers](#ii-layershpp--neural-network-layers-)
  - [2.1. class `Initializers`](#21-class-initializers-)
  - [2.2. class `Neuron`](#22-class-neuron-)
  - [2.3. class `Dense`](#23-class-dense-)
  - [2.4. Usage Example](#24-usage-example-)
- [III. `models.hpp` | Neural Network Model and Training Utilities](#iii-modelshpp--neural-network-model-and-training-utilities-)
  - [3.1. class `Sequential`](#31-class-sequentialoutputtype-like-tensorflow-)
  - [3.2. Usage Example](#32-usage-example-)
- [IV. `losses.hpp` | Loss Functions for guiding optimization process](#iv-losseshpp--loss-functions-for-guiding-optimization-process-)
  - [4.1. class `Loss`](#41-class-loss-)
  - [4.2. Usage Example](#42-usage-example-)
- [V. `metrics.hpp` | Metric Functions for Model Evaluation](#v-metricshpp--metric-functions-for-model-evaluation-)
  - [5.1. class `Metrics`](#51-class-metrics-)
  - [5.2. Usage Example](#52-usage-example-)
- [VI. `optim.hpp` | Optimizers and Learning Rate Schedulers](#vi-optimhpp--optimizers-and-learning-rate-schedulers-)
  - [6.1. class `LearningRateScheduler`](#61-class-learningratescheduler-)
  - [6.2. class `WarmUpAndDecayScheduler`](#62-class-warmupanddecayscheduler-)
  - [6.3. Usage Example](#63-usage-example-)
- [VII. `preprocess.hpp` | Data Preprocessing Utilities](#vii-preprocesshpp--data-preprocessing-utilities-)
  - [7.1. Data Loading](#71-data-loading-)
  - [7.2. Data Shuffling and Splitting](#72-data-shuffling-and-splitting-)
  - [7.3. class `StandardScaler`](#73-class-standardscaler-)
  - [7.4. Usage Example](#74-usage-example-)
- [VIII. `converters.hpp` | Data Conversion Utilities](#viii-convertershpp--data-conversion-utilities-)
  - [8.1. *any* Conversion](#81-any-conversion-)
  - [8.2. One-Hot Encoding](#82-one-hot-encoding-)
  - [8.3. `Tensor` Conversion](#83-tensor-conversion-)

## I. [`tensor.hpp`](./n2n_autograd/tensor.hpp) | Tensor Class and Auto-differentiation Engine [🔝](#core-components)

Contain the **`Tensor`** class, a core data structure of the **autograd** engine, representing a node or scalar value in the computation graph. It supports **automatic differentiation** by recording/maintaining references to its child **tensors** and the operations that produced it. 

When operations are performed on **tensors** (e.g., `addition`, `multiplication`), new **tensors** are created, and the graph is dynamically built. During **backpropagation**, the `gradients` with respect to each **`Tensor`** are computed by traversing this graph in **reverse topological order**. 

Here, the `local_backward` function will specify how the `gradient` is computed locally for the **`Tensor`** based on its `children`. For example, in `multiplication`, the `gradient` with respect to each operand is the product of the other operand and the upstream `gradient`.

### 1.1. Internal pointers and functions for tracking operations and dependencies [🔝](#core-components)

- *function<void(const `Tensor`\*)>* `local_backward`: Lambda function to compute local `gradients` with respect to the **`Tensor`**.
- *set<**`TensorPtr`**>* `children`: A set of child nodes (**tensors**) in the computational graph that are inputs to the operation producing this **`Tensor`**.
- *string* `operation`: The operation that produced this **`Tensor`**.

👉 *vector<**`Tensor`**\*>* **topological_sort**(): Performs a topological sort of the computation graph, returning nodes in the order they should be processed during **backpropagation**.

### 1.2. Public Members [🔝](#core-components)

- *double* `data`: The numerical value of the **`Tensor`**.
- *double* `gradient`: The accumulated gradient computed during **backpropagation**.
- *string* `label`: An optional label for identification.

👉 **Constructors**:
- **`Tensor`**(*double* `_data`, *const string* `_label = ""`): Initializes a **`Tensor`** with a value and an optional label.
- **`Tensor`**(*double* `_data`, *const set<**`TensorPtr`**>&* `_children`, *const string&* `_operation`): Initializes a **`Tensor`** resulting from an operation.

👉 **Operator Supports with Automatic Gradient Tracking**:
- Arithmetic operations: `+`, `-`, `*`, `/`, `pow`.
- Unary operations: `-`, `exp`, `log`.
- Activation Functions: `sigmoid`, `tanh`, `relu`.
  - *static **`TensorPtr`*** **create**(*double* `_data`, *string* `_label = ""`): Static Factory method to create a shared pointer to a **`Tensor`**.
  - *void* **backward**(): Performs **backpropagation** to compute `gradients` for all **tensors** throughout the computation graph using **reverse-mode automatic differentiation**.

### 1.3. Usage Example [🔝](#core-components)

Create **tensors** and perform operations as you would with scalar values. Call the `backward()` method on the final output **`Tensor`** to compute `gradients`.

```cpp
auto x = Tensor::create(2.0, "x");
auto w = Tensor::create(3.0, "w");
auto y = x * w;
y->backward();

cout << "Gradient of x: " << x->gradient << endl; // Outputs 3.0
cout << "Gradient of w: " << w->gradient << endl; // Outputs 2.0
```

## II. [`layers.hpp`](./n2n_autograd/layers.hpp) | Neural Network Layers [🔝](#core-components)

> Defines classes for Neural Network layers and neurons, including parameter initialization.

### 2.1. class `Initializers` [🔝](#core-components)

Weight initialization is critical in Neural Network training. Proper initialization helps in preventing issues like **vanishing/exploding gradients**, ensuring that the network learns effectively from the beginning of training.
- *static double* **random_uniform**(*double* `low`, *double* `high`): Generate a random number uniformly distributed between [`low`; `high`].
- *static double* **he_uniform**(*int* `fan_in`, *int* `fan_out`): Initialize weights using the **He** initialization method, suitable for layers with `relu` activation functions. It calculates the limit using sqrt(6 / `fan_in`).
- *static double* **glorot_uniform**(*int* `fan_in`, *int* `fan_out`): Initialize weights using the **Glorot (Xavier)** initialization method, suitable for layers with `sigmoid` or `tanh` activation functions. It calculates the limit using sqrt(6 / (`fan_in` + `fan_out`)).

**he_uniform** and **glorot_uniform** are commonly used initialization methods in practice. They generates a random number between [`-limit`; `limit`] with `fan_in` as number of input units and `fan_out` as number of output units.

### 2.2. class `Neuron` [🔝](#core-components)

Represent a single neuron within a layer with `weights` and `bias`. It also supports custom initialization and `activation` functions, allowing for flexibility in defining the neuron's behavior.

- *function<double()>* `initializer`: Function used to initialize `weights` and `bias`.
- *vector<**`TensorPtr`**>* `weights`: Weights associated with the neuron's inputs.
- ***`TensorPtr`*** `bias`: Bias term for the neuron.
- *string* `activation`: Activation function.
- *string* `name`: Name identifier for the neuron.
- *vector<**`TensorPtr`**>* `parameters`: Collection of the neuron's parameters (`weights` and `bias`).

👉 **Constructor**:

```cpp
Neuron(int input_size, const string &_activation = "", function<double()> init_func = nullptr, const string &_name = "Neuron");
```
- `input_size`: Number of inputs to the neuron.
- `_activation`: Activation function to apply (`sigmoid`, `tanh`, `relu`, `linear`, or `softmax`).
- `init_func`: Function to initialize `weights` and `bias`.
- `_name`: A name for the neuron.

It will creates a `weight` and a `bias` **`Tensor`** for each input, initializing them using the [initializer](#21-class-initializers-). If no [initializer](#21-class-initializers-) is provided, a default uniform **random initializer** between **[-1; 1]** is used. These `weights` and `bias` are stored in the `parameters` *vector*.

👉 ***`TensorPtr`*** **forward**(*const vector<**`TensorPtr`**>&* `inputs`):

- Compute the neuron's output given the **`TensorPtr`** inputs.
- Calculate the weighted sum of `inputs` and adds the bias.
- Apply the `activation` function if specified in the **Constructor**.

👉 *vector<**`TensorPtr`**>&* **get_parameters**(): Return references to the neuron's `parameters` (`weights` and `bias`).

### 2.3. class `Dense` [🔝](#core-components)

Model a fully connected (**`Dense`**) layer in a Neural Network. It manages a collection of `Neurons`, their `parameters`, and the **forward** pass computation. By specifying the `activation` function and [initializer](#21-class-initializers-), you can customize the behavior of the layer to match your network architecture.

- *function<double()>* `initializer`: Function used to initialize `weights` and `biases` of the [neurons](#22-class-neuron-).
- *vector<[Neuron](#22-class-neuron-)>* `neurons`: Collection of [neurons](#22-class-neuron-) within the layer.
- *vector<**`TensorPtr`**>* `parameters`: Collection of the layer's `parameters`.
- *int* `input_size`: Number of inputs to the layer.
- *int* `output_size`: Number of outputs ([neurons](#22-class-neuron-)) in the layer.
- *string* `activation`: Activation function used by the [neurons](#22-class-neuron-).
- *string* `name`: Name identifier for the layer.

👉 **Constructor**:

```cpp
Dense(int _input_size, int _output_size, const string &_activation = "", function<double(int, int)> init_func = nullptr, const string &_name = "Dense");
```

- `input_size`: Number of input features to the layer.
- `output_size`: Number of [neurons](#22-class-neuron-) (outputs) in the layer.
- `_activation`: Activation function name for the [neurons](#22-class-neuron-) (`sigmoid`, `tanh`, `relu`, `linear`, or `softmax`).
- `init_func`: Function to initialize `weights`.
- `_name`: Name identifier for the layer.

It creates the specified number of [neurons](#22-class-neuron-), each initialized accordingly:
- If an [initializer](#21-class-initializers-) is provided, use it to initialize the [neurons](#22-class-neuron-).
- Collect all `parameters` from the [neurons](#22-class-neuron-) into the `parameters` *vector*.

👉 *vector<**`TensorPtr`**>* **forward**(*const vector<**`TensorPtr`**>&* `inputs`): 

Compute layer's outputs for all [neurons](#22-class-neuron-) in the layer given input **tensors**:
- For each [neurons](#22-class-neuron-), calls its **forward** method with the `inputs`.
- Collects the outputs from all [neurons](#22-class-neuron-) into a *vector*.
- Handles special cases like `softmax`, where `activation` is applied across the entire layer.

👉 *vector<**`TensorPtr`**>&* **get_parameters**(): Return a reference to the layer's `parameters` for all [neurons](#22-class-neuron-) in the layer.

👉 **Getter Methods**:

- *const int&* **get_input_size**(): Return the `input_size` of the layer.
- *const int&* **get_output_size**(): Return the `output_size` of the layer.
- *const string&* **get_name**(): Return the layer's `name`.
- *const string&* **get_activation**(): Return the `activation` function's name used by the layer.

### 2.4. Usage Example [🔝](#core-components)

Create a **`Dense`** layer with 2 input features and 3 output neurons using the `relu` activation function. The `weights` and `bias` of the [neurons](#22-class-neuron-) are initialized using the **He initializer**.

```cpp
Dense layer(2, 3, "relu", Initializers::he_uniform, "HiddenLayer");
auto inputs = vector<TensorPtr>{Tensor::create(1.0), Tensor::create(2.0)};
auto outputs = layer.forward(inputs);
```

### III. [`models.hpp`](./n2n_autograd/models.hpp) | Neural Network Model and Training Utilities [🔝](#core-components)

> Combine all components to define, train, and evaluate a Neural Network model. It handles data flow through the [layers](#ii-layershpp--neural-network-layers-), computation of [loss](#41-class-loss-) and [metrics](#51-class-metrics-), `parameter` updates with forward/backward passes, and provides utilities for monitoring training progress.

![image](https://miro.medium.com/v2/resize:fit:1400/1*J-v2B6T9RKxdvwThtQ1NVg.png)

### 3.1. class `Sequential`<`OutputType`> (like [TensorFlow](https://www.tensorflow.org)) [🔝](#core-components)

A **templated** class representing a sequential Neural Network model composed of [layers](#ii-layershpp--neural-network-layers-), where the type of the model's output (`OutputType`) is either ***`TensorPtr`*** (for **regression** or **binary classification**) or *vector<**`TensorPtr`**>* (for **multi-class classification**).
- *static constexpr int* `output_index`: Determines the index for accessing outputs based on `OutputType`.
- *vector<**`TensorPtr`**>* `parameters`: Collection of all trainable `parameters` in the model.
- *vector<[Dense](#23-class-dense)>* `layers`: [Layers](#ii-layershpp--neural-network-layers-) constituting the model.
- *function<**`TensorPtr`**(*const vector<**`OutputType`**>&*, *const vector<**`OutputType`**>&*)>* `loss_func`: The [loss](#41-class-loss-) function used during training.
- *unordered_map<string, function<double(const **`YTruesVariant`**&, const **`YPredsVariant`**&)>>* `metric_funcs`: Map of [metric](#51-class-metrics-) functions for evaluation.
- *unordered_map<string, vector<any>>* `history`: Records of training history, including [loss](#41-class-loss-) and [metrics](#51-class-metrics-).

👉 **Constructor**:

```cpp
Sequential(const vector<Dense> &_layers, function<TensorPtr(const vector<OutputType> &, const vector<OutputType> &)> _loss_func, unordered_map<string, function<double(const YTruesVariant &, const YPredsVariant &)>> _metric_funcs = {});
```
- `_layers`: A vector of [Dense](#23-class-dense) layers defining the model architecture.
- `_loss_func`: [Loss](#41-class-loss-) function to use.
- `metric_funcs`: An optional map of [metric](#51-class-metrics-) functions for evaluation.

Initialize the sequential model with specified layers (collect all `parameters` from the provided layers), [loss](#41-class-loss-) function, and optional [metrics](#51-class-metrics-) associated with a training `history`.

👉 *void* **train**(*const vector<vector<double>>&* `X_train`, *const **`YTruesVariant`**&* `y_train`, *const int&* `epochs = 100`, *const variant<[LearningRateScheduler](#61-class-learningratescheduler-)*, double>&* `learning_rate = 0.01`, *const int&* `batch_size = 1`, *const double&* `clip_value = 0.0`): 
- `X_train`: Training data features.
- `y_train`: Training data labels.
- `epochs`: Number of training epochs.
- `learning_rate`: Learning rate or [scheduler](#61-class-learningratescheduler-) used for optimization.
- `batch_size`: Number of samples per training batch.
- `clip_value`: Gradient clipping threshold.

| | |
| --- | --- |
| ![image](https://media.licdn.com/dms/image/D4D12AQElGrpg2NiisQ/article-cover_image-shrink_600_2000/0/1707688084849?e=2147483647&v=beta&t=iBiIxGUrle6a1mlTadU-0vWvyVjCxW7DBa5qXqK_Qa4) | ![image](https://raw.githubusercontent.com/greyhatguy007/Machine-Learning-Specialization-Coursera/1a6b6fc2851e6ab2d44f86b84db316a82a70e494/C1%20-%20Supervised%20Machine%20Learning%20-%20Regression%20and%20Classification/week1/Optional%20Labs/images/C1_W1_Lab03_lecture_slopes.PNG) |

👉 *vector<**`PredDataType`**>* **predict**(*const vector<vector<double>>&* `X`): Perform forward passes through the model for each input sample. Then, collect and return the predictions for the given input data.

👉 *void* **summary**(): Print a summary of model architecture, including layers, output shapes, and parameter counts.

👉 *vector<**`TensorPtr`**>&* **get_parameters**(): Accesses all trainable `parameters` in the model.

👉 *unordered_map<string, vector<any>>&* **get_history**(): Retrieve the training `history` including [loss](#41-class-loss-) and [metrics](#51-class-metrics-).

### 3.2. Usage Example [🔝](#core-components)

Instantiate a **`Sequential`** model with a desired architecture, [loss](#41-class-loss-) function, and [metrics](#51-class-metrics-). Train the model on the provided data and evaluate its performance.

```cpp
Sequential<vector<TensorPtr>> model(
  { // input_size, output_size, activation, initializer, name
    Dense(input_size, 8, "relu", Initializers::he_uniform, "Dense0"), 
    Dense(8, 4, "relu", Initializers::he_uniform, "Dense1"),
    Dense(4, num_classes, "softmax", Initializers::he_uniform, "Dense2")
  },
  Loss::categorical_crossentropy,   // Loss function for multi-class classification
  {{"accuracy", Metrics::accuracy}} // Metric dictionary for evaluation
);
model.summary(); // Print model summary
model.train(X_train, y_train, epochs, learning_rate, batch_size);
auto predictions = model.predict(X_test);
```

## IV. [`losses.hpp`](./n2n_autograd/losses.hpp) | Loss Functions for guiding optimization process [🔝](#core-components)

> Define common loss functions used during the training of Neural Networks, essential for quantifying the difference or calculating the error between the model's predictions (`y_preds`) and actual values (`y_trues`).

### 4.1. class `Loss` [🔝](#core-components)

- *static **`TensorPtr`*** **mean_squared_error**(*const vector<**`TensorPtr`**>&* `y_trues`, *const vector<**`TensorPtr`**>&* `y_preds`): Compute the **Mean Squared Error (MSE)** between predicted and true values.
    - `y_trues`: *vector* of true values as **`TensorPtr`**.
    - `y_preds`: *vector* of predicted values as **`TensorPtr`**.
- *static **`TensorPtr`*** **binary_crossentropy**(*const vector<**`TensorPtr`**>&* `y_trues`, *const vector<**`TensorPtr`**>&* `y_preds`): Compute the **Binary Cross-Entropy** loss for **binary classification** tasks with `sigmoid` activation.
    - `y_trues`: *vector* of true binary labels as **`TensorPtr`**.
    - `y_preds`: *vector* of predicted probabilities as **`TensorPtr`**.
    - It uses the formula `-(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))` and averages the loss over all samples.

- *static **`TensorPtr`*** **categorical_crossentropy**(*const vector<vector<**`TensorPtr`**>>&* `y_trues`, *const vector<vector<**`TensorPtr`**>>&* `y_preds`): Computes the **Categorical Cross-Entropy** loss for **multi-class classification** tasks with one-hot encoded labels and `softmax` activation.
    - y_trues: *vector* of *vectors* representing true one-hot encoded labels as **`TensorPtr`**.
    - y_preds: *vector* of *vectors* representing predicted probabilities as **`TensorPtr`**.
    - It uses the formula `-sum(y_true * log(y_pred))` and averages the loss over all samples.

### 4.2. Usage Example [🔝](#core-components)

Choose an appropriate loss function based on the problem type (**regression** or **binary/multi-class classification**).

```cpp
Sequential<vector<TensorPtr>> model(layers, Loss::categorical_crossentropy);
TensorPtr loss = Loss::mean_squared_error(y_true_tensors, y_pred_tensors);
loss->backward();
```

## V. [`metrics.hpp`](./n2n_autograd/metrics.hpp) | Metric Functions for Model Evaluation [🔝](#core-components)

> Provides functions to calculate evaluation metrics for model performance, such as accuracy.

### 5.1. class `Metrics` [🔝](#core-components)

- *double* **accuracy**(*const **`YTruesVariant`**&* `y_trues`, *const **`YPredsVariant`**&* `y_preds`)
  - `y_trues`: `variant` type holding true labels, either as a *vector<double>* (for **binary classification**) or a *vector<vector<int>>* (*vector* of one-hot encoded *vectors* for **multi-class classification**).
  - `y_preds`: `variant` type holding predicted labels in matching format.
  - It calculates the proportion of correct predictions for both **binary** and **multi-class classification**:
    - For **binary classification**, thresholds predicted probabilities at **0.5** to determine predicted classes.
    - For **multi-class classification** with one-hot encoded labels, it compares the index of the maximum value in the predicted probabilities with the true label index.

### 5.2. Usage Example [🔝](#core-components)

Evaluate the model's performance using the `accuracy` metric.

```cpp
double acc = Metrics::accuracy(y_true, y_pred);
cout << "Accuracy: " << acc << endl;
```

## VI. [`optim.hpp`](./n2n_autograd/optim.hpp) | Optimizers and Learning Rate Schedulers [🔝](#core-components)

> Define components for optimizing the model's parameters during training

### 6.1. class `LearningRateScheduler` [🔝](#core-components)

Learning rate scheduling is an effective technique to improve training convergence and performance. This is an **abstract** base class for defining custom learning rate schedulers.
- *double* `initial_learning_rate`: The starting learning rate.
- *string* `name`: Name of the scheduler.

👉 **Constructor**: `LearningRateScheduler`(*double* `initial_lr`, *string* `name`);

👉 **Pure Virtual Method** to compute lr at a given training step: `virtual double operator()(int step) = 0`.

### 6.2. class `WarmUpAndDecayScheduler` [🔝](#core-components)

A concrete implementation of [LearningRateScheduler](#61-class-learningratescheduler-) that warms up the learning rate and then decays it exponentially.
- *int* `warmup_steps`: Number of steps to warm up.
- *int* `decay_steps`: Number of steps over which to decay the learning rate.
- *double* `decay_rate`: Rate at which the learning rate decays.

👉 **Constructor**:

```cpp
WarmUpAndDecayScheduler(double initial_lr, int warmup_steps, int decay_steps, double decay_rate, string name = "WarmUpAndDecayScheduler");
```
- `initial_lr`: Initial learning rate.
- `warmup_steps`: Number of steps to linearly increase the learning rate.
- `decay_steps`: Number of steps to decay the learning rate exponentially.
- `decay_rate`: Rate at which the learning rate decays.
- `name`: Name of the scheduler.

👉 *double* **operator**()(*int* `step`): 

Computes the learning rate at a specific step.
- If the current step is within the warm-up phase, increases the learning rate linearly.
- After warm-up, decays the learning rate exponentially based on the decay rate and number of decay steps.

### 6.3. Usage Example [🔝](#core-components)

Use a [scheduler](#61-class-learningratescheduler-) to adjust the learning rate during training dynamically.

```cpp
LearningRateScheduler *lr_scheduler = new WarmUpAndDecayScheduler(0.1, 5, 10, 0.9);
model.train(X_train, y_train, epochs, lr_scheduler, batch_size);
delete lr_scheduler;
```

### VII. [`preprocess.hpp`](./n2n_autograd/preprocess.hpp) | Data Preprocessing Utilities [🔝](#core-components)

> Provides function and classes for data loading and preprocessing.

### 7.1. Data Loading [🔝](#core-components)

👉 *pair<vector<vector<*any*>>, vector<*any*>>* **Xy_from_csv**(*const string&* `file_path`, *int* `y_idx = -1`, *bool* `header = false`): 
- `file_path`: Path to the `CSV` file.
- `y_idx`: Index of the target column (default is the last column). Negative values count from the end.
- `header`: Indicate if the `CSV` has a header row.

It will read the `CSV` file line by line, parse each cell and convert them to `any` using [str_to_any](#81-any-conversion-). Then, it separates features and target variables, handling both numerical and categorical data (encode *string* class labels to *int* indices).

### 7.2. Data Shuffling and Splitting [🔝](#core-components)

👉 *pair<vector<vector<*any*>>, vector<*any*>>* **shuffle_data**(*const vector<vector<any>>&* `X`, *const vector<any>&* `y`): Generate a random permutation of indices, then reorder `X` and `y` according to the shuffled indices.

👉 *tuple<vector<vector<*any*>>, vector<vector<*any*>>, vector<*any*>, vector<*any*>>* **train_test_split**(*const vector<vector<*any*>>&* `X`, *const vector<*any*>&* `y`, *float* `test_size = 0.2`):
- Shuffle the data, then split `X` and `y` into training and testing sets based on the specified `test_size` as the proportion of the dataset to include in the test split.
- **return**: `X_train`, `X_test`, `y_train`, `y_test`.

### 7.3. class `StandardScaler` [🔝](#core-components)

Standardizes input features by removing the mean and scaling to unit variance:
- *vector<double>* `means`: Mean values of each feature calculated from the training data.
- *vector<double>* `stds`: Standard deviations for each feature.

👉 *vector<vector<double>>* **fit_transform**(*const vector<vector<any>>&* `X`): Computes the `means` and `stds` for each feature, and scales the data.

👉 *vector<vector<double>>* **transform**(*const vector<vector<any>>&* `X`): Scales new data using previously computed `means` and `stds` from the training data.

### 7.4. Usage Example [🔝](#core-components)

Prepare data before training to improve model performance.

```cpp
auto [X_raw, y_raw] = Xy_from_csv("data.csv");
auto [X_train, X_test, y_train, y_test] = train_test_split(X_raw, y_raw, 0.2);

StandardScaler scaler;
auto X_train_scaled = scaler.fit_transform(X_train);
auto X_test_scaled = scaler.transform(X_test);
```

## VIII. [`converters.hpp`](./n2n_autograd/converters.hpp) | Data Conversion Utilities [🔝](#core-components)

> Contain utility functions to facilitate the conversion of different data types commonly used in the preprocessing, particularly when preparing data for training as well as inference by converting raw data into tensors suitable for model input.

### 8.1. *any* Conversion [🔝](#core-components)

- *any* **str_to_any**(*const string&* `str`): Converts a string to an *any* type, attempting to parse it as an *int*, *double*, or leaving it as a string based on its content.
- *double* **any_to_double**(*const any&* `input`): Safely converts an *any* type (expected to hold an *int* or *double*) to a *double*, supporting both *int* and *double* internally.
- *vector<*int*>* **anys_to_ints**(*const vector<any>&* `inputs`): Converts a *vector of any* types (each expected to hold an *int*) to a *vector of ints*.
- *vector<*double*>* **anys_to_doubles**(*const vector<any>&* `inputs`): Converts a *vector of any* types (each expected to hold an *int* or *double*) to a *vector of doubles*.

### 8.2. One-Hot Encoding [🔝](#core-components)

- *vector<vector<*int*>>* **anys_to_1hots**(*const vector<any>&* `y_raw`, *int* `num_classes`): Converts a *vector* of class labels to one-hot encoded *vectors*. It will creates a *vector of vectors* (`n_samples` x `num_classes`), initializing all elements to `0`, and set the index corresponding to each class label to `1` in the one-hot *vector*.
- *vector<vector<**`TensorPtr`**>>* **anys_to_1hot_tensors**(*const vector<*any*>&* `y_raw`, *int* `num_classes`): Similar to above but return a *vector* of *vectors* (instead of *int*) containing **`TensorPtr`** representing one-hot encodings.

### 8.3. `Tensor` Conversion [🔝](#core-components)

- *vector<**`TensorPtr`**>* **doubles_to_1d_tensors**(*const vector<*double*>&* `data`): Converts a *vector* of *doubles* to a *vector* of **1D** `Tensor` pointers by iterating over the data and creating a **`TensorPtr`** for each value.
- *vector<**`TensorPtr`**>* **doubles_to_2d_tensors**(*const vector<vector<*double*>>&* `data`): Converts a **2D** *vector* of doubles to a **2D** *vector* of **`TensorPtr`**.

## Potential Improvements

- [ ] **Extend Tensor Support**: Implement support for multi-dimensional **`Tensor`** (**`Tensor`** with > 1 dimension).
- [ ] **Additional Layers**: Add more types of layers such as convolutional layers and recurrent layers.
- [ ] **Optimizers**: Implement more sophisticated optimization algorithms like Adam or RMSProp.
- [ ] **Concurrency**: The code currently runs on a single thread. Multi-threading or GPU acceleration can be explored for more computational efficiency or performance improvements on large datasets.
- [ ] **Model Serialization**: Add functionality to save and load trained models.