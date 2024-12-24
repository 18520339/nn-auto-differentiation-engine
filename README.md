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

# Getting Started

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

The engine is organized into several header files (`.hpp`) located in the [n2n_autograd](./n2n_autograd/) folder, each including classes and functions responsible for different aspects of the Neural Network and auto-differentiation operations.

## I. [`tensor.hpp`](./n2n_autograd/tensor.hpp) | Tensor Class and Auto-differentiation Engine

Contain the `Tensor` class, a core data structure of the **autograd** engine, representing a node or scalar value in the computation graph. It supports **automatic differentiation** by recording/maintaining references to its child **tensors** and the operations that produced it. 

When operations are performed on **tensors** (e.g., `addition`, `multiplication`), new **tensors** are created, and the graph is dynamically built. During **backpropagation**, the `gradients` with respect to each **tensor** are computed by traversing this graph in **reverse topological order**. 

Here, the `local_backward` function will specify how the `gradient` is computed locally for the **tensor** based on its `children`. For example, in `multiplication`, the `gradient` with respect to each operand is the product of the other operand and the upstream `gradient`.

### 1.1. Private Members

- *double* `data`: The numerical value of the **tensor**.
- *double* `gradient`: The accumulated gradient computed during **backpropagation**.
- *string* `label`: An optional label for identification.
- Internal pointers and functions for tracking operations and dependencies:
  - *function<void(const `Tensor`\*)>* `local_backward`: Lambda function to compute local `gradients` with respect to the **tensor**.
  - *set<`TensorPtr`>* `children`: A set of child nodes (**tensors**) in the computational graph that are inputs to the operation producing this **tensor**.
  - *string* `operation`: The operation that produced this **tensor**.

👉 *vector<`Tensor`\*>* **topological_sort**(): Performs a topological sort of the computation graph, returning nodes in the order they should be processed during **backpropagation**.

### 1.2. Public Methods

👉 **Constructors**:
- **`Tensor`**(*double* `_data`, *const string* `_label = ""`): Initializes a **tensor** with a value and an optional label.
- **`Tensor`**(*double* `_data`, *const set<**`TensorPtr`**>&* `_children`, *const string&* `_operation`): Initializes a **tensor** resulting from an operation.

👉 **Operator Supports with Automatic Gradient Tracking**:
- Arithmetic operations: `+`, `-`, `*`, `/`, `pow`.
- Unary operations: `-`, `exp`, `log`.
- Activation Functions: `sigmoid`, `tanh`, `relu`.
  - *static **`TensorPtr`*** **create**(*double* `_data`, *string* `_label = ""`): Static Factory method to create a shared pointer to a Tensor.
  - *void* **backward**(): Performs **backpropagation** to compute `gradients` for all **tensors** throughout the computation graph using reverse-mode **automatic differentiation**.

### 1.3. Usage Example

Create **tensors** and perform operations as you would with scalar values. Call the `backward()` method on the final output **tensor** to compute `gradients`.

```cpp
auto x = Tensor::create(2.0, "x");
auto w = Tensor::create(3.0, "w");
auto y = x * w;
y->backward();

cout << "Gradient of x: " << x->gradient << endl; // Outputs 3.0
cout << "Gradient of w: " << w->gradient << endl; // Outputs 2.0
```

## II. [`converters.hpp`](./n2n_autograd/converters.hpp) | Utility Functions for Data Conversion

Contain utility functions to facilitate the conversion of different data types commonly used in the preprocessing, particularly when preparing data for training as well as inference by converting raw data into tensors suitable for model input.

### 2.1. *any* Conversion

- *any* **str_to_any**(*const string&* `str`): Converts a string to an *any* type, attempting to parse it as an *int*, *double*, or leaving it as a string based on its content.
- *double* **any_to_double**(*const any&* `input`): Safely converts an *any* type (expected to hold an *int* or *double*) to a *double*, supporting both *int* and *double* internally.
- *vector<*int*>* **anys_to_ints**(*const vector<any>&* `inputs`): Converts a *vector of any* types (each expected to hold an *int*) to a *vector of ints*.
- *vector<*double*>* **anys_to_doubles**(*const vector<any>&* `inputs`): Converts a *vector of any* types (each expected to hold an *int* or *double*) to a *vector of doubles*.

### 2.2. One-Hot Encoding

- *vector<vector<*int*>>* **anys_to_1hots**(*const vector<any>&* `y_raw`, *int* `num_classes`): Converts a *vector* of class labels to one-hot encoded *vectors*. It will creates a *vector of vectors* (`n_samples` x `num_classes`), initializing all elements to `0`, and set the index corresponding to each class label to `1` in the one-hot *vector*.
- *vector<vector<**`TensorPtr`**>>* **anys_to_1hot_tensors**(*const vector<*any*>&* `y_raw`, *int* `num_classes`): Similar to **anys_to_1hots** but return a *vector* of *vectors* (instead of *int*) containing **`TensorPtr`** representing one-hot encodings.

### 2.3. `Tensor` Conversion

- *vector<**`TensorPtr`**>* **doubles_to_1d_tensors**(*const vector<*double*>&* `data`): Converts a *vector* of *doubles* to a *vector* of **1D** `Tensor` pointers by iterating over the data and creating a **`TensorPtr`** for each value.
- *vector<**`TensorPtr`**>* **doubles_to_2d_tensors**(*const vector<vector<*double*>>&* `data`): Converts a **2D** *vector* of doubles to a **2D** *vector* of **`TensorPtr`**.

## Potential Improvements

- [ ] **Extend Tensor Support**: Implement support for multi-dimensional `Tensor` (`Tensor` with more than 1 dimension).
- [ ] **Additional Layers**: Add more types of layers such as convolutional layers and recurrent layers.
- [ ] **Optimizers**: Implement more sophisticated optimization algorithms like Adam or RMSProp.
- [ ] **Concurrency**: The code currently runs on a single thread. Multi-threading or GPU acceleration can be explored for more computational efficiency or performance improvements on large datasets.
- [ ] **Model Serialization**: Add functionality to save and load trained models.