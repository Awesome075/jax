# My JAX Learning Journey

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository documents my process of learning JAX, from the fundamentals concepts to implementing Deep Learning models. It contains both my personal easy to understand notes and the code for a series of progressively complex projects.

## Table of Contents

- [My JAX Learning Journey](#my-jax-learning-journey)
  - [Table of Contents](#table-of-contents)
  - [JAX Phase 1](#jax-phase-1)
    - [The Core Concepts](#the-core-concepts)
    - [Project 1: Multi-Layer Perceptron in Functional JAX](#project-1-multi-layer-perceptron-in-functional-jax)
  - [Setup and Installation](#setup-and-installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## JAX Phase 1

This section contains my write-ups on the core concepts of JAX, based on the official documentation and my own experiments. Also a project to build an MLP in jax based on the concepts learned.

### The Core Concepts

-   **Chapter 1: [The Basics of JAX](https://github.com/Awesome075/jax/blob/main/notes/Phase%201/01_JAX_Basics.md)**
    -   Covers the core transformations: `jit` (Just-In-Time Compilation), `vmap`(Vectorization), and `grad`(Automatic Differentiation).

-   **Chapter 2: [Core Components](https://github.com/Awesome075/jax/blob/main/notes/Phase%201/02_Core_Components.md)**
    -   Explores essential building blocks: `jax.numpy`, explicit random numbers generation with `PRNGKey`, `jax.nn` for functional building blocks, and `Pytrees` for state management.

### Project 1: Multi-Layer Perceptron in Functional JAX

-   **[Notebook](https://github.com/Awesome075/jax/blob/main/notes/Phase%201/MLP_in_JAX.ipynb)**
    -   A complete implementation of a neural network from scratch, built in a functional style using JAX. This project was a direct translation of an [object-oriented Numpy implementation](https://github.com/Awesome075/Neural-Networks-Numpy-) , demonstrating a full grasp of JAX's core concepts discused in Chapter 1 and Chapter 2.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Awesome075/jax.git
    cd jax
    ```

2. **Create a virtual environment:**
```bash
python -m venv venv
```

3. **Activate the Environment**
    
    - *On Windows:*
        ```bash
        venv\Scripts\activate
        ```

    - *On Linux/macOS:*
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project is organized into notebooks and markdown files. You can explore the `notes` directory to follow my learning path or run the project notebooks to see the code in action.

To run the Jupyter notebooks, first set up the kernel:

```bash
python -m ipykernel install --user --name=jax-env
```

Then, you can run the notebook using:

```bash
jupyter notebook notes/"Phase 1"/MLP_in_JAX.ipynb
```

Inside the notebook, make sure to select the `jax-env` kernel.

## Contributing

Contributions are welcome! If you have any suggestions or find any bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
