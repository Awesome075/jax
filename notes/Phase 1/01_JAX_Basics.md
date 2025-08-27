# JIT

A type of function transformation in JAX (basically making a copy of a function and modifying it)
Just in time compilation
Uses XLA Compiler which does :-

1. Trace the Function with inputs (First Run)
2. Compile to Machine Code
3. Cache and reuse the Machine Code for the traced function

Two Ways to use ```jax.jit``` (this can be said for other jax transformations such as `grad` & `vmap`)

```python
import jax
import jax.numpy as jnp

@jax.jit
def fun(x):
    return jnp.sin(jnp.cos(x) * 2.0)
```

or

```python
def fun(x):
  return jnp.sin(jnp.cos(x) * 2.0)

a = jax.jit(fun)
result = a(5)
```

Jax is designed to work on pure functions.
A pure function is one where:

- The output depends only on input arguments.
- It has no ***side effects***. Side effects are anything other than returning a value, like printing to the screen, modifying global variable, or writing to a file.

Use `where` instead of `if-else` inside jit'd function
Use `partial` from `functools` or `jax.tree_util.Partial` for static variables in functions, it's a good practice.

# VMAP (For-Loop Killer)

Vectoring Map or VMAP let us write a function that works on a single example, and it will automatically transform it into a function that works on a batch of samples.

### How to use `vmap` and the `in_axes` parameter

The main argument to control `vmap` is `in_axes`. It tells `vmap` which dimension of our input array is the batch dimension, `in_axes` is a tuple where each element corresponds to the argument of the function we are mapping.

- `0` : The most common value. It means the batch dimension is the first axis (meaning the data is like in a list of list or list of array or list of tuple or list of int , etc.)
- `None` : Tells `vmap` not to map over this argument. The same argument will be applied to every single element in the batch. This is used for shared data, like model weights.
- ***Default***: If we did not provide `in_axes`, the default is `0` for all arguments.

### The most common pattern: `in_axes=(None,0)`

In deep learning, we almost always appply the same set of weights to a whole batch of data. This pattern is perfect for that.

```python
import jax
from jax.numpy import jnp

def predict(weights, input_vector):
  return jnp.dot(weights,input_vector)

weights = jnp.ones((2,3))
batched_inputs = jnp.ones((10,3))

batched_predict = jax.vmap(predict, in_axes=(None,0))

result = batched_predict(weights, batched_inputs)
```

### Composability: Stacking with jit

Like all JAX transformations, `vmap` can be combined with others. We can stack it with `jit` to create a function that is both vectorized for batching and compiled for speed.

```python
@jax.jit
@jax.vmap
def fast_and_batched_predict(weights, input_vector):
  return jnp.dot(weights, input_vector)
```

# GRAD

`jax.grad` is a function that performs automatic differentiation. It takes a python function that computes a value (scaler) and gives us back a new function that computes its gradient.

### Example of $f(x) = x^2$

```python
import jax
import jax.numpy as jnp

def square(x):
  return x**2

grad = jax.grad(square)

gradient_at_3 = grad(3.0) # JAX often requires float

print(f"The gradient of x^2 at x=3 is: {gradient_at_3}")
#Output: The gradient of x^2 at x=3 is: 6.0
```

### Using grad for ML

The `argnums` argument of `jax.grad` lets us specify which argument of the funtion we want to differentiate with respect to.

```python
def predict(weights, inputs):
  return jnp.dot(weights,inputs)


def loss_fn(weights, inputs, targets):
  predictions = predict(weights, inputs)
  return jnp.mean((predictions-targets)**2) #MSE Error

#Differentiate with respect to weights
grad_loss_fn = jax.grad(loss_fn, argnums=0)

weights = jnp.array([1.0, 2.0, 3.0])
inputs = jnp.array([0.5, 0.2, 0.1])
targets = jnp.array([1.5])

gradients = grad_loss_fn(weights, inputs, targets)
```

### Efficiency with `jax.value_and_grad`

In a training loop, we need both the loss value (for printing/logging) and the gradients (for updating weights). `jax.value_and_grad` is an efficient transformation that returns both at the same time avoiding the need to run the forward pass twice.

```python
value_and_grad_fn = jax.value_and_grad(loss_fn, argnums=0)

loss, gradients = value_and_grad_fn(weights, inputs, targets)
```


