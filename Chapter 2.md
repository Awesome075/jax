# Numpy
`jax.numpy` is the reimplementation of the numpy library that is tightly integrated with the JAX ecosystem.

The syntax is almost the same as ***original Numpy***
```python
import numpy as np
import jax.numpy as jnp
```
It exists as a separate is because jax arrays have special properties that standard numpy arrays don't :
- They can be used with `jit`, `grad`, and `vmap`.
- They can live on accelerators like GPUs and TPUs

### JAX arrays are immutable

```python
import numpy as np
x = np.array([1,2,3])
x[0] = 99 
# X is now [99,2,3] 
```

In JAX, this will give an error. This is because JAX's pure function paradigm require that function can't have any side effects and modifying an array is a side effect.

### The JAX way `.at[]` Syntax

```python
import jax.numpy as jnp
x = jnp.array([1,2,3])

# We can not change x 
# Therefore we create a new array with some values of x as 
# it is while we update what we need using .at[]

y = x.at[0].set(99)

# y is [99,2,3]
# x is [1,2,3] 
```

We can also do more complex updates like `y = x.at[0].add(10)`

# Random Numbers in JAX

In Numpy, we might do this:
```python
import numpy as np
np.random.seed(0)
print(np.random.rand()) # First random number
print(np.random.rand()) # Second, different random number
```

Numpy has a hidden, global ***"random state"***. Everytime we call `np.random.rand()`, it uses this hidden state to generate a number and then secretly update the state. This is a side effect, which breaks the pure function rule that JAX relies on. If `jit` compiled a function with `np.random.rand()` inside, it might actually cache a single random number and return the same one every time.

JAX force us to manage the state manually. In JAX, random functions are deterministic. The same input key will always produce the same "random" output.

### The Key and Split Way in JAX
JAX approach is a simple two-step process:
- Create a master PRNG key (Pseudo random number generator key)
- Split this key everytime we need a new random number.

Think of it as having a magical seed, if we plant this seed directly we might lose it , what instead we can do is split it into two new seeds. We plant one (`subkey`) and keep the other (`key`) for the next time we need to split.

```python
from jax import random

#1. Create a master key from a seed.
key = random.PRNGKey(0)

#2. Need a random number? Split the key first.
key, subkey = random.split(key)

#3. Use the SUBKEY for our operation.
r1 = random.normal(subkey)

#4. Need another random number? split the key again.
key, subkey = random.split(key)

#5. Use the new SUBKEY.
r2 = random.normal(subkey)
```
Note :- We can also use `SUBKEY` for splitting, but don't use `SUBKEY` for splitting as it is a bad practice and may break the clean flow and increase the risk of bugs. One should always follows the good practices while writing code.

# jax.nn
`jax.nn` is a collection of common, neural network functions like "ReLU", "Softmax", and "Sigmoid". It does not provide the concept of "layers" (like nn.linear or Keras.layers.Dense) like Pytorch.nn or Keras.

`jax.nn` module primarily conatins two categories of functions:

### 1. Activation Functions
- `jax.nn.relu`
- `jax.nn.sigmoid`
- `jax.nn.tanh`
- `jax.nn.softmax`
- `jax.nn.log_softmax` - Numerically Stable Softmax

### 2. Initializers
- `jax.nn.initializers.glorot_normal` : Xavier Initialization
- `jax.nn.initializers.he_normal` : he initialization
- `jax.nn.initializers.ones()`
- `jax.nn.initializers.zeros()`

Example Usage
```python
from jax import random, nn

key = random.PRNGKey(0)

weight_matrix = nn.initializers.he_normal()(key, (1024,256))
```

# PyTrees
A pytree is a very simple concept, it is just a nested Python container like a dictionary or a list that holds JAX arrays as its "leaves". 

```python
# A typical pytree for a simple MLP

model_params = {
	'linear1':{
		'w': jnp.ones((1024, 128)), #Leaf
		'b': jnp.zeros(128)         #Leaf 
	}
	'linear2':{
		'w': jnp.ones((128, 10)),   #Leaf
		'b': jnp.zeros(10) 		 #Leaf
	}
}
```
The power of pytrees is that JAX's transformations (`jit`,`grad`,`vmap`,etc.) can operate on them directly. We can pass this entire `model_params` dictionary into a function, and JAX knows how to handle it.

### The main tool: `tree_map`
The most important function is `jax.tree_util.tree_map`. `tree_map` applies a function to every single leaf(array) in a pytree. It's like a super-powered `for` loop that automatically navigates any nested structure for us.

```python
from jax import tree_util

def square_leaf(x):
	return x**2

squared_params = tree_util.tree_map(square_leaf, model_params)
```
`tree_map` returns a new pytree with the exact same structure but with the function applied to every leaf.

### Why pytrees are important for Deep Learning?
This become incredibly powerful when paired with optimizers. Our optimizer's update step will calculate the changes for all our parameters. How do we apply these changes? With `tree_map`

```python
def update_step(params, gradients, learning_rate):
	def update_leaf(p, g):
		return p - learning_rate * g
	return tree_util.tree_map(update_leaf, params, gradients)
```

