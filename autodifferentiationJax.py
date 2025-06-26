import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x[0] * x[1], jnp.sin(x[0])])

x = jnp.array([2.0, 3.0])
v = jnp.array([1.0, 0.0])

y, jvp = jax.jvp(f, (x,), (v,))

print("Function output:", y)
print("Jacobian-vector product (JVP):", jvp)
