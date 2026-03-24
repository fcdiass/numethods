# NuMethods

Numerical methods for engineers in Python – built to be studied and understood, not just used.

## About

This library implements classical numerical methods used in engineering: root finding, linear systems, optimization, integration, differential equations, and more. The goal is a personal reference that surfaces the intermediate steps that production libraries hide – iteration history, convergence behavior, per-step error estimates – so that the method itself remains visible and inspectable.

Production libraries such as SciPy solve efficiently but return only the final result. NuMethods is a companion for anyone who wants to understand *how* a method works, not just *that* it works.

NumPy is the single low-level dependency. All numerical methods are built strictly on top of it.


## Installation

```bash
pip install numethods
```

```bash
uv add numethods
```


## Quick Start

```python
import numpy as np
from numethods.roots.bracketing_methods import bisection

# Terminal velocity of a falling parachutist (drag coefficient unknown)
def f(c):
    return (667.38 / c) * (1 - np.exp(-0.146843 * c)) - 40

result = bisection(f, xl=12, xu=16, verbose=True)
# ------------------------------------------------------------------------------------------------
#                                   Bisection method iterations
# ------------------------------------------------------------------------------------------------
#   i |                   xl |                   xu |                   xr |                   ea
#   1 | 12.0000000000000000 | 16.0000000000000000 | 14.0000000000000000 |   0.1428571428571429
#   2 | 14.0000000000000000 | 16.0000000000000000 | 15.0000000000000000 |   0.0666666666666667
#   ...

print(result.solution)   # 14.780208468437195
print(result.nit)        # number of iterations
print(result.nfev)       # number of function evaluations

# Full iteration history is available
for step in result.history:
    print(step.i, step.sol, step.ea)
```


## Topics

Currently implemented: bracketing methods for root finding (bisection and false position).

See [ROADMAP.md](ROADMAP.md) for the full list of topics planned.


## License

MIT

## Author

Francisco Corrêa Dias<br>
PhD in Civil Engineering – PUC-Rio<br>
Researcher in Computational Mechanics
