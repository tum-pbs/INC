"""
Solvers for 1D and 2D fluid dynamics problems
"""
try:
    from .solver_1d import BurgersSolverTorch, KSSolverTorch, weno_reconstruction
    __all__ = ['BurgersSolverTorch', 'KSSolverTorch', 'weno_reconstruction']
except ImportError:
    # Handle import errors gracefully
    __all__ = []