r"""
Initialization module for vector operations.
This module imports and exposes the main vector classes and functions.
"""

from .vector import Vector
from .operations import (
    add,
    sub,
    mul,
    div,
    dot,
    cross,
    abs_vector,
    compare
)

__all__ = [
    'Vector',
    'add',
    'sub',
    'mul',
    'div',
    'dot',
    'cross',
    'abs_vector',
    'compare'
]