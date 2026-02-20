r"""
Initialization module for mathematical expansion operations.
This module imports and exposes the main mathematical expansion functions.
"""

from .operations import (
    mul,
    div,
    add,
    sub,
    iterate,
    is_vector,
    ThreadManager
)

__all__ = [
    'mul',
    'div',
    'add',
    'sub',
    'iterate',
    'is_vector',
    'ThreadManager'
]