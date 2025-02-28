from typing import Type


def assert_type(obj: object, target_type: Type):
    """Mypy helper to enforce type correctness at runtime"""
    if not isinstance(obj, target_type):
        raise TypeError(f"Expected object to be type {target_type}, got {type(obj)}")
