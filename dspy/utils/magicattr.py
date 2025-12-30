"""
Compatibility layer for magicattr that works with Python 3.14+

This module provides a patched version of magicattr's functionality
that is compatible with Python 3.14's removal of ast.Num and ast.Str.

Based on magicattr 0.1.6 by Jairus Martin (MIT License)
https://github.com/frmdstryr/magicattr
"""
import ast
import sys
from functools import reduce

_AST_TYPES = (ast.Name, ast.Attribute, ast.Subscript, ast.Call)
_STRING_TYPE = str


def get(obj, attr, **kwargs):
    """A getattr that supports nested lookups on objects, dicts, lists, and
    any combination in between.
    """
    for chunk in _parse(attr):
        try:
            obj = _lookup(obj, chunk)
        except Exception as ex:
            if "default" in kwargs:
                return kwargs["default"]
            else:
                raise ex
    return obj


def set(obj, attr, val):
    """A setattr that supports nested lookups on objects, dicts, lists, and
    any combination in between.
    """
    obj, attr_or_key, is_subscript = lookup(obj, attr)
    if is_subscript:
        obj[attr_or_key] = val
    else:
        setattr(obj, attr_or_key, val)


def delete(obj, attr):
    """A delattr that supports deletion of a nested lookups on objects,
    dicts, lists, and any combination in between.
    """
    obj, attr_or_key, is_subscript = lookup(obj, attr)
    if is_subscript:
        del obj[attr_or_key]
    else:
        delattr(obj, attr_or_key)


def lookup(obj, attr):
    """Like get but instead of returning the final value it returns the
    object and action that will be done.
    """
    nodes = tuple(_parse(attr))
    if len(nodes) > 1:
        obj = reduce(_lookup, nodes[:-1], obj)
        node = nodes[-1]
    else:
        node = nodes[0]
    if isinstance(node, ast.Attribute):
        return obj, node.attr, False
    elif isinstance(node, ast.Subscript):
        return obj, _lookup_subscript_value(node.slice), True
    elif isinstance(node, ast.Name):
        return obj, node.id, False
    raise NotImplementedError("Node is not supported: %s" % node)


def _parse(attr):
    """Parse and validate an attr string"""
    if not isinstance(attr, _STRING_TYPE):
        raise TypeError("Attribute name must be a string")
    nodes = ast.parse(attr).body
    if not nodes or not isinstance(nodes[0], ast.Expr):
        raise ValueError("Invalid expression: %s" % attr)
    return reversed([n for n in ast.walk(nodes[0]) if isinstance(n, _AST_TYPES)])


def _lookup_subscript_value(node):
    """Lookup the value of ast node on the object.

    Compatible with Python 3.14+ which removed ast.Num and ast.Str
    """
    if isinstance(node, ast.Index):
        node = node.value

    # Python 3.14+ uses ast.Constant for all constants
    if isinstance(node, ast.Constant):
        return node.value

    # Fallback for older Python versions
    if sys.version_info < (3, 14):
        # Handle numeric indexes
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return node.n
        # Handle string keys
        elif hasattr(ast, "Str") and isinstance(node, ast.Str):
            return node.s

    # Handle negative indexes
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        if isinstance(operand, ast.Constant):
            return -operand.value
        # Fallback for older Python
        elif sys.version_info < (3, 14) and hasattr(ast, "Num") and isinstance(operand, ast.Num):
            return -operand.n

    raise NotImplementedError("Subscript node is not supported: %s" % ast.dump(node))


def _lookup(obj, node):
    """Lookup the given ast node on the object."""
    if isinstance(node, ast.Attribute):
        return getattr(obj, node.attr)
    elif isinstance(node, ast.Subscript):
        return obj[_lookup_subscript_value(node.slice)]
    elif isinstance(node, ast.Name):
        return getattr(obj, node.id)
    elif isinstance(node, ast.Call):
        raise ValueError("Function calls are not allowed.")
    raise NotImplementedError("Node is not supported: %s" % node)
