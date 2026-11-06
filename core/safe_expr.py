"""
core/safe_expr.py

Safe AST-based arithmetic expression evaluator for pandas DataFrames.
Strictly allows only column names, numbers, basic arithmetic (+, -, *, /),
and parentheses. Blocks arbitrary code execution, function calls, attributes,
and imports.
"""

import ast
import operator
from typing import Any, Set
import pandas as pd
import numpy as np

# Allowed binary and unary operators
ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains disallowed tokens, function calls, or attributes."""
    pass


def extract_variables(expr: str) -> Set[str]:
    """Parses expression and returns set of column names referenced."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"Invalid arithmetic syntax: {e}")

    variables = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            variables.add(node.id)
    return variables


def _eval_node(node: ast.AST, df: pd.DataFrame) -> Any:
    """Recursively evaluates an AST node strictly against DataFrame columns and numbers."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, df)

    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(f"Disallowed literal value '{node.value}'. Only numbers allowed.")
        return node.value

    # Compatibility for Python <= 3.7 Num node
    elif hasattr(ast, "Num") and isinstance(node, ast.Num):
        return node.n

    elif isinstance(node, ast.Name):
        col = node.id
        if col not in df.columns:
            raise KeyError(f"Column '{col}' referenced in expression does not exist in data.")
        return df[col]

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BINOPS:
            raise UnsafeExpressionError(f"Disallowed operator '{op_type.__name__}'. Only +, -, *, / allowed.")
        left_val = _eval_node(node.left, df)
        right_val = _eval_node(node.right, df)
        op_func = ALLOWED_BINOPS[op_type]
        return op_func(left_val, right_val)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARYOPS:
            raise UnsafeExpressionError(f"Disallowed unary operator '{op_type.__name__}'.")
        operand_val = _eval_node(node.operand, df)
        op_func = ALLOWED_UNARYOPS[op_type]
        return op_func(operand_val)

    else:
        raise UnsafeExpressionError(f"Disallowed syntax element '{type(node).__name__}'. No functions or attributes allowed.")


def evaluate_safe_expr(expr: str, df: pd.DataFrame) -> pd.Series:
    """
    Safely evaluates an arithmetic expression on a pandas DataFrame.
    Example: '(TLR + RPC) / 2' or 'score - 50'
    """
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafeExpressionError("Expression must be a non-empty string.")

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"Syntax error in expression: {e}")

    result = _eval_node(tree, df)

    # Convert scalar to Series if needed
    if isinstance(result, (int, float)):
        return pd.Series(result, index=df.index)
    elif isinstance(result, pd.Series):
        return result
    else:
        return pd.Series(result, index=df.index)

