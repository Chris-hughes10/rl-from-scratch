"""The toy task used throughout the notebook: integer arithmetic with a 0/1 reward."""

import ast
import operator
import random
import re

_BINARY_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    """Evaluate `+ - * /` arithmetic by walking the syntax tree; anything else is rejected."""
    if len(expression) > 200:
        raise ValueError("expression too long")

    def evaluate_node(node):
        if isinstance(node, ast.Expression):  # the root of the tree
            return evaluate_node(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):  # a number
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:  # e.g. 2 + 3
            apply_op = _BINARY_OPS[type(node.op)]
            return apply_op(evaluate_node(node.left), evaluate_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:  # e.g. -5
            apply_op = _UNARY_OPS[type(node.op)]
            return apply_op(evaluate_node(node.operand))
        raise ValueError(f"unsupported expression: {expression!r}")  # names, calls, **, ...

    return evaluate_node(ast.parse(expression, mode="eval"))


def make_problems(num_problems: int, seed: int, hard: bool = False) -> list[dict]:
    rng = random.Random(seed)
    problems = []
    for _ in range(num_problems):
        if hard:  # e.g. "482 * 917 - 3051": too big to multiply reliably without a calculator
            expression = " ".join([str(rng.randint(100, 999)), "*", str(rng.randint(100, 999)), "-", str(rng.randint(1000, 9999))])
        else:     # e.g. "37 + 12 * 5"
            expression = " ".join([
                str(rng.randint(2, 99)), rng.choice("+-*"),
                str(rng.randint(2, 99)), rng.choice("+-*"),
                str(rng.randint(2, 30)),
            ])
        problems.append({"question": f"What is {expression}?", "answer": int(safe_eval(expression))})
    return problems


BOXED = re.compile(r"\\boxed\{\s*(-?\d[\d,]*)\s*\}")


def reward_fn(text: str, answer: int) -> float:
    """1.0 if the last \\boxed{...} in `text` holds `answer`, else 0.0."""
    boxed_numbers = BOXED.findall(text)  # e.g. ["12", "1,234"]
    if not boxed_numbers:
        return 0.0
    final_answer = int(boxed_numbers[-1].replace(",", ""))
    return 1.0 if final_answer == answer else 0.0
