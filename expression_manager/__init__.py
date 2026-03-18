"""Expression manager for factor expression parsing, execution, and AST analysis."""

from expression_manager.expr_parser import parse_expression, parse_symbol
from expression_manager.factor_ast import (
    parse_expression as parse_ast,
    find_largest_common_subtree,
    compare_expressions,
    count_all_nodes,
    count_free_args,
    count_unique_vars,
)

__all__ = [
    "parse_expression",
    "parse_symbol",
    "parse_ast",
    "find_largest_common_subtree",
    "compare_expressions",
    "count_all_nodes",
    "count_free_args",
    "count_unique_vars",
]
