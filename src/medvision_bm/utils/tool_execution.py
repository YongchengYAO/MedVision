import ast
import contextlib
import io
import math

# Names callable directly (bare-name calls) inside the sandbox.
_ALLOWED_CALL_NAMES = frozenset(
    {"print", "round", "abs", "min", "max", "len", "sum"}
)
# Only attributes on this module name are reachable (e.g. math.sqrt).
_ALLOWED_ATTR_ROOT = "math"


class _UnsafeCode(Exception):
    """Raised when AST validation rejects a construct."""


def _validate_ast(tree: ast.AST) -> None:
    """Walk the AST and reject anything outside the arithmetic allowlist.

    Allowed: numeric/string constants, bare-name loads/stores, plain-name
    assignments (incl. tuple unpacking), arithmetic (BinOp/UnaryOp/Compare/
    BoolOp), f-strings, tuples, `import math`, calls to a small builtin
    allowlist, and calls to math.<func>. Everything else is rejected.
    """
    for node in ast.walk(tree):
        # Block any dunder / private attribute access (e.g. __class__,
        # __globals__, __subclasses__) regardless of where it appears.
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise _UnsafeCode(f"attribute '{node.attr}'")
            # Only `math.<name>` attribute chains are permitted.
            if not (isinstance(node.value, ast.Name)
                    and node.value.id == _ALLOWED_ATTR_ROOT):
                raise _UnsafeCode("attribute access")
            continue

        # Subscript can be used for dunder traversal (e.g. __bases__[0]); never needed here.
        if isinstance(node, ast.Subscript):
            raise _UnsafeCode("subscript")

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _ALLOWED_CALL_NAMES:
                    raise _UnsafeCode(f"call to '{func.id}'")
            elif isinstance(func, ast.Attribute):
                # math.<name> only; dunder/other roots already rejected above
                # when the Attribute node itself is walked.
                if (func.attr.startswith("_")
                        or not (isinstance(func.value, ast.Name)
                                and func.value.id == _ALLOWED_ATTR_ROOT)):
                    raise _UnsafeCode("call target")
            else:
                raise _UnsafeCode("call target")
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom) or names != [_ALLOWED_ATTR_ROOT]:
                raise _UnsafeCode("import (only 'import math' is allowed)")
            continue

        # Names used as assignment targets must be plain identifiers; reject
        # anything else (handled implicitly by the node allowlist below).
        if isinstance(
            node,
            (
                ast.Module,
                ast.Expr,
                ast.Assign,
                ast.Name,
                ast.Load,
                ast.Store,
                ast.Constant,
                ast.BinOp,
                ast.UnaryOp,
                ast.BoolOp,
                ast.Compare,
                ast.Tuple,
                ast.List,
                ast.JoinedStr,
                ast.FormattedValue,
                # operators / comparators
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.Pow,
                ast.USub,
                ast.UAdd,
                ast.And,
                ast.Or,
                ast.Not,
                ast.Eq,
                ast.NotEq,
                ast.Lt,
                ast.LtE,
                ast.Gt,
                ast.GtE,
                ast.keyword,
                ast.alias,
            ),
        ):
            continue

        raise _UnsafeCode(f"syntax '{type(node).__name__}'")


def safe_exec_python(code: str) -> str:
    """Execute Python code in a restricted environment with only math available."""

    # AST allowlist: reject dunder traversal / arbitrary calls / imports before
    # any code runs. This is the primary sandbox; the restricted builtins below
    # are defense-in-depth.
    try:
        tree = ast.parse(code, "<string>", "exec")
        _validate_ast(tree)
    except _UnsafeCode as e:
        return "ERROR: disallowed " + str(e)
    except SyntaxError as e:
        return "ERROR: " + str(e)

    # Intercepts all import() calls within exec'd code; only "math" is permitted.
    def restricted_import(name, *args, **kwargs):
        if name != "math":
            raise ImportError(f"Import of '{name}' is not allowed")
        return math

    namespace = {
        "__builtins__": {
            "__import__": restricted_import,
            "print": print,
            "round": round,
            "abs": abs,
            "min": min,
            "max": max,
            "len": len,
            "sum": sum,
        }
    }

    buf = io.StringIO()
    try:
        compiled = compile(tree, "<string>", "exec")
        with contextlib.redirect_stdout(buf):
            exec(compiled, namespace)  # noqa: S102 - restricted namespace, math-only
        return buf.getvalue().strip()
    except Exception as e:
        return "ERROR: " + str(e)
