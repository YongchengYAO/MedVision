import contextlib
import io
import math


def safe_exec_python(code: str) -> str:
    """Execute Python code in a restricted environment with only math available."""

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
        }
    }

    buf = io.StringIO()
    try:
        compiled = compile(code, "<string>", "exec")
        with contextlib.redirect_stdout(buf):
            exec(compiled, namespace)  # noqa: S102 - restricted namespace, math-only
        return buf.getvalue().strip()
    except Exception as e:
        return "ERROR: " + str(e)
