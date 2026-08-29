"""Python-level guard for generated training scripts.

This is defense in depth, not an operating-system sandbox. It blocks ordinary
Python file reads of raw outcomes/secrets/prior runs plus child subprocesses and
network sockets. The executor still hashes protected files before and after a
run and recomputes validation metrics in the parent.
"""
from __future__ import annotations

import json
import os
import runpy
import sys


def _normal(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _install_guard() -> None:
    allowed_run = _normal(os.environ["KUAIRAND_GUARD_RUN_DIR"])
    solution_path = _normal(os.environ["KUAIRAND_GUARD_SOLUTION_PATH"])
    blocked = [
        _normal(path)
        for path in json.loads(os.environ["KUAIRAND_GUARD_BLOCKED_PATHS"])
    ]
    protected = [
        _normal(path)
        for path in json.loads(os.environ["KUAIRAND_GUARD_PROTECTED_PATHS"])
    ]

    def forbidden(path: str) -> bool:
        return any(path == root or _within(path, root) for root in blocked)

    def mutation_forbidden(path: str) -> bool:
        return (path == solution_path or _within(solution_path, path)
                or any(path == root or _within(root, path)
                       for root in protected)
                or not _within(path, allowed_run))

    def path_arg(value) -> str | None:
        if isinstance(value, int):
            return None
        try:
            return _normal(os.fsdecode(value))
        except (TypeError, ValueError):
            return None

    def audit(event, args):
        if event == "open" and args:
            path = path_arg(args[0])
            if path is None:
                return
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            writes = ((isinstance(mode, str)
                       and any(mark in mode for mark in ("w", "a", "x", "+")))
                      or (isinstance(flags, int)
                          and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND
                                            | os.O_CREAT | os.O_TRUNC))))
            if writes and mutation_forbidden(path):
                raise PermissionError(
                    "generated-code guard blocked protected-file mutation")
            if _within(path, allowed_run):
                return
            if forbidden(path):
                raise PermissionError(
                    f"generated-code guard blocked file access: {path}")
        if event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod",
                     "os.truncate"} and args:
            path = path_arg(args[0])
            # The output directory already exists, and libraries commonly call
            # makedirs(..., exist_ok=True). That harmless probe must remain usable.
            if event == "os.mkdir" and path == allowed_run:
                return
            if path is not None and mutation_forbidden(path):
                raise PermissionError(
                    f"generated-code guard blocked filesystem mutation: {event}")
        if event in {"os.rename", "os.link", "os.symlink"} and len(args) >= 2:
            paths = [path_arg(args[0]), path_arg(args[1])]
            if any(path is not None and mutation_forbidden(path)
                   for path in paths):
                raise PermissionError(
                    f"generated-code guard blocked filesystem mutation: {event}")
        if event in {
            "subprocess.Popen", "os.system", "os.posix_spawn",
            "os.spawn", "os.exec", "os.startfile",
            "socket.connect", "socket.__new__", "ctypes.dlopen",
            "ctypes.dlsym", "ctypes.call_function",
        }:
            raise PermissionError(
                f"generated-code guard blocked operation: {event}")

    sys.addaudithook(audit)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("child_guard requires a generated solution path")
    code_path = os.path.abspath(sys.argv[1])
    os.environ["KUAIRAND_GUARD_SOLUTION_PATH"] = code_path
    sys.argv = [code_path, *sys.argv[2:]]
    # Load the trusted numerical runtime before the hook. NumPy legitimately
    # resolves native symbols during import; later ctypes calls remain blocked.
    import train_lib  # noqa: F401
    _install_guard()
    runpy.run_path(code_path, run_name="__main__")


if __name__ == "__main__":
    main()
