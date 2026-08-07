#!/usr/bin/env bash
# dump_evaluate_head.sh — print the top of d4_harness.evaluate + how KGE calls it,
# to a file so a tmux scrollback limit can't truncate it.
set -euo pipefail
export PYTHONPATH="${PWD}/scripts"
OUT="evaluate_head.txt"

python3 - > "$OUT" 2>&1 <<'PY'
import importlib, inspect
h = importlib.import_module("d4_harness")

def head(fn, n=45):
    try:
        src = inspect.getsource(fn).splitlines()
    except OSError:
        return "(source unavailable)"
    return "\n".join(src[:n])

print("========== evaluate() — first 45 lines ==========")
print(f"signature: {inspect.signature(h.evaluate)}\n")
print(head(h.evaluate, 45))

# What is `ref`? Show any builder/dataclass that likely constructs it.
print("\n========== candidates for the `ref` object ==========")
for name in dir(h):
    obj = getattr(h, name)
    if inspect.isclass(obj) and any(k in name.lower() for k in ("ref", "context", "bundle", "data", "eval")):
        print(f"\n--- class {name} ---")
        print(head(obj, 30))
    if inspect.isfunction(obj) and any(k in name.lower() for k in ("ref", "load", "build", "prepare", "main", "run")):
        print(f"\n--- def {name}{inspect.signature(obj)} ---")
        print(head(obj, 30))
PY

echo "wrote $OUT"
echo "----- preview (first 60 lines) -----"
sed -n '1,60p' "$OUT"