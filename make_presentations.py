"""Собирает презентации-отчёты по всем лабораторным работам одной командой."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABS = ["lab1", "lab2", "lab3", "lab4", "lab5", "lab6"]

failed = []
for lab in LABS:
    script = ROOT / lab / "make_presentation.py"
    print(f"\n=== {lab} ===")
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode:
        failed.append(lab)

if failed:
    print(f"\nне собрались: {', '.join(failed)}")
    sys.exit(1)
print("\nвсе презентации собраны")
