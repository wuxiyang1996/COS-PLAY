"""Smoke test for the `osworld` conda env (xlang-ai/OSWorld)."""
from __future__ import annotations

import sys

failures: list[tuple[str, str]] = []


def check(label: str, fn, required: bool = True) -> None:
    try:
        out = fn()
        print(f"  [OK]   {label}{(': ' + str(out)) if out else ''}")
    except Exception as exc:
        if required:
            failures.append((label, str(exc)))
            print(f"  [FAIL] {label}: {exc}")
        else:
            print(f"  [WARN] {label}: {exc}")


print(f"Python {sys.version.split()[0]}\n")

print("Core:")
check("gymnasium",            lambda: __import__('gymnasium').__version__)
check("torch",                lambda: __import__('torch').__version__)
check("transformers",         lambda: __import__('transformers').__version__)
check("Pillow",               lambda: __import__('PIL').__version__)
check("numpy",                lambda: __import__('numpy').__version__)
check("pandas",               lambda: __import__('pandas').__version__)
print()

print("OSWorld / desktop-env:")
check("desktop_env",          lambda: (__import__('desktop_env', fromlist=['*']), "imported")[-1])
check("desktop_env.desktop_env", lambda: (__import__('desktop_env.desktop_env', fromlist=['*']), "imported")[-1])


def _desktop_cls() -> str:
    from desktop_env.desktop_env import DesktopEnv  # noqa: F401
    return "DesktopEnv class importable"
check("DesktopEnv class",     _desktop_cls)

check("docker SDK",           lambda: __import__('docker').__version__, required=False)
check("playwright",           lambda: __import__('playwright').__version__ if hasattr(__import__('playwright'), '__version__') else "imported", required=False)
check("pyautogui",            lambda: __import__('pyautogui').__version__, required=False)
check("easyocr",              lambda: __import__('easyocr').__version__, required=False)
print()

print("=" * 50)
if failures:
    print(f"{len(failures)} REQUIRED check(s) FAILED:")
    for label, err in failures:
        print(f"  - {label}: {err}")
    sys.exit(1)
print("All required checks passed.")
print("Note: actually running OSWorld tasks requires a VM backend")
print("      (Docker: docker pull happysixd/osworld-docker, or VMware).")
print("=" * 50)
