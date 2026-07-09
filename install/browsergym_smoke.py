"""Smoke test for the `browsergym` conda env (ServiceNow/BrowserGym)."""
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
check("playwright",           lambda: __import__('playwright').__version__ if hasattr(__import__('playwright'), '__version__') else "imported")
check("Pillow",               lambda: __import__('PIL').__version__)
check("numpy",                lambda: __import__('numpy').__version__)
check("torch",                lambda: __import__('torch').__version__)
print()

print("BrowserGym:")
check("browsergym",           lambda: (__import__('browsergym', fromlist=['*']), "imported")[-1])
check("browsergym.core",      lambda: (__import__('browsergym.core', fromlist=['*']), "imported")[-1])
check("browsergym.miniwob",   lambda: (__import__('browsergym.miniwob', fromlist=['*']), "imported")[-1])
check("browsergym.webarena",  lambda: (__import__('browsergym.webarena', fromlist=['*']), "imported")[-1])
check("browsergym.visualwebarena", lambda: (__import__('browsergym.visualwebarena', fromlist=['*']), "imported")[-1])
check("browsergym.assistantbench", lambda: (__import__('browsergym.assistantbench', fromlist=['*']), "imported")[-1], required=False)
check("browsergym.experiments", lambda: (__import__('browsergym.experiments', fromlist=['*']), "imported")[-1], required=False)
check("libwebarena",          lambda: (__import__('libwebarena', fromlist=['*']), "imported")[-1], required=False)
check("libvisualwebarena",    lambda: (__import__('libvisualwebarena', fromlist=['*']), "imported")[-1], required=False)
print()


def _chromium_launch() -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        b.close()
    return "chromium launches headless"
check("playwright chromium",  _chromium_launch)
print()

print("=" * 50)
if failures:
    print(f"{len(failures)} REQUIRED check(s) FAILED:")
    for label, err in failures:
        print(f"  - {label}: {err}")
    sys.exit(1)
print("All required checks passed.")
print("=" * 50)
