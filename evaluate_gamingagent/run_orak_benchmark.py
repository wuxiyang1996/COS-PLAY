#!/usr/bin/env python
"""
Backward-compatible shim -- delegates to evaluate_orak.run_orak_benchmark.

All Orak benchmark code now lives in evaluate_orak/run_orak_benchmark.py.
"""
from evaluate_orak.run_orak_benchmark import main

if __name__ == "__main__":
    main()
