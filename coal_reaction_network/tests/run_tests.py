"""Run the regression suite using only the Python standard library."""

import importlib
import inspect
import sys
import unittest
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root.parent / "src"))
    suite = unittest.TestSuite()
    for path in sorted(root.glob("test_*.py")):
        module = importlib.import_module(path.stem)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and function.__module__ == module.__name__:
                suite.addTest(unittest.FunctionTestCase(function))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
