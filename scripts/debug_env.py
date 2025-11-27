import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("sys.path:")
for p in sys.path:
    print(f"  {p}")

print("-" * 20)
try:
    import numpy
    print(f"✅ NumPy version: {numpy.__version__}")
    print(f"   Path: {numpy.__file__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import pandas
    print(f"✅ pandas version: {pandas.__version__}")
    print(f"   Path: {pandas.__file__}")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")
