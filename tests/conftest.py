"""
Conftest to mock tkinter when it's not available (e.g., CI/headless environments).
This allows testing math modules without a display server.
"""
import sys
from unittest.mock import MagicMock

# Mock tkinter before any engine imports happen
if 'tkinter' not in sys.modules:
    sys.modules['tkinter'] = MagicMock()
    sys.modules['tkinter.font'] = MagicMock()
