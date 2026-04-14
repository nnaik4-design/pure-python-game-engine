"""
Whitebox tests for Logger - targeting uncovered branches.
Focus on module-level functions and enable_colors iteration.
"""
import pytest
from engine.core.logger import (
    Logger, LogLevel, set_global_log_level, enable_colors, configure_timestamps
)


class TestModuleLevelFunctions:
    """Whitebox: cover module-level functions (lines 171, 175, 179)"""

    def test_set_global_log_level(self):
        """Branch: set_global_log_level updates manager"""
        set_global_log_level(LogLevel.ERROR)
        # Reset to avoid affecting other tests
        set_global_log_level(LogLevel.INFO)

    def test_enable_colors_module(self):
        """Branch: module-level enable_colors"""
        enable_colors(False)
        enable_colors(True)

    def test_configure_timestamps_module(self):
        """Branch: module-level configure_timestamps"""
        configure_timestamps(False)
        configure_timestamps(True)


class TestLoggerEnableColorsIteration:
    """Whitebox: cover enable_colors loop through all levels (line 56->exit)"""

    def test_enable_colors_iterates_all_levels(self):
        """Branch: disabling colors clears all level color codes"""
        logger = Logger()
        logger.enable_colors(False)
        for level in LogLevel:
            assert logger.colors[level] == ''
        assert logger.reset_color == ''
