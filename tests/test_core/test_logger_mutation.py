"""
Mutation-killing tests for Logger.
Targets surviving mutants from mutmut analysis.
"""
import io
import sys
import pytest
from engine.core.logger import Logger, LoggerManager, LogLevel, get_logger, set_global_log_level, enable_colors, configure_timestamps


class TestLoggerMutationKillers:

    def test_error_level_value(self):
        """Kill mutant 7: ERROR = 3 mutated to ERROR = 4"""
        assert LogLevel.ERROR.value == 3

    def test_output_stream_default_stdout(self):
        """Kill mutant 12: output_stream default sys.stdout mutated to None"""
        logger = Logger("test")
        assert logger.output_stream is sys.stdout

    def test_error_stream_default_stderr(self):
        """Kill mutant 13: error_stream default sys.stderr mutated to None"""
        logger = Logger("test")
        assert logger.error_stream is sys.stderr

    def test_debug_color_code(self):
        """Kill mutant 20: DEBUG color code mutation"""
        logger = Logger("test")
        assert logger.colors[LogLevel.DEBUG] == '\033[90m'

    def test_info_color_code(self):
        """Kill mutant 21: INFO color code mutation"""
        logger = Logger("test")
        assert logger.colors[LogLevel.INFO] == '\033[94m'

    def test_warning_color_code(self):
        """Kill mutant 22: WARNING color code mutation"""
        logger = Logger("test")
        assert logger.colors[LogLevel.WARNING] == '\033[93m'

    def test_error_color_code(self):
        """Kill mutant 23: ERROR color code mutation"""
        logger = Logger("test")
        assert logger.colors[LogLevel.ERROR] == '\033[91m'

    def test_enable_colors_default_true(self):
        """Kill mutant 30: enable_colors default param True → False"""
        logger = Logger("test")
        # Call with no args - should keep colors (default=True means colors stay)
        logger.enable_colors()
        assert logger.colors[LogLevel.DEBUG] == '\033[90m'

    def test_enable_colors_false_clears(self):
        """Verify enable_colors(False) clears colors"""
        logger = Logger("test")
        logger.enable_colors(False)
        assert logger.colors[LogLevel.DEBUG] == ''
        assert logger.reset_color == ''

    def test_format_level_name_in_brackets(self):
        """Kill mutant 42: f'[{level_name}]' format mutation"""
        logger = Logger("test")
        logger.show_timestamps = False
        logger.show_logger_name = False
        logger.enable_colors(False)
        formatted = logger._format_message(LogLevel.INFO, "hello")
        assert "[INFO]" in formatted
        # Should be exactly [INFO] not XX[INFO]XX
        assert "XX" not in formatted

    def test_write_message_newline(self):
        """Kill mutant 57: '\\n' mutated to 'XX\\nXX'"""
        logger = Logger("test")
        logger.show_timestamps = False
        logger.show_logger_name = False
        logger.show_level = False
        logger.enable_colors(False)
        buf = io.StringIO()
        logger.set_output_stream(buf)
        logger.info("hello")
        output = buf.getvalue()
        assert output.endswith("hello\n")
        assert "XX" not in output

    def test_format_message_join_and_message(self):
        """Kill mutants 25/26/80: format string parts joined with space + message"""
        logger = Logger("test")
        logger.show_timestamps = False
        logger.show_logger_name = True
        logger.show_level = True
        logger.enable_colors(False)
        formatted = logger._format_message(LogLevel.INFO, "testmsg")
        # Should be "[test] [INFO] testmsg"
        assert formatted == "[test] [INFO] testmsg"

    def test_format_name_in_brackets(self):
        """Kill mutant 40: f'[{self.name}]' format string mutation"""
        logger = Logger("MyLogger")
        logger.show_timestamps = False
        logger.show_level = False
        logger.enable_colors(False)
        formatted = logger._format_message(LogLevel.INFO, "msg")
        assert formatted == "[MyLogger] msg"

    def test_color_wraps_message(self):
        """Kill mutants 83/84: color prefix and reset_color suffix"""
        logger = Logger("test")
        logger.show_timestamps = False
        logger.show_logger_name = False
        logger.show_level = False
        formatted = logger._format_message(LogLevel.INFO, "msg")
        assert formatted.startswith('\033[94m')
        assert formatted.endswith('\033[0m')

    def test_set_global_level_stores_level(self):
        """Kill mutant 66: default_level = level mutated to default_level = None"""
        mgr = LoggerManager()
        mgr.set_global_level(LogLevel.DEBUG)
        assert mgr.default_level == LogLevel.DEBUG

    def test_enable_colors_global_default_true(self):
        """Kill mutant 72: enable_colors default param True → False at module level"""
        # Module-level enable_colors should have default True
        import inspect
        sig = inspect.signature(enable_colors)
        assert sig.parameters['enabled'].default is True

    def test_configure_timestamps_global_default_true(self):
        """Kill mutant 73: configure_timestamps default param True → False"""
        import inspect
        sig = inspect.signature(configure_timestamps)
        assert sig.parameters['enabled'].default is True

    def test_timestamp_format_in_brackets(self):
        """Kill mutants 37/38: timestamp format string mutation"""
        logger = Logger("test")
        logger.show_timestamps = True
        logger.show_logger_name = False
        logger.show_level = False
        logger.enable_colors(False)
        formatted = logger._format_message(LogLevel.INFO, "msg")
        # Should contain [HH:MM:SS] format
        import re
        assert re.search(r'\[\d{2}:\d{2}:\d{2}\]', formatted)

    def test_reset_color_value(self):
        """Kill mutant 25: reset_color mutation"""
        logger = Logger("test")
        assert logger.reset_color == '\033[0m'

    def test_info_writes_to_output_not_error(self):
        """Kill mutants: stream routing for info"""
        logger = Logger("test")
        logger.show_timestamps = False
        logger.show_logger_name = False
        logger.show_level = False
        logger.enable_colors(False)
        out = io.StringIO()
        err = io.StringIO()
        logger.set_output_stream(out)
        logger.set_error_stream(err)
        logger.info("hello")
        assert out.getvalue().strip() == "hello"
        assert err.getvalue() == ""

    def test_error_writes_to_error_stream(self):
        """Kill mutants: stream routing for error"""
        logger = Logger("test")
        logger.show_timestamps = False
        logger.show_logger_name = False
        logger.show_level = False
        logger.enable_colors(False)
        out = io.StringIO()
        err = io.StringIO()
        logger.set_output_stream(out)
        logger.set_error_stream(err)
        logger.error("bad")
        assert err.getvalue().strip() == "bad"
        assert out.getvalue() == ""

    def test_new_logger_inherits_default_level(self):
        """Kill mutant 66 via side effect: after set_global_level, new loggers use it"""
        mgr = LoggerManager()
        mgr.set_global_level(LogLevel.DEBUG)
        new_logger = mgr.get_logger("new_system")
        assert new_logger.min_level == LogLevel.DEBUG
