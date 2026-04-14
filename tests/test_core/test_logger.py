"""
Blackbox tests for Logger and LoggerManager classes.
Tests use Equivalence Partitioning (EP), Boundary Analysis (BA), and Error Guessing (EG).
"""
import io
import pytest
from engine.core.logger import Logger, LogLevel, LoggerManager, get_logger, set_global_log_level


# ================================================================
# Logger Constructor Tests
# ================================================================

class TestLoggerConstructor:

    def test_default_constructor(self):
        """EP: Default logger has name 'GameEngine' and INFO level"""
        logger = Logger()
        assert logger.name == "GameEngine"
        assert logger.min_level == LogLevel.INFO

    def test_custom_name_and_level(self):
        """EP: Custom name and level"""
        logger = Logger("Physics", LogLevel.DEBUG)
        assert logger.name == "Physics"
        assert logger.min_level == LogLevel.DEBUG

    def test_default_settings(self):
        """EP: Default settings are enabled"""
        logger = Logger()
        assert logger.show_timestamps is True
        assert logger.show_level is True
        assert logger.show_logger_name is True


# ================================================================
# Log Level Filtering Tests
# ================================================================

class TestLogLevelFiltering:

    def test_should_log_at_same_level(self):
        """EP: Message at min_level is logged"""
        logger = Logger(min_level=LogLevel.INFO)
        assert logger._should_log(LogLevel.INFO) is True

    def test_should_log_above_level(self):
        """EP: Message above min_level is logged"""
        logger = Logger(min_level=LogLevel.INFO)
        assert logger._should_log(LogLevel.WARNING) is True
        assert logger._should_log(LogLevel.ERROR) is True

    def test_should_not_log_below_level(self):
        """EP: Message below min_level is not logged"""
        logger = Logger(min_level=LogLevel.WARNING)
        assert logger._should_log(LogLevel.DEBUG) is False
        assert logger._should_log(LogLevel.INFO) is False

    def test_debug_level_logs_everything(self):
        """BA: DEBUG level logs all messages"""
        logger = Logger(min_level=LogLevel.DEBUG)
        for level in LogLevel:
            assert logger._should_log(level) is True

    def test_error_level_only_errors(self):
        """BA: ERROR level only logs errors"""
        logger = Logger(min_level=LogLevel.ERROR)
        assert logger._should_log(LogLevel.DEBUG) is False
        assert logger._should_log(LogLevel.INFO) is False
        assert logger._should_log(LogLevel.WARNING) is False
        assert logger._should_log(LogLevel.ERROR) is True

    def test_set_level(self):
        """EP: set_level changes the minimum level"""
        logger = Logger(min_level=LogLevel.INFO)
        logger.set_level(LogLevel.DEBUG)
        assert logger.min_level == LogLevel.DEBUG


# ================================================================
# Message Output Tests
# ================================================================

class TestLoggerOutput:

    def test_info_writes_to_output_stream(self):
        """EP: Info messages go to output stream"""
        logger = Logger(min_level=LogLevel.DEBUG)
        output = io.StringIO()
        logger.set_output_stream(output)
        logger.info("test message")
        content = output.getvalue()
        assert "test message" in content

    def test_debug_writes_to_output_stream(self):
        """EP: Debug messages go to output stream"""
        logger = Logger(min_level=LogLevel.DEBUG)
        output = io.StringIO()
        logger.set_output_stream(output)
        logger.debug("debug msg")
        assert "debug msg" in output.getvalue()

    def test_warning_writes_to_error_stream(self):
        """EP: Warning messages go to error stream"""
        logger = Logger(min_level=LogLevel.DEBUG)
        error_output = io.StringIO()
        logger.set_error_stream(error_output)
        logger.warning("warn msg")
        assert "warn msg" in error_output.getvalue()

    def test_error_writes_to_error_stream(self):
        """EP: Error messages go to error stream"""
        logger = Logger(min_level=LogLevel.DEBUG)
        error_output = io.StringIO()
        logger.set_error_stream(error_output)
        logger.error("error msg")
        assert "error msg" in error_output.getvalue()

    def test_filtered_message_not_written(self):
        """EP: Filtered messages produce no output"""
        logger = Logger(min_level=LogLevel.ERROR)
        output = io.StringIO()
        logger.set_output_stream(output)
        logger.debug("should not appear")
        logger.info("should not appear")
        assert output.getvalue() == ""

    def test_log_method(self):
        """EP: Generic log method works"""
        logger = Logger(min_level=LogLevel.DEBUG)
        output = io.StringIO()
        logger.set_output_stream(output)
        logger.log(LogLevel.INFO, "generic log")
        assert "generic log" in output.getvalue()


# ================================================================
# Message Format Tests
# ================================================================

class TestLoggerFormat:

    def test_format_includes_timestamp(self):
        """EP: Formatted message includes timestamp when enabled"""
        logger = Logger()
        logger.show_timestamps = True
        logger.show_level = False
        logger.show_logger_name = False
        msg = logger._format_message(LogLevel.INFO, "hello")
        assert "[" in msg  # Has timestamp brackets

    def test_format_includes_level(self):
        """EP: Formatted message includes level name"""
        logger = Logger()
        logger.show_timestamps = False
        logger.show_logger_name = False
        logger.show_level = True
        msg = logger._format_message(LogLevel.WARNING, "hello")
        assert "WARNING" in msg

    def test_format_includes_logger_name(self):
        """EP: Formatted message includes logger name"""
        logger = Logger("MyLogger")
        logger.show_timestamps = False
        logger.show_level = False
        logger.show_logger_name = True
        msg = logger._format_message(LogLevel.INFO, "hello")
        assert "MyLogger" in msg

    def test_format_all_disabled(self):
        """BA: All formatting options disabled shows just message"""
        logger = Logger()
        logger.show_timestamps = False
        logger.show_level = False
        logger.show_logger_name = False
        logger.enable_colors(False)
        msg = logger._format_message(LogLevel.INFO, "hello")
        assert "hello" in msg

    def test_disable_colors(self):
        """EP: Disabling colors removes ANSI codes"""
        logger = Logger()
        logger.enable_colors(False)
        assert logger.reset_color == ''
        for level in LogLevel:
            assert logger.colors[level] == ''


# ================================================================
# LoggerManager Tests
# ================================================================

class TestLoggerManager:

    def test_default_engine_logger(self):
        """EP: LoggerManager creates default Engine logger"""
        mgr = LoggerManager()
        assert "Engine" in mgr.loggers
        assert mgr.engine_logger.name == "Engine"

    def test_get_logger_creates_new(self):
        """EP: get_logger creates new logger if not exists"""
        mgr = LoggerManager()
        logger = mgr.get_logger("Physics")
        assert logger.name == "Physics"
        assert "Physics" in mgr.loggers

    def test_get_logger_returns_existing(self):
        """EP: get_logger returns same instance for same name"""
        mgr = LoggerManager()
        l1 = mgr.get_logger("Audio")
        l2 = mgr.get_logger("Audio")
        assert l1 is l2

    def test_set_global_level(self):
        """EP: set_global_level updates all loggers"""
        mgr = LoggerManager()
        mgr.get_logger("A")
        mgr.get_logger("B")
        mgr.set_global_level(LogLevel.ERROR)
        for logger in mgr.loggers.values():
            assert logger.min_level == LogLevel.ERROR

    def test_enable_colors_globally(self):
        """EP: enable_colors_globally disables for all"""
        mgr = LoggerManager()
        mgr.get_logger("A")
        mgr.enable_colors_globally(False)
        for logger in mgr.loggers.values():
            assert logger.reset_color == ''

    def test_configure_timestamps(self):
        """EP: configure_timestamps sets all loggers"""
        mgr = LoggerManager()
        mgr.get_logger("A")
        mgr.configure_timestamps(False)
        for logger in mgr.loggers.values():
            assert logger.show_timestamps is False


# ================================================================
# LogLevel Enum Tests
# ================================================================

class TestLogLevel:

    def test_level_ordering(self):
        """EP: Log levels have correct ordering"""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value
        assert LogLevel.INFO.value < LogLevel.WARNING.value
        assert LogLevel.WARNING.value < LogLevel.ERROR.value

    def test_level_names(self):
        """EP: Level names are correct"""
        assert LogLevel.DEBUG.name == "DEBUG"
        assert LogLevel.INFO.name == "INFO"
        assert LogLevel.WARNING.name == "WARNING"
        assert LogLevel.ERROR.name == "ERROR"


# ================================================================
# Module-Level Function Tests
# ================================================================

class TestModuleFunctions:

    def test_get_logger_returns_logger(self):
        """EP: Module get_logger returns a Logger"""
        logger = get_logger("Test")
        assert isinstance(logger, Logger)
        assert logger.name == "Test"

    def test_get_logger_default_name(self):
        """EP: Default name is 'Engine'"""
        logger = get_logger()
        assert logger.name == "Engine"


# ================================================================
# Error Guessing
# ================================================================

class TestLoggerErrorGuessing:

    def test_empty_message(self):
        """EG: Empty message string doesn't crash"""
        logger = Logger(min_level=LogLevel.DEBUG)
        output = io.StringIO()
        logger.set_output_stream(output)
        logger.info("")
        # Should produce output (with formatting) but no crash
        assert output.getvalue() != ""

    def test_very_long_message(self):
        """EG: Very long message doesn't crash"""
        logger = Logger(min_level=LogLevel.DEBUG)
        output = io.StringIO()
        logger.set_output_stream(output)
        long_msg = "A" * 10000
        logger.info(long_msg)
        assert long_msg in output.getvalue()

    def test_special_characters_in_message(self):
        """EG: Special characters in message don't crash"""
        logger = Logger(min_level=LogLevel.DEBUG)
        output = io.StringIO()
        logger.set_output_stream(output)
        logger.info("Test: \n\t\r\0 special chars!")
        assert "special chars" in output.getvalue()
