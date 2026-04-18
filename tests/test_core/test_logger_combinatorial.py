"""
Combinatorial testing for the Logger logic.
Applies Each Choice Coverage (ECC) to systematically test multiple formatting 
configuration parameters without combinatorial explosion.
"""
import pytest
from engine.core.logger import Logger, LogLevel

# Partitions identified:
# show_timestamps: [True, False]
# show_logger_name: [True, False]
# show_level: [True, False]
# colors_enabled: [True, False]
# log_level: [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]

@pytest.mark.parametrize(
    "show_timestamps, show_logger_name, show_level, colors_enabled, log_level",
    [
        # Combination 1: Covers all Trues and DEBUG level
        (True, True, True, True, LogLevel.DEBUG),
        
        # Combination 2: Covers all Falses and INFO level
        (False, False, False, False, LogLevel.INFO),
        
        # Combination 3: Mixes True/False to cover WARNING level
        (True, False, True, False, LogLevel.WARNING),
        
        # Combination 4: Mixes False/True to cover ERROR level
        (False, True, False, True, LogLevel.ERROR),
    ]
)
def test_logger_format_ecc(show_timestamps, show_logger_name, show_level, colors_enabled, log_level):
    """
    Combinatorial testing: Each Choice Coverage (ECC) for Logger._format_message.
    Ensures every parameter partition is executed at least once across 4 minimal test cases.
    """
    logger = Logger("EccTestLogger")
    
    # Apply combinatorial state
    logger.show_timestamps = show_timestamps
    logger.show_logger_name = show_logger_name
    logger.show_level = show_level
    logger.enable_colors(colors_enabled)
    
    test_message = "Combinatorial Message"
    
    # Execute the method under test
    formatted_msg = logger._format_message(log_level, test_message)
    
    # 1. Validate the core message is always there
    assert test_message in formatted_msg
    
    # 2. Validate timestamps
    # Formatted timestamps usually append a colon via time format, or bracket "[HH:MM:SS]"
    # Just checking for brackets since timestamps might fluctuate 
    if show_timestamps:
        assert "[" in formatted_msg 
        
    # 3. Validate name
    if show_logger_name:
        assert "EccTestLogger" in formatted_msg
    else:
        assert "EccTestLogger" not in formatted_msg
        
    # 4. Validate log level text
    if show_level:
        assert log_level.name in formatted_msg
    else:
        assert log_level.name not in formatted_msg
        
    # 5. Validate ANSI Color formatting
    ansi_escape_start = '\033['
    if colors_enabled:
        assert ansi_escape_start in formatted_msg
        assert logger.reset_color in formatted_msg
    else:
        assert ansi_escape_start not in formatted_msg