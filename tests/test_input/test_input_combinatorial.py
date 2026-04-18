"""
Combinatorial testing for the InputManager module.
Applies Each Choice Coverage (ECC) to systematically test gamepad stick simulation.
"""
import pytest
from engine.input.input_manager import InputManager
from engine.math.vector2 import Vector2

# Partitions identified (Building blocks for ECC):
# stick: ["left" (Normal), "right" (Normal), "invalid" (Error Guessing)]
# x: [-1.5 (Below Clamp), 0.5 (Nominal), 1.5 (Above Clamp)]
# y: [-2.0 (Below Clamp), -0.5 (Nominal), 2.0 (Above Clamp)]
# gamepad_id: [0 (Player 1), 1 (Player 2), 99 (Disconnected/Invalid)]

@pytest.mark.parametrize(
    "stick, x, y, gamepad_id, expected_x, expected_y, expected_success",
    [
        # Combination 1: Covers Left stick, Negative out-of-bounds, Player 1
        ("left", -1.5, -2.0, 0, -1.0, -1.0, True),
        
        # Combination 2: Covers Right stick, Nominal values, Player 2
        ("right", 0.5, -0.5, 1, 0.5, -0.5, True),
        
        # Combination 3: Covers Invalid stick, Positive out-of-bounds, Disconnected profile
        ("invalid", 1.5, 2.0, 99, 0.0, 0.0, False),
    ]
)
def test_simulate_gamepad_stick_ecc(stick, x, y, gamepad_id, expected_x, expected_y, expected_success):
    """
    Combinatorial testing: Each Choice Coverage (ECC) for simulate_gamepad_stick_input.
    Ensures every parameter partition is executed at least once across 3 minimal test cases.
    """
    im = InputManager()
    
    # Setup state: Connect valid gamepads
    if gamepad_id in [0, 1]:
        im.simulate_gamepad_connection(gamepad_id)
        
    # Execute method under test
    im.simulate_gamepad_stick_input(stick, x, y, gamepad_id)
    
    # Validation logic
    if expected_success:
        # Pull the actual stick data based on whichever stick we targeted
        actual_stick_vector = im.get_gamepad_stick(stick, gamepad_id)
        
        # Validate clamping and assignment occurred correctly
        assert actual_stick_vector.x == pytest.approx(expected_x)
        assert actual_stick_vector.y == pytest.approx(expected_y)
    else:
        # If expected to fail/ignore, ensure default zero vectors were maintained
        # For an invalid gamepad ID, the method should safely return a zero vector
        assert im.get_gamepad_stick("left", gamepad_id) == Vector2.zero()
        assert im.get_gamepad_stick("right", gamepad_id) == Vector2.zero()