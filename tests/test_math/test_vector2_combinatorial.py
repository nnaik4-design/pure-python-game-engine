import pytest
import math
import itertools
from engine.math.vector2 import Vector2

# --- Test Values ---
ANGLES = [0.0, math.pi/4, math.pi/2, math.pi, 2*math.pi, -math.pi/2]
MAGNITUDES = [0.0, 0.5, 1.0, 10.0, -1.0]

# --- ACoC Generators ---
def get_from_angle_combinations():
    return list(itertools.product(ANGLES, MAGNITUDES))


@pytest.mark.parametrize("angle, magnitude", get_from_angle_combinations())
def test_from_angle_acoc(angle, magnitude):
    result = Vector2.from_angle(angle, magnitude)
    
    expected_x = math.cos(angle) * magnitude
    expected_y = math.sin(angle) * magnitude
    
    assert abs(result.x - expected_x) < 1e-5
    assert abs(result.y - expected_y) < 1e-5
