"""
Combinatorial testing for the Math module.
Applies Each Choice Coverage (ECC) to systematically test Vector3.lerp boundaries.
"""
import pytest
from engine.math.vector3 import Vector3

# Partitions identified (Building blocks for ECC):
# start_vec: [Zero (0,0,0), Positive (10,10,10), Negative (-5,-5,-5)]
# end_vec: [Zero (0,0,0), Positive (20,20,20), Negative (-10,-10,-10)]
# t: [-1.0 (Clamp Low), 0.0 (Min), 0.5 (Mid), 1.0 (Max), 2.0 (Clamp High)]

@pytest.mark.parametrize(
    "start_vec, end_vec, t, expected_result",
    [
        # Combination 1: Covers Start-Zero, End-Zero, t-ClampLow
        (Vector3(0, 0, 0), Vector3(0, 0, 0), -1.0, Vector3(0, 0, 0)),
        
        # Combination 2: Covers Start-Positive, End-Positive, t-Min
        (Vector3(10, 10, 10), Vector3(20, 20, 20), 0.0, Vector3(10, 10, 10)),
        
        # Combination 3: Covers Start-Negative, End-Negative, t-Mid
        (Vector3(-5, -5, -5), Vector3(-10, -10, -10), 0.5, Vector3(-7.5, -7.5, -7.5)),
        
        # Combination 4: Re-uses Start-Zero, End-Positive. Covers t-Max
        (Vector3(0, 0, 0), Vector3(20, 20, 20), 1.0, Vector3(20, 20, 20)),
        
        # Combination 5: Re-uses Start-Positive, End-Negative. Covers t-ClampHigh
        (Vector3(10, 10, 10), Vector3(-10, -10, -10), 2.0, Vector3(-10, -10, -10)),
    ]
)
def test_vector3_lerp_ecc(start_vec, end_vec, t, expected_result):
    """
    Combinatorial testing: Each Choice Coverage (ECC) for Vector3.lerp calculation.
    Ensures every parameter partition is executed at least once across 5 minimal test cases.
    """
    result = start_vec.lerp(end_vec, t)
    
    # Use approximation to handle potential float precision issues during interpolation
    assert result.x == pytest.approx(expected_result.x)
    assert result.y == pytest.approx(expected_result.y)
    assert result.z == pytest.approx(expected_result.z)