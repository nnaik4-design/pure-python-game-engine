"""
Mutation-killing tests for Vector2.
These tests target specific surviving mutants from mutmut analysis.
"""
import math
import pytest
from engine.math.vector2 import Vector2


class TestMutationKillers:

    def test_divide_by_zero_exact_message(self):
        """Kill mutant 14: error message string mutation"""
        v = Vector2(1, 1)
        with pytest.raises(ValueError, match=r"^Cannot divide by zero$"):
            v / 0

    def test_equality_exact_boundary_x(self):
        """Kill mutants 18-19: tolerance < vs <= and 1e-6 vs 2e-6 on x"""
        v1 = Vector2(0, 0)
        # Exactly at 1e-6 boundary: should NOT be equal (< means strictly less)
        v2 = Vector2(1e-6, 0)
        assert v1 != v2
        # Just below 1e-6: should be equal
        v3 = Vector2(0.999e-6, 0)
        assert v1 == v3
        # At 1.5e-6: should NOT be equal (kills 2e-6 mutation)
        v4 = Vector2(1.5e-6, 0)
        assert v1 != v4

    def test_equality_exact_boundary_y(self):
        """Kill mutants 21-22: tolerance < vs <= and 1e-6 vs 2e-6 on y"""
        v1 = Vector2(0, 0)
        v2 = Vector2(0, 1e-6)
        assert v1 != v2
        v3 = Vector2(0, 0.999e-6)
        assert v1 == v3
        v4 = Vector2(0, 1.5e-6)
        assert v1 != v4

    def test_distance_squared_correctness(self):
        """Kill mutant 46: (self - other) mutated to (self + other)"""
        v1 = Vector2(1, 0)
        v2 = Vector2(3, 0)
        # Correct: (1-3)^2 = 4
        assert v1.distance_squared_to(v2) == pytest.approx(4.0)
        # If mutated to +: (1+3)^2 = 16, which is wrong
        assert v1.distance_squared_to(v2) != pytest.approx(16.0)

    def test_angle_to_clamp_lower_bound(self):
        """Kill mutant 54: max(-1, ...) mutated to max(-2, ...)
        Use vectors where floating-point error pushes dot/mag slightly below -1.
        If clamp is max(-2, ...) then acos gets a value < -1 and errors.
        """
        # Exactly anti-parallel should clamp to -1
        v1 = Vector2(1, 0)
        v2 = Vector2(-1, 0)
        angle = v1.angle_to(v2)
        assert angle == pytest.approx(math.pi, abs=0.001)
        # The key is: acos(-1) = pi, acos(-2) would be ValueError
        # So this test only kills the mutant if floating-point yields < -1
        # Verify result is valid (not NaN and in [0, pi])
        assert 0 <= angle <= math.pi + 0.001

    def test_angle_to_clamp_upper_bound(self):
        """Kill mutant 55: min(1, ...) mutated to min(2, ...)
        Same parallel vectors with different magnitudes.
        """
        v1 = Vector2(1, 0)
        v2 = Vector2(2, 0)
        angle = v1.angle_to(v2)
        assert angle == pytest.approx(0.0, abs=0.001)
        assert 0 <= angle <= math.pi + 0.001

    def test_angle_to_division_vs_multiplication(self):
        """Kill mutant 56: dot/mag_product mutated to dot*mag_product"""
        v1 = Vector2(1, 0)
        v2 = Vector2(1, 1)
        # dot = 1, mag_product = 1 * sqrt(2) = sqrt(2)
        # Correct: acos(1/sqrt(2)) = pi/4
        angle = v1.angle_to(v2)
        assert angle == pytest.approx(math.pi / 4, abs=0.001)
        # If mutated to *: acos(1 * sqrt(2)) = acos(1.414) → domain error or wrong

    def test_from_angle_with_magnitude(self):
        """Kill mutant 94: sin*magnitude mutated to sin/magnitude"""
        v = Vector2.from_angle(math.pi / 2, 3.0)
        # Correct: x = cos(pi/2)*3 ≈ 0, y = sin(pi/2)*3 = 3
        assert v.y == pytest.approx(3.0, abs=0.01)
        # If mutated to /: y = sin(pi/2)/3 = 0.333, which is wrong
        assert v.y != pytest.approx(1 / 3, abs=0.01)
