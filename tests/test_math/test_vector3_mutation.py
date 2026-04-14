"""
Mutation-killing tests for Vector3.
Targets surviving mutants from mutmut analysis.
"""
import math
import pytest
from engine.math.vector3 import Vector3


class TestVector3MutationKillers:

    def test_divide_by_zero_exact_message(self):
        """Kill mutant 19: error message string mutation"""
        v = Vector3(1, 1, 1)
        with pytest.raises(ValueError, match=r"^Cannot divide by zero$"):
            v / 0

    def test_equality_exact_boundary_x(self):
        """Kill mutants 24-25: tolerance boundary on x"""
        v1 = Vector3(0, 0, 0)
        assert v1 != Vector3(1e-6, 0, 0)
        assert v1 == Vector3(0.999e-6, 0, 0)

    def test_equality_exact_boundary_y(self):
        """Kill mutants 27-28: tolerance boundary on y"""
        v1 = Vector3(0, 0, 0)
        assert v1 != Vector3(0, 1e-6, 0)
        assert v1 == Vector3(0, 0.999e-6, 0)

    def test_equality_exact_boundary_z(self):
        """Kill mutants 30-31: tolerance boundary on z"""
        v1 = Vector3(0, 0, 0)
        assert v1 != Vector3(0, 0, 1e-6)
        assert v1 == Vector3(0, 0, 0.999e-6)

    def test_magnitude_squared_first_term(self):
        """Kill mutant 40: x*x mutated to x/x"""
        v = Vector3(3, 0, 0)
        # Correct: 3*3 = 9, mutated: 3/3 = 1
        assert v.magnitude_squared == pytest.approx(9.0)

    def test_normalize_y_component(self):
        """Kill mutant 52: y/mag mutated to y*mag"""
        v = Vector3(0, 3, 0)
        n = v.normalize()
        # Correct: y/3 = 1.0, mutated: y*3 = 9.0
        assert n.y == pytest.approx(1.0)

    def test_distance_squared_subtraction(self):
        """Kill mutant 69: (self-other) mutated to (self+other)"""
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(3, 0, 0)
        assert v1.distance_squared_to(v2) == pytest.approx(4.0)

    def test_angle_to_division(self):
        """Kill mutant 79: dot/mag mutated to dot*mag"""
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(1, 1, 0)
        angle = v1.angle_to(v2)
        assert angle == pytest.approx(math.pi / 4, abs=0.001)

    def test_lerp_subtraction(self):
        """Kill mutant 84: (other-self) mutated to (other+self)"""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 0, 0)
        result = v1.lerp(v2, 0.5)
        # Correct: 0 + (10-0)*0.5 = 5
        assert result.x == pytest.approx(5.0)
        # If mutated to +: 0 + (10+0)*0.5 = 5 (same here, try different values)
        v3 = Vector3(2, 0, 0)
        v4 = Vector3(8, 0, 0)
        result2 = v3.lerp(v4, 0.5)
        # Correct: 2 + (8-2)*0.5 = 5
        # Mutated: 2 + (8+2)*0.5 = 7
        assert result2.x == pytest.approx(5.0)
