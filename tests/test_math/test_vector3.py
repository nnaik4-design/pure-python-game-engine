"""
Blackbox tests for Vector3 using Equivalence Partitioning (EP),
Boundary Analysis (BA), and Error Guessing (EG).
"""
import math
import pytest
from engine.math.vector3 import Vector3
from engine.math.vector2 import Vector2


# ============================================================
# Constructor Tests
# ============================================================

class TestVector3Constructor:

    def test_default_constructor(self):
        """EP: Default partition - no arguments"""
        v = Vector3()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_positive_values(self):
        """EP: Positive value partition"""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0

    def test_negative_values(self):
        """EP: Negative value partition"""
        v = Vector3(-1.0, -2.0, -3.0)
        assert v.x == -1.0 and v.y == -2.0 and v.z == -3.0

    def test_mixed_sign_values(self):
        """EP: Mixed sign partition"""
        v = Vector3(1.0, -2.0, 3.0)
        assert v.x == 1.0 and v.y == -2.0 and v.z == 3.0

    def test_integer_inputs_converted_to_float(self):
        """EP: Integer input partition"""
        v = Vector3(1, 2, 3)
        assert isinstance(v.x, float) and isinstance(v.y, float) and isinstance(v.z, float)

    def test_very_large_values(self):
        """BA: Extremely large values"""
        v = Vector3(1e308, 1e308, 1e308)
        assert v.x == 1e308


# ============================================================
# Arithmetic Tests
# ============================================================

class TestVector3Arithmetic:

    def test_add_positive(self):
        """EP: Addition of positive vectors"""
        result = Vector3(1, 2, 3) + Vector3(4, 5, 6)
        assert result == Vector3(5, 7, 9)

    def test_add_zero_vector(self):
        """BA: Adding zero vector identity"""
        v = Vector3(1, 2, 3)
        assert v + Vector3.zero() == v

    def test_subtract_self(self):
        """BA: Subtracting self gives zero"""
        v = Vector3(7, 8, 9)
        assert v - v == Vector3.zero()

    def test_multiply_positive_scalar(self):
        """EP: Scalar multiplication"""
        assert Vector3(1, 2, 3) * 3 == Vector3(3, 6, 9)

    def test_multiply_by_zero(self):
        """BA: Multiply by zero"""
        assert Vector3(5, 10, 15) * 0 == Vector3.zero()

    def test_multiply_by_negative(self):
        """EP: Multiply by negative scalar reverses direction"""
        assert Vector3(1, 2, 3) * -1 == Vector3(-1, -2, -3)

    def test_rmul(self):
        """EP: Reverse multiply (scalar * vector)"""
        assert 2 * Vector3(3, 4, 5) == Vector3(6, 8, 10)

    def test_divide_by_scalar(self):
        """EP: Division by positive scalar"""
        assert Vector3(10, 20, 30) / 5 == Vector3(2, 4, 6)

    def test_divide_by_zero_raises(self):
        """BA: Division by zero raises ValueError"""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            Vector3(1, 2, 3) / 0

    def test_divide_by_one(self):
        """BA: Division by 1 identity"""
        v = Vector3(7, 8, 9)
        assert v / 1 == v


# ============================================================
# Equality Tests
# ============================================================

class TestVector3Equality:

    def test_equal_vectors(self):
        """EP: Identical vectors are equal"""
        assert Vector3(1, 2, 3) == Vector3(1, 2, 3)

    def test_unequal_vectors(self):
        """EP: Different vectors are not equal"""
        assert not (Vector3(1, 2, 3) == Vector3(4, 5, 6))

    def test_equality_within_tolerance(self):
        """BA: Within 1e-6 tolerance"""
        assert Vector3(1.0, 2.0, 3.0) == Vector3(1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7)

    def test_inequality_outside_tolerance(self):
        """BA: Outside 1e-6 tolerance"""
        assert not (Vector3(1.0, 2.0, 3.0) == Vector3(1.0 + 1e-5, 2.0, 3.0))


# ============================================================
# Magnitude and Normalize Tests
# ============================================================

class TestVector3Magnitude:

    def test_magnitude_unit_x(self):
        """EP: Unit vector along x"""
        assert abs(Vector3(1, 0, 0).magnitude - 1.0) < 1e-6

    def test_magnitude_3d(self):
        """EP: Known 3D magnitude (1,2,2) -> 3"""
        assert abs(Vector3(1, 2, 2).magnitude - 3.0) < 1e-6

    def test_magnitude_zero(self):
        """BA: Zero vector magnitude"""
        assert Vector3.zero().magnitude == 0.0

    def test_magnitude_negative_components(self):
        """EP: Magnitude with negative components"""
        assert abs(Vector3(-1, -2, -2).magnitude - 3.0) < 1e-6

    def test_magnitude_squared(self):
        """EP: Squared magnitude"""
        assert abs(Vector3(1, 2, 2).magnitude_squared - 9.0) < 1e-6

    def test_normalize_positive(self):
        """EP: Normalize positive vector"""
        result = Vector3(0, 0, 5).normalize()
        assert abs(result.magnitude - 1.0) < 1e-6
        assert result == Vector3(0, 0, 1)

    def test_normalize_zero(self):
        """BA: Normalize zero vector returns zero"""
        assert Vector3.zero().normalize() == Vector3.zero()

    def test_normalized_alias(self):
        """EP: normalized() matches normalize()"""
        v = Vector3(3, 4, 5)
        assert v.normalized() == v.normalize()


# ============================================================
# Dot and Cross Product Tests
# ============================================================

class TestVector3DotCross:

    def test_dot_perpendicular(self):
        """EP: Perpendicular vectors dot = 0"""
        assert Vector3(1, 0, 0).dot(Vector3(0, 1, 0)) == 0.0

    def test_dot_parallel(self):
        """EP: Parallel same-direction vectors"""
        assert Vector3(1, 0, 0).dot(Vector3(5, 0, 0)) == 5.0

    def test_dot_anti_parallel(self):
        """EP: Anti-parallel vectors"""
        assert Vector3(1, 0, 0).dot(Vector3(-3, 0, 0)) == -3.0

    def test_dot_with_zero(self):
        """BA: Dot with zero vector"""
        assert Vector3(5, 10, 15).dot(Vector3.zero()) == 0.0

    def test_dot_commutative(self):
        """EG: Dot product is commutative"""
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        assert abs(a.dot(b) - b.dot(a)) < 1e-6

    def test_cross_standard_basis(self):
        """EP: i x j = k"""
        result = Vector3(1, 0, 0).cross(Vector3(0, 1, 0))
        assert result == Vector3(0, 0, 1)

    def test_cross_reverse_basis(self):
        """EP: j x i = -k"""
        result = Vector3(0, 1, 0).cross(Vector3(1, 0, 0))
        assert result == Vector3(0, 0, -1)

    def test_cross_parallel_vectors(self):
        """EP: Parallel vectors cross = zero"""
        result = Vector3(2, 0, 0).cross(Vector3(5, 0, 0))
        assert result == Vector3.zero()

    def test_cross_with_self(self):
        """BA: Self cross product is zero"""
        v = Vector3(1, 2, 3)
        assert v.cross(v) == Vector3.zero()

    def test_cross_anti_commutative(self):
        """EG: a x b = -(b x a)"""
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        assert a.cross(b) == Vector3.zero() - b.cross(a)


# ============================================================
# Distance and Angle Tests
# ============================================================

class TestVector3DistanceAngle:

    def test_distance_same_point(self):
        """BA: Distance to self is 0"""
        v = Vector3(1, 2, 3)
        assert v.distance_to(v) == 0.0

    def test_distance_known(self):
        """EP: Known distance"""
        assert abs(Vector3(0, 0, 0).distance_to(Vector3(1, 2, 2)) - 3.0) < 1e-6

    def test_distance_symmetric(self):
        """EG: Distance is symmetric"""
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        assert abs(a.distance_to(b) - b.distance_to(a)) < 1e-6

    def test_distance_squared(self):
        """EP: Squared distance"""
        assert abs(Vector3(0, 0, 0).distance_squared_to(Vector3(1, 2, 2)) - 9.0) < 1e-6

    def test_angle_perpendicular(self):
        """EP: 90 degrees between perpendicular"""
        angle = Vector3(1, 0, 0).angle_to(Vector3(0, 1, 0))
        assert abs(angle - math.pi / 2) < 1e-6

    def test_angle_same_direction(self):
        """BA: 0 degrees for same direction"""
        angle = Vector3(1, 0, 0).angle_to(Vector3(5, 0, 0))
        assert abs(angle) < 1e-6

    def test_angle_opposite(self):
        """BA: Pi for opposite"""
        angle = Vector3(1, 0, 0).angle_to(Vector3(-1, 0, 0))
        assert abs(angle - math.pi) < 1e-6

    def test_angle_with_zero(self):
        """BA: Angle with zero vector returns 0"""
        assert Vector3(1, 0, 0).angle_to(Vector3.zero()) == 0


# ============================================================
# Lerp Tests
# ============================================================

class TestVector3Lerp:

    def test_lerp_at_zero(self):
        """BA: t=0 returns start"""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 20, 30)
        assert a.lerp(b, 0) == a

    def test_lerp_at_one(self):
        """BA: t=1 returns end"""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 20, 30)
        assert a.lerp(b, 1) == b

    def test_lerp_at_half(self):
        """EP: t=0.5 midpoint"""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 20, 30)
        assert a.lerp(b, 0.5) == Vector3(5, 10, 15)

    def test_lerp_clamps_negative(self):
        """BA: t < 0 clamped to 0"""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 10, 10)
        assert a.lerp(b, -1) == a

    def test_lerp_clamps_above_one(self):
        """BA: t > 1 clamped to 1"""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 10, 10)
        assert a.lerp(b, 2) == b


# ============================================================
# Projection and Reflection Tests
# ============================================================

class TestVector3ProjectReflect:

    def test_project_onto_plane(self):
        """EP: Project (1,1,1) onto XY plane (normal = Z)"""
        v = Vector3(1, 1, 1)
        result = v.project_onto_plane(Vector3(0, 0, 1))
        assert result == Vector3(1, 1, 0)

    def test_project_onto_plane_already_on_plane(self):
        """BA: Vector already on plane stays unchanged"""
        v = Vector3(5, 3, 0)
        result = v.project_onto_plane(Vector3(0, 0, 1))
        assert result == v

    def test_project_onto_plane_parallel_to_normal(self):
        """BA: Vector parallel to normal projects to zero"""
        v = Vector3(0, 0, 5)
        result = v.project_onto_plane(Vector3(0, 0, 1))
        assert result == Vector3.zero()

    def test_reflect_off_horizontal(self):
        """EP: Reflect downward vector off horizontal surface"""
        v = Vector3(1, -1, 0)
        normal = Vector3(0, 1, 0)
        result = v.reflect(normal)
        assert abs(result.x - 1) < 1e-6 and abs(result.y - 1) < 1e-6

    def test_reflect_perpendicular(self):
        """BA: Reflecting a vector along the normal reverses it"""
        v = Vector3(0, -5, 0)
        normal = Vector3(0, 1, 0)
        result = v.reflect(normal)
        assert abs(result.y - 5) < 1e-6


# ============================================================
# Conversion and Factory Tests
# ============================================================

class TestVector3ConversionFactory:

    def test_to_tuple(self):
        """EP: Convert to tuple"""
        assert Vector3(1, 2, 3).to_tuple() == (1.0, 2.0, 3.0)

    def test_to_vector2(self):
        """EP: Convert to Vector2 drops z"""
        result = Vector3(1, 2, 3).to_vector2()
        assert result == Vector2(1, 2)

    def test_copy_independent(self):
        """EP: Copy is independent"""
        v = Vector3(1, 2, 3)
        c = v.copy()
        c.x = 99
        assert v.x == 1.0

    def test_from_vector2(self):
        """EP: Create from Vector2"""
        result = Vector3.from_vector2(Vector2(1, 2))
        assert result == Vector3(1, 2, 0)

    def test_from_vector2_with_z(self):
        """EP: Create from Vector2 with z"""
        result = Vector3.from_vector2(Vector2(1, 2), z=5)
        assert result == Vector3(1, 2, 5)

    def test_static_factories(self):
        """EP: All static factories return correct values"""
        assert Vector3.zero() == Vector3(0, 0, 0)
        assert Vector3.one() == Vector3(1, 1, 1)
        assert Vector3.up() == Vector3(0, 1, 0)
        assert Vector3.down() == Vector3(0, -1, 0)
        assert Vector3.left() == Vector3(-1, 0, 0)
        assert Vector3.right() == Vector3(1, 0, 0)
        assert Vector3.forward() == Vector3(0, 0, 1)
        assert Vector3.back() == Vector3(0, 0, -1)

    def test_str_format(self):
        """EP: String representation"""
        assert str(Vector3(1, 2, 3)) == "Vector3(1.00, 2.00, 3.00)"


# ============================================================
# Error Guessing
# ============================================================

class TestVector3ErrorGuessing:

    def test_chained_operations(self):
        """EG: Chain add, multiply, subtract"""
        result = (Vector3(1, 2, 3) + Vector3(4, 5, 6)) * 2 - Vector3(1, 1, 1)
        assert result == Vector3(9, 13, 17)

    def test_normalize_very_small(self):
        """EG: Normalize very small nonzero vector"""
        v = Vector3(1e-15, 0, 0)
        result = v.normalize()
        assert abs(result.magnitude - 1.0) < 1e-6

    def test_cross_product_right_hand_rule(self):
        """EG: Verify right-hand rule j x k = i"""
        result = Vector3(0, 1, 0).cross(Vector3(0, 0, 1))
        assert result == Vector3(1, 0, 0)

    def test_multiply_divide_roundtrip(self):
        """EG: Multiply then divide returns original"""
        v = Vector3(3, 7, 11)
        assert (v * 4) / 4 == v
