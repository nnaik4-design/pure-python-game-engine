"""
Blackbox tests for Vector2 using Equivalence Partitioning (EP),
Boundary Analysis (BA), and Error Guessing (EG).
"""
import math
import pytest
from engine.math.vector2 import Vector2


# ============================================================
# Constructor Tests
# ============================================================

class TestVector2Constructor:
    """EP: Partition inputs into default, positive, negative, zero, large values"""

    def test_default_constructor(self):
        """EP: Default partition - no arguments provided"""
        v = Vector2()
        assert v.x == 0.0 and v.y == 0.0

    def test_positive_values(self):
        """EP: Positive value partition"""
        v = Vector2(3.5, 7.2)
        assert v.x == 3.5 and v.y == 7.2

    def test_negative_values(self):
        """EP: Negative value partition"""
        v = Vector2(-4.0, -9.1)
        assert v.x == -4.0 and v.y == -9.1

    def test_mixed_sign_values(self):
        """EP: Mixed sign partition (positive x, negative y)"""
        v = Vector2(5.0, -3.0)
        assert v.x == 5.0 and v.y == -3.0

    def test_integer_inputs_converted_to_float(self):
        """EP: Integer input partition - should be stored as float"""
        v = Vector2(3, 7)
        assert isinstance(v.x, float) and isinstance(v.y, float)

    def test_very_large_values(self):
        """BA: Extremely large values near float limits"""
        v = Vector2(1e308, 1e308)
        assert v.x == 1e308 and v.y == 1e308

    def test_very_small_values(self):
        """BA: Very small values near zero"""
        v = Vector2(1e-15, 1e-15)
        assert v.x == 1e-15 and v.y == 1e-15


# ============================================================
# Arithmetic Operation Tests
# ============================================================

class TestVector2Arithmetic:

    def test_add_positive_vectors(self):
        """EP: Addition with two positive vectors"""
        result = Vector2(1, 2) + Vector2(3, 4)
        assert result == Vector2(4, 6)

    def test_add_negative_vectors(self):
        """EP: Addition with two negative vectors"""
        result = Vector2(-1, -2) + Vector2(-3, -4)
        assert result == Vector2(-4, -6)

    def test_add_zero_vector(self):
        """BA: Adding zero vector (identity element) should return same vector"""
        v = Vector2(5, 10)
        result = v + Vector2.zero()
        assert result == v

    def test_subtract_same_vector(self):
        """BA: Subtracting a vector from itself should give zero vector"""
        v = Vector2(7.5, -3.2)
        result = v - v
        assert result == Vector2.zero()

    def test_subtract_positive_from_negative(self):
        """EP: Subtraction across sign boundary"""
        result = Vector2(-1, -2) - Vector2(3, 4)
        assert result == Vector2(-4, -6)

    def test_multiply_by_positive_scalar(self):
        """EP: Scalar multiplication with positive scalar"""
        result = Vector2(2, 3) * 4
        assert result == Vector2(8, 12)

    def test_multiply_by_negative_scalar(self):
        """EP: Scalar multiplication with negative scalar (reverses direction)"""
        result = Vector2(2, 3) * -1
        assert result == Vector2(-2, -3)

    def test_multiply_by_zero(self):
        """BA: Scalar multiplication by zero boundary"""
        result = Vector2(5, 10) * 0
        assert result == Vector2.zero()

    def test_multiply_by_one(self):
        """BA: Scalar multiplication by 1 (identity)"""
        v = Vector2(5, 10)
        result = v * 1
        assert result == v

    def test_rmul_scalar_on_left(self):
        """EP: Reverse multiplication (scalar * vector)"""
        result = 3 * Vector2(2, 4)
        assert result == Vector2(6, 12)

    def test_divide_by_positive_scalar(self):
        """EP: Division by positive scalar"""
        result = Vector2(10, 20) / 5
        assert result == Vector2(2, 4)

    def test_divide_by_negative_scalar(self):
        """EP: Division by negative scalar"""
        result = Vector2(10, 20) / -5
        assert result == Vector2(-2, -4)

    def test_divide_by_one(self):
        """BA: Division by 1 (identity)"""
        v = Vector2(7, 3)
        result = v / 1
        assert result == v

    def test_divide_by_zero_raises_error(self):
        """BA: Division by zero boundary - should raise ValueError"""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            Vector2(1, 2) / 0

    def test_divide_by_very_small_scalar(self):
        """BA: Division by very small scalar near zero boundary"""
        result = Vector2(1, 1) / 1e-10
        assert result == Vector2(1e10, 1e10)


# ============================================================
# Equality Tests
# ============================================================

class TestVector2Equality:

    def test_equal_vectors(self):
        """EP: Two identical vectors should be equal"""
        assert Vector2(1.0, 2.0) == Vector2(1.0, 2.0)

    def test_unequal_vectors(self):
        """EP: Two different vectors should not be equal"""
        assert not (Vector2(1.0, 2.0) == Vector2(3.0, 4.0))

    def test_equality_within_tolerance(self):
        """BA: Vectors within 1e-6 tolerance should be equal"""
        assert Vector2(1.0, 2.0) == Vector2(1.0 + 1e-7, 2.0 - 1e-7)

    def test_inequality_outside_tolerance(self):
        """BA: Vectors outside 1e-6 tolerance should not be equal"""
        assert not (Vector2(1.0, 2.0) == Vector2(1.0 + 1e-5, 2.0))


# ============================================================
# Magnitude Tests
# ============================================================

class TestVector2Magnitude:

    def test_magnitude_unit_x(self):
        """EP: Unit vector along x-axis has magnitude 1"""
        assert abs(Vector2(1, 0).magnitude - 1.0) < 1e-6

    def test_magnitude_3_4_5_triangle(self):
        """EP: Classic 3-4-5 right triangle"""
        assert abs(Vector2(3, 4).magnitude - 5.0) < 1e-6

    def test_magnitude_zero_vector(self):
        """BA: Zero vector has magnitude 0"""
        assert Vector2.zero().magnitude == 0.0

    def test_magnitude_negative_components(self):
        """EP: Magnitude is always non-negative regardless of component signs"""
        assert abs(Vector2(-3, -4).magnitude - 5.0) < 1e-6

    def test_magnitude_squared(self):
        """EP: Squared magnitude avoids sqrt, should be sum of squares"""
        assert abs(Vector2(3, 4).magnitude_squared - 25.0) < 1e-6

    def test_magnitude_squared_zero_vector(self):
        """BA: Squared magnitude of zero vector is 0"""
        assert Vector2.zero().magnitude_squared == 0.0


# ============================================================
# Normalize Tests
# ============================================================

class TestVector2Normalize:

    def test_normalize_unit_vector(self):
        """BA: Normalizing a unit vector should return itself"""
        v = Vector2(1, 0)
        result = v.normalize()
        assert abs(result.magnitude - 1.0) < 1e-6

    def test_normalize_positive_vector(self):
        """EP: Normalizing a positive vector gives magnitude 1"""
        v = Vector2(3, 4)
        result = v.normalize()
        assert abs(result.magnitude - 1.0) < 1e-6

    def test_normalize_negative_vector(self):
        """EP: Normalizing a negative vector gives magnitude 1"""
        v = Vector2(-5, -12)
        result = v.normalize()
        assert abs(result.magnitude - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        """BA: Normalizing zero vector should return zero vector (special case)"""
        result = Vector2.zero().normalize()
        assert result == Vector2.zero()

    def test_normalize_preserves_direction(self):
        """EP: Normalized vector should point in same direction"""
        v = Vector2(10, 0)
        result = v.normalize()
        assert result == Vector2(1, 0)

    def test_normalized_alias(self):
        """EP: normalized() should return same result as normalize()"""
        v = Vector2(3, 4)
        assert v.normalized() == v.normalize()


# ============================================================
# Dot Product Tests
# ============================================================

class TestVector2Dot:

    def test_dot_perpendicular_vectors(self):
        """EP: Perpendicular vectors have dot product of 0"""
        assert Vector2(1, 0).dot(Vector2(0, 1)) == 0.0

    def test_dot_parallel_vectors(self):
        """EP: Parallel same-direction vectors have positive dot product"""
        result = Vector2(1, 0).dot(Vector2(3, 0))
        assert result == 3.0

    def test_dot_anti_parallel_vectors(self):
        """EP: Anti-parallel vectors have negative dot product"""
        result = Vector2(1, 0).dot(Vector2(-3, 0))
        assert result == -3.0

    def test_dot_with_zero_vector(self):
        """BA: Dot product with zero vector is always 0"""
        result = Vector2(5, 10).dot(Vector2.zero())
        assert result == 0.0

    def test_dot_with_self(self):
        """EP: Dot product with self equals magnitude squared"""
        v = Vector2(3, 4)
        assert abs(v.dot(v) - v.magnitude_squared) < 1e-6


# ============================================================
# Cross Product Tests
# ============================================================

class TestVector2Cross:

    def test_cross_perpendicular_vectors(self):
        """EP: Cross product of perpendicular unit vectors"""
        result = Vector2(1, 0).cross(Vector2(0, 1))
        assert result == 1.0

    def test_cross_parallel_vectors(self):
        """EP: Cross product of parallel vectors is 0"""
        result = Vector2(2, 0).cross(Vector2(5, 0))
        assert result == 0.0

    def test_cross_with_self(self):
        """BA: Cross product of vector with itself is 0"""
        v = Vector2(3, 7)
        assert v.cross(v) == 0.0

    def test_cross_anti_commutative(self):
        """EP: a x b = -(b x a) — cross product is anti-commutative"""
        a = Vector2(1, 2)
        b = Vector2(3, 4)
        assert abs(a.cross(b) + b.cross(a)) < 1e-6


# ============================================================
# Distance Tests
# ============================================================

class TestVector2Distance:

    def test_distance_same_point(self):
        """BA: Distance from a point to itself is 0"""
        v = Vector2(3, 4)
        assert v.distance_to(v) == 0.0

    def test_distance_positive(self):
        """EP: Distance between two distinct points"""
        assert abs(Vector2(0, 0).distance_to(Vector2(3, 4)) - 5.0) < 1e-6

    def test_distance_symmetry(self):
        """EP: Distance is symmetric — d(a,b) = d(b,a)"""
        a = Vector2(1, 2)
        b = Vector2(4, 6)
        assert abs(a.distance_to(b) - b.distance_to(a)) < 1e-6

    def test_distance_squared(self):
        """EP: Squared distance should equal distance^2"""
        a = Vector2(0, 0)
        b = Vector2(3, 4)
        assert abs(a.distance_squared_to(b) - 25.0) < 1e-6


# ============================================================
# Angle Tests
# ============================================================

class TestVector2Angle:

    def test_angle_perpendicular(self):
        """EP: Angle between perpendicular vectors is pi/2"""
        angle = Vector2(1, 0).angle_to(Vector2(0, 1))
        assert abs(angle - math.pi / 2) < 1e-6

    def test_angle_same_direction(self):
        """BA: Angle between same-direction vectors is 0"""
        angle = Vector2(1, 0).angle_to(Vector2(5, 0))
        assert abs(angle) < 1e-6

    def test_angle_opposite_direction(self):
        """BA: Angle between opposite vectors is pi"""
        angle = Vector2(1, 0).angle_to(Vector2(-1, 0))
        assert abs(angle - math.pi) < 1e-6

    def test_angle_with_zero_vector(self):
        """BA: Angle involving zero vector returns 0 (edge case)"""
        angle = Vector2(1, 0).angle_to(Vector2.zero())
        assert angle == 0


# ============================================================
# Rotation Tests
# ============================================================

class TestVector2Rotate:

    def test_rotate_90_degrees(self):
        """EP: Rotating (1,0) by 90 degrees gives (0,1)"""
        result = Vector2(1, 0).rotate(math.pi / 2)
        assert abs(result.x - 0) < 1e-6 and abs(result.y - 1) < 1e-6

    def test_rotate_180_degrees(self):
        """EP: Rotating (1,0) by 180 degrees gives (-1,0)"""
        result = Vector2(1, 0).rotate(math.pi)
        assert abs(result.x - (-1)) < 1e-6 and abs(result.y - 0) < 1e-6

    def test_rotate_360_degrees(self):
        """BA: Full rotation returns to original vector"""
        v = Vector2(3, 4)
        result = v.rotate(2 * math.pi)
        assert abs(result.x - v.x) < 1e-6 and abs(result.y - v.y) < 1e-6

    def test_rotate_zero_degrees(self):
        """BA: Zero rotation returns same vector"""
        v = Vector2(3, 4)
        result = v.rotate(0)
        assert result == v

    def test_rotate_negative_angle(self):
        """EP: Negative rotation (clockwise)"""
        result = Vector2(0, 1).rotate(-math.pi / 2)
        assert abs(result.x - 1) < 1e-6 and abs(result.y - 0) < 1e-6

    def test_rotate_zero_vector(self):
        """BA: Rotating zero vector gives zero vector"""
        result = Vector2.zero().rotate(math.pi / 4)
        assert result == Vector2.zero()


# ============================================================
# Lerp Tests
# ============================================================

class TestVector2Lerp:

    def test_lerp_at_zero(self):
        """BA: t=0 returns the start vector"""
        a = Vector2(0, 0)
        b = Vector2(10, 10)
        assert a.lerp(b, 0) == a

    def test_lerp_at_one(self):
        """BA: t=1 returns the end vector"""
        a = Vector2(0, 0)
        b = Vector2(10, 10)
        assert a.lerp(b, 1) == b

    def test_lerp_at_half(self):
        """EP: t=0.5 returns the midpoint"""
        a = Vector2(0, 0)
        b = Vector2(10, 20)
        result = a.lerp(b, 0.5)
        assert result == Vector2(5, 10)

    def test_lerp_clamps_negative_t(self):
        """BA: t < 0 should be clamped to 0 (returns start)"""
        a = Vector2(0, 0)
        b = Vector2(10, 10)
        result = a.lerp(b, -0.5)
        assert result == a

    def test_lerp_clamps_t_above_one(self):
        """BA: t > 1 should be clamped to 1 (returns end)"""
        a = Vector2(0, 0)
        b = Vector2(10, 10)
        result = a.lerp(b, 1.5)
        assert result == b

    def test_lerp_same_vectors(self):
        """EG: Lerping between identical vectors returns that vector"""
        v = Vector2(5, 5)
        result = v.lerp(v, 0.5)
        assert result == v


# ============================================================
# Conversion Tests
# ============================================================

class TestVector2Conversion:

    def test_to_tuple(self):
        """EP: Conversion to tuple"""
        assert Vector2(3.5, 7.2).to_tuple() == (3.5, 7.2)

    def test_to_int_tuple(self):
        """EP: Conversion to int tuple truncates"""
        assert Vector2(3.9, 7.1).to_int_tuple() == (3, 7)

    def test_to_int_tuple_negative(self):
        """EP: Negative float to int truncates toward zero"""
        assert Vector2(-3.9, -7.1).to_int_tuple() == (-3, -7)

    def test_copy_creates_independent_vector(self):
        """EP: Copy creates a new independent vector"""
        v = Vector2(1, 2)
        c = v.copy()
        c.x = 99
        assert v.x == 1  # Original unchanged

    def test_to_vector3(self):
        """EP: Conversion to Vector3 with default z=0"""
        v = Vector2(3, 4)
        v3 = v.to_vector3()
        assert v3.x == 3 and v3.y == 4 and v3.z == 0

    def test_to_vector3_with_z(self):
        """EP: Conversion to Vector3 with custom z"""
        v = Vector2(3, 4)
        v3 = v.to_vector3(z=5.0)
        assert v3.x == 3 and v3.y == 4 and v3.z == 5


# ============================================================
# Static Factory Tests
# ============================================================

class TestVector2StaticFactories:

    def test_zero(self):
        """EP: Zero vector factory"""
        assert Vector2.zero() == Vector2(0, 0)

    def test_one(self):
        """EP: One vector factory"""
        assert Vector2.one() == Vector2(1, 1)

    def test_up(self):
        """EP: Up vector is (0, -1) in screen coordinates"""
        assert Vector2.up() == Vector2(0, -1)

    def test_down(self):
        """EP: Down vector is (0, 1) in screen coordinates"""
        assert Vector2.down() == Vector2(0, 1)

    def test_left(self):
        """EP: Left vector"""
        assert Vector2.left() == Vector2(-1, 0)

    def test_right(self):
        """EP: Right vector"""
        assert Vector2.right() == Vector2(1, 0)

    def test_from_angle_zero(self):
        """BA: Angle 0 produces (1, 0) with default magnitude"""
        result = Vector2.from_angle(0)
        assert abs(result.x - 1) < 1e-6 and abs(result.y - 0) < 1e-6

    def test_from_angle_90(self):
        """EP: Angle pi/2 produces (0, 1)"""
        result = Vector2.from_angle(math.pi / 2)
        assert abs(result.x - 0) < 1e-6 and abs(result.y - 1) < 1e-6

    def test_from_angle_with_magnitude(self):
        """EP: Custom magnitude scales the resulting vector"""
        result = Vector2.from_angle(0, magnitude=5.0)
        assert abs(result.x - 5) < 1e-6 and abs(result.y - 0) < 1e-6


# ============================================================
# String Representation Tests
# ============================================================

class TestVector2String:

    def test_str(self):
        """EP: String representation format"""
        assert str(Vector2(1, 2)) == "Vector2(1.00, 2.00)"

    def test_repr(self):
        """EP: Repr matches str"""
        v = Vector2(1, 2)
        assert repr(v) == str(v)


# ============================================================
# Error Guessing Tests
# ============================================================

class TestVector2ErrorGuessing:

    def test_chained_operations(self):
        """EG: Chaining multiple operations should work correctly"""
        result = (Vector2(1, 2) + Vector2(3, 4)) * 2 - Vector2(1, 1)
        assert result == Vector2(7, 11)

    def test_normalize_very_small_vector(self):
        """EG: Normalizing an extremely small (but nonzero) vector"""
        v = Vector2(1e-15, 0)
        result = v.normalize()
        assert abs(result.magnitude - 1.0) < 1e-6

    def test_large_vector_operations(self):
        """EG: Operations with very large component values"""
        v = Vector2(1e150, 1e150)
        result = v + v
        assert result == Vector2(2e150, 2e150)

    def test_dot_product_commutative(self):
        """EG: Dot product should be commutative"""
        a = Vector2(3, 7)
        b = Vector2(11, 13)
        assert abs(a.dot(b) - b.dot(a)) < 1e-6

    def test_multiply_then_divide_returns_original(self):
        """EG: Multiplying then dividing by same scalar returns original"""
        v = Vector2(3, 7)
        result = (v * 5) / 5
        assert result == v

    def test_distance_to_uses_subtraction(self):
        """EG: distance_to should give same result as manual calculation"""
        a = Vector2(1, 2)
        b = Vector2(4, 6)
        expected = math.sqrt((4-1)**2 + (6-2)**2)
        assert abs(a.distance_to(b) - expected) < 1e-6
