import pytest
import math
from itertools import product
from allpairspy import AllPairs
from engine.math.vector2 import Vector2


# ============================================================================
# 1. Vector2 Basic Operations - PWC
# ============================================================================
_vec_op_parameters = [
    [0.0, 1.0, -1.0, 5.0],  # x1
    [0.0, 1.0, -1.0, 5.0],  # y1
    [0.0, 1.0, -1.0, 5.0],  # x2
    [0.0, 1.0, -1.0, 5.0],  # y2
]
_vec_op_pwc_cases = [tuple(case) for case in AllPairs(_vec_op_parameters)]

@pytest.mark.parametrize("x1, y1, x2, y2", _vec_op_pwc_cases)
def test_vector2_operations_pwc(x1, y1, x2, y2):
    """Test Vector2 arithmetic operations with comprehensive validation"""
    v1 = Vector2(x1, y1)
    v2 = Vector2(x2, y2)

    # Test addition
    sum_vec = v1 + v2
    assert abs(sum_vec.x - (x1 + x2)) < 1e-10
    assert abs(sum_vec.y - (y1 + y2)) < 1e-10

    # Test subtraction
    diff_vec = v1 - v2
    assert abs(diff_vec.x - (x1 - x2)) < 1e-10
    assert abs(diff_vec.y - (y1 - y2)) < 1e-10

    # Test scalar multiplication
    scalar = 2.5
    scaled = v1 * scalar
    assert abs(scaled.x - x1 * scalar) < 1e-10
    assert abs(scaled.y - y1 * scalar) < 1e-10

    # Test reverse scalar multiplication
    r_scaled = scalar * v1
    assert abs(r_scaled.x - scaled.x) < 1e-10
    assert abs(r_scaled.y - scaled.y) < 1e-10

    # Test division (except when dividing by zero)
    if x1 != 0.0 or y1 != 0.0:
        div_scalar = 2.0
        divided = v1 / div_scalar
        assert abs(divided.x - x1 / div_scalar) < 1e-10
        assert abs(divided.y - y1 / div_scalar) < 1e-10


def test_vector2_division_by_zero():
    """Test that division by zero raises ValueError"""
    v = Vector2(1.0, 2.0)
    with pytest.raises(ValueError):
        v / 0


# ============================================================================
# 2. Vector2 Magnitude and Normalization - PWC
# ============================================================================
_magnitude_parameters = [
    [0.0, 1.0, 3.0, -2.0, 10.0],  # x
    [0.0, 1.0, 4.0, -2.0, 10.0],  # y
]
_magnitude_pwc_cases = [tuple(case) for case in AllPairs(_magnitude_parameters)]

@pytest.mark.parametrize("x, y", _magnitude_pwc_cases)
def test_vector2_magnitude_pwc(x, y):
    """Test magnitude calculation"""
    v = Vector2(x, y)
    expected_mag = math.sqrt(x*x + y*y)

    assert abs(v.magnitude - expected_mag) < 1e-10
    assert abs(v.magnitude_squared - (x*x + y*y)) < 1e-10


@pytest.mark.parametrize("x, y", _magnitude_pwc_cases)
def test_vector2_normalize_pwc(x, y):
    """Test normalization with edge cases"""
    v = Vector2(x, y)

    if x == 0.0 and y == 0.0:
        # Zero vector should normalize to zero vector
        normalized = v.normalize()
        assert abs(normalized.x) < 1e-10
        assert abs(normalized.y) < 1e-10
    else:
        # Non-zero vector should normalize to magnitude 1.0
        normalized = v.normalize()
        mag = normalized.magnitude
        assert abs(mag - 1.0) < 1e-6, f"Magnitude {mag} != 1.0 for ({x}, {y})"

        # Verify direction is preserved
        original_angle = math.atan2(y, x)
        normalized_angle = math.atan2(normalized.y, normalized.x)
        assert abs(original_angle - normalized_angle) < 1e-6


# ============================================================================
# 3. Vector2 Dot and Cross Products - PWC
# ============================================================================
_product_parameters = [
    [-1.0, 0.0, 1.0, 2.0],  # x1
    [-1.0, 0.0, 1.0, 2.0],  # y1
    [-1.0, 0.0, 1.0, 2.0],  # x2
    [-1.0, 0.0, 1.0, 2.0],  # y2
]
_product_pwc_cases = [tuple(case) for case in AllPairs(_product_parameters)]

@pytest.mark.parametrize("x1, y1, x2, y2", _product_pwc_cases)
def test_vector2_products_pwc(x1, y1, x2, y2):
    """Test dot and cross products"""
    v1 = Vector2(x1, y1)
    v2 = Vector2(x2, y2)

    # Test dot product
    expected_dot = x1 * x2 + y1 * y2
    assert abs(v1.dot(v2) - expected_dot) < 1e-10

    # Dot product should be commutative
    assert abs(v1.dot(v2) - v2.dot(v1)) < 1e-10

    # Test cross product (2D cross product is scalar)
    expected_cross = x1 * y2 - y1 * x2
    assert abs(v1.cross(v2) - expected_cross) < 1e-10

    # Cross product should be anti-commutative: v1 × v2 = -(v2 × v1)
    assert abs(v1.cross(v2) + v2.cross(v1)) < 1e-10

    # Dot product of perpendicular vectors should be 0
    if not (x1 == 0 and y1 == 0) and not (x2 == 0 and y2 == 0):
        perp_v = Vector2(-y1, x1)  # Perpendicular vector
        assert abs(v1.dot(perp_v)) < 1e-10


# ============================================================================
# 4. Vector2 Rotation - PWC
# ============================================================================
_rotation_parameters = [
    [0.0, math.pi/4, math.pi/2, math.pi, -math.pi/2],  # angle
    [0.0, 1.0, 2.0, -1.0],  # x
    [0.0, 1.0, 2.0, -1.0],  # y
]
_rotation_pwc_cases = [tuple(case) for case in AllPairs(_rotation_parameters)]

@pytest.mark.parametrize("angle, x, y", _rotation_pwc_cases)
def test_vector2_rotate_pwc(angle, x, y):
    """Test vector rotation"""
    v = Vector2(x, y)
    rotated = v.rotate(angle)

    # Magnitude should be preserved
    assert abs(v.magnitude - rotated.magnitude) < 1e-10

    # Verify rotation using complex numbers: (x + yi) * e^(iθ)
    expected_x = x * math.cos(angle) - y * math.sin(angle)
    expected_y = x * math.sin(angle) + y * math.cos(angle)

    assert abs(rotated.x - expected_x) < 1e-10
    assert abs(rotated.y - expected_y) < 1e-10

    # Rotating by 2π should return to original
    full_rotation = v.rotate(2 * math.pi)
    assert abs(full_rotation.x - v.x) < 1e-10
    assert abs(full_rotation.y - v.y) < 1e-10


# ============================================================================
# 5. Vector2 Distance - PWC
# ============================================================================
_distance_parameters = [
    [-2.0, 0.0, 3.0],  # x1
    [-2.0, 0.0, 3.0],  # y1
    [-2.0, 0.0, 3.0],  # x2
    [-2.0, 0.0, 3.0],  # y2
]
_distance_pwc_cases = [tuple(case) for case in AllPairs(_distance_parameters)]

@pytest.mark.parametrize("x1, y1, x2, y2", _distance_pwc_cases)
def test_vector2_distance_pwc(x1, y1, x2, y2):
    """Test distance calculations"""
    v1 = Vector2(x1, y1)
    v2 = Vector2(x2, y2)

    # Calculate expected distance
    dx = x2 - x1
    dy = y2 - y1
    expected_distance = math.sqrt(dx*dx + dy*dy)
    expected_distance_sq = dx*dx + dy*dy

    assert abs(v1.distance_to(v2) - expected_distance) < 1e-10
    assert abs(v1.distance_squared_to(v2) - expected_distance_sq) < 1e-10

    # Distance should be symmetric
    assert abs(v1.distance_to(v2) - v2.distance_to(v1)) < 1e-10

    # Distance from a point to itself should be 0
    assert abs(v1.distance_to(v1)) < 1e-10


# ============================================================================
# 6. Vector2 Angle - PWC
# ============================================================================
@pytest.mark.parametrize("x1, y1, x2, y2", _distance_pwc_cases)
def test_vector2_angle_to_pwc(x1, y1, x2, y2):
    """Test angle between vectors"""
    v1 = Vector2(x1, y1)
    v2 = Vector2(x2, y2)

    if v1.magnitude == 0 or v2.magnitude == 0:
        # Zero vectors: angle should be 0
        angle = v1.angle_to(v2)
        assert abs(angle) < 1e-10
    else:
        angle = v1.angle_to(v2)

        # Angle should be between 0 and π
        assert 0 <= angle <= math.pi + 1e-6

        # Verify using dot product formula: cos(θ) = (a·b) / (|a||b|)
        dot_prod = v1.dot(v2)
        mag_product = v1.magnitude * v2.magnitude
        expected_cos = dot_prod / mag_product
        expected_cos = max(-1, min(1, expected_cos))  # Clamp for numerical stability
        expected_angle = math.acos(expected_cos)

        assert abs(angle - expected_angle) < 1e-6


# ============================================================================
# 7. Vector2 Interpolation (Lerp) - PWC
# ============================================================================
_lerp_parameters = [
    [0.0, 1.0, -1.0],  # x1
    [0.0, 1.0, -1.0],  # y1
    [0.0, 1.0, -1.0],  # x2
    [0.0, 1.0, -1.0],  # y2
    [-0.5, 0.0, 0.5, 1.0, 1.5],  # t
]
_lerp_pwc_cases = [tuple(case) for case in AllPairs(_lerp_parameters)]

@pytest.mark.parametrize("x1, y1, x2, y2, t", _lerp_pwc_cases)
def test_vector2_lerp_pwc(x1, y1, x2, y2, t):
    """Test linear interpolation"""
    v1 = Vector2(x1, y1)
    v2 = Vector2(x2, y2)

    result = v1.lerp(v2, t)

    # t should be clamped to [0, 1]
    t_clamped = max(0, min(1, t))
    expected_x = x1 + (x2 - x1) * t_clamped
    expected_y = y1 + (y2 - y1) * t_clamped

    assert abs(result.x - expected_x) < 1e-10
    assert abs(result.y - expected_y) < 1e-10

    # At t=0, should return v1
    assert abs((v1.lerp(v2, 0.0) - v1).magnitude) < 1e-10

    # At t=1, should return v2
    assert abs((v1.lerp(v2, 1.0) - v2).magnitude) < 1e-10


# ============================================================================
# 8. Vector2 Static Constructors and Properties - ECC
# ============================================================================
class TestVector2Static:
    """Test static constructors and special vectors"""

    def test_from_angle(self):
        """Test creating vector from angle and magnitude"""
        test_cases = [
            (0.0, 1.0, 1.0, 0.0),           # Right
            (math.pi/2, 1.0, 0.0, 1.0),     # Down
            (math.pi, 1.0, -1.0, 0.0),      # Left
            (-math.pi/2, 1.0, 0.0, -1.0),   # Up
            (math.pi/4, 1.0, math.sqrt(2)/2, math.sqrt(2)/2),
            (0.0, 2.5, 2.5, 0.0),           # With magnitude
            (0.0, 0.0, 0.0, 0.0),           # Zero magnitude
        ]

        for angle, magnitude, expected_x, expected_y in test_cases:
            v = Vector2.from_angle(angle, magnitude)
            assert abs(v.x - expected_x) < 1e-6, f"x mismatch for angle {angle}, mag {magnitude}"
            assert abs(v.y - expected_y) < 1e-6, f"y mismatch for angle {angle}, mag {magnitude}"

    def test_static_vectors(self):
        """Test static vector constructors"""
        # Zero vector
        zero = Vector2.zero()
        assert zero.x == 0.0 and zero.y == 0.0
        assert zero.magnitude == 0.0

        # Unit vector
        one = Vector2.one()
        assert one.x == 1.0 and one.y == 1.0

        # Direction vectors
        up = Vector2.up()
        assert up.x == 0.0 and up.y == -1.0  # Negative Y is up

        down = Vector2.down()
        assert down.x == 0.0 and down.y == 1.0

        left = Vector2.left()
        assert left.x == -1.0 and left.y == 0.0

        right = Vector2.right()
        assert right.x == 1.0 and right.y == 0.0

    def test_vector2_copy(self):
        """Test vector copying"""
        v1 = Vector2(3.0, 4.0)
        v2 = v1.copy()

        # Should have same values
        assert v2.x == v1.x
        assert v2.y == v1.y

        # But be different objects
        assert v1 is not v2

        # Modifying copy shouldn't affect original
        v2.x = 10.0
        assert v1.x == 3.0

    def test_vector2_tuple_conversion(self):
        """Test tuple conversions"""
        v = Vector2(3.7, 4.2)

        # Float tuple
        tup = v.to_tuple()
        assert abs(tup[0] - 3.7) < 1e-10
        assert abs(tup[1] - 4.2) < 1e-10

        # Int tuple
        int_tup = v.to_int_tuple()
        assert int_tup[0] == 3
        assert int_tup[1] == 4

    def test_vector2_equality(self):
        """Test vector equality with tolerance"""
        v1 = Vector2(1.0, 2.0)
        v2 = Vector2(1.0, 2.0)
        v3 = Vector2(1.0 + 1e-7, 2.0)  # Within tolerance
        v4 = Vector2(1.0 + 1e-5, 2.0)  # Outside tolerance

        assert v1 == v2
        assert v1 == v3
        assert not (v1 == v4)


# ============================================================================
# 9. Edge Cases and Boundary Conditions
# ============================================================================
class TestVector2EdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_small_vectors(self):
        """Test operations on very small vectors"""
        v = Vector2(1e-10, 1e-10)
        normalized = v.normalize()

        # Should either be zero or properly normalized
        if v.magnitude > 1e-12:
            assert abs(normalized.magnitude - 1.0) < 1e-6

    def test_very_large_vectors(self):
        """Test operations on very large vectors"""
        v = Vector2(1e10, 1e10)
        normalized = v.normalize()
        assert abs(normalized.magnitude - 1.0) < 1e-6

    def test_negative_vector_operations(self):
        """Test operations preserve sign correctness"""
        v1 = Vector2(-3.0, -4.0)
        v2 = Vector2(3.0, 4.0)

        # Magnitude should be same regardless of sign
        assert abs(v1.magnitude - v2.magnitude) < 1e-10

        # But direction should be opposite
        angle = v1.angle_to(v2)
        assert abs(angle - math.pi) < 1e-6

    def test_orthogonal_vectors(self):
        """Test orthogonal vector properties"""
        v1 = Vector2(1.0, 0.0)
        v2 = Vector2(0.0, 1.0)

        # Perpendicular vectors have dot product = 0
        assert abs(v1.dot(v2)) < 1e-10

        # Angle between perpendicular vectors is π/2
        assert abs(v1.angle_to(v2) - math.pi/2) < 1e-6

    def test_parallel_vectors(self):
        """Test parallel vector properties"""
        v1 = Vector2(1.0, 2.0)
        v2 = Vector2(2.0, 4.0)  # Parallel (2x)

        # Angle between parallel vectors is 0
        assert abs(v1.angle_to(v2)) < 1e-6

        # Cross product of parallel vectors is 0
        assert abs(v1.cross(v2)) < 1e-10
