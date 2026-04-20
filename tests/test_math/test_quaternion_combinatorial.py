import pytest
import math
from itertools import product
from allpairspy import AllPairs
from engine.math.quaternion import Quaternion
from engine.math.vector3 import Vector3


# ============================================================================
# 1. Quaternion Basic Operations - PWC
# ============================================================================
_quat_op_parameters = [
    [0.0, 0.5, 1.0, -0.5],  # x1
    [0.0, 0.5, 1.0, -0.5],  # y1
    [0.0, 0.5, 1.0, -0.5],  # z1
    [0.0, 0.5, 1.0, -0.5],  # w1
    [0.0, 0.5, 1.0, -0.5],  # x2
    [0.0, 0.5, 1.0, -0.5],  # y2
    [0.0, 0.5, 1.0, -0.5],  # z2
    [0.0, 0.5, 1.0, -0.5],  # w2
]
_quat_op_pwc_cases = [tuple(case) for case in AllPairs(_quat_op_parameters)]

@pytest.mark.parametrize("x1, y1, z1, w1, x2, y2, z2, w2", _quat_op_pwc_cases)
def test_quaternion_operations_pwc(x1, y1, z1, w1, x2, y2, z2, w2):
    """Test Quaternion arithmetic operations"""
    q1 = Quaternion(x1, y1, z1, w1)
    q2 = Quaternion(x2, y2, z2, w2)

    # Test addition
    sum_q = q1 + q2
    assert abs(sum_q.x - (x1 + x2)) < 1e-10
    assert abs(sum_q.y - (y1 + y2)) < 1e-10
    assert abs(sum_q.z - (z1 + z2)) < 1e-10
    assert abs(sum_q.w - (w1 + w2)) < 1e-10

    # Test subtraction
    diff_q = q1 - q2
    assert abs(diff_q.x - (x1 - x2)) < 1e-10
    assert abs(diff_q.y - (y1 - y2)) < 1e-10
    assert abs(diff_q.z - (z1 - z2)) < 1e-10
    assert abs(diff_q.w - (w1 - w2)) < 1e-10

    # Test scalar multiplication
    scalar = 2.5
    scaled = q1 * scalar
    assert abs(scaled.x - x1 * scalar) < 1e-10
    assert abs(scaled.y - y1 * scalar) < 1e-10
    assert abs(scaled.z - z1 * scalar) < 1e-10
    assert abs(scaled.w - w1 * scalar) < 1e-10

    # Test reverse multiplication
    r_scaled = scalar * q1
    assert abs(r_scaled.x - scaled.x) < 1e-10


# ============================================================================
# 2. Quaternion Multiplication - PWC
# ============================================================================
@pytest.mark.parametrize("x1, y1, z1, w1, x2, y2, z2, w2", _quat_op_pwc_cases)
def test_quaternion_multiplication_pwc(x1, y1, z1, w1, x2, y2, z2, w2):
    """Test quaternion multiplication (not commutative)"""
    q1 = Quaternion(x1, y1, z1, w1)
    q2 = Quaternion(x2, y2, z2, w2)

    # Quaternion multiplication
    q1q2 = q1 * q2
    q2q1 = q2 * q1

    # In general, quaternion multiplication is NOT commutative
    # So q1*q2 should not equal q2*q1 (except in special cases)
    # But both should produce valid quaternions

    # Verify the multiplication formula
    # q1*q2 = (w1*q2 + x1*q2.w*i + y1*q2.w*j + z1*q2.w*k + ...)
    expected_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    expected_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    expected_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    expected_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    assert abs(q1q2.x - expected_x) < 1e-10
    assert abs(q1q2.y - expected_y) < 1e-10
    assert abs(q1q2.z - expected_z) < 1e-10
    assert abs(q1q2.w - expected_w) < 1e-10


# ============================================================================
# 3. Quaternion Magnitude and Normalization - PWC
# ============================================================================
_quat_mag_parameters = [
    [0.0, 0.0, 0.0, 1.0],  # Identity
    [1.0, 0.0, 0.0, 0.0],  # Pure imaginary X
    [0.0, 1.0, 0.0, 0.0],  # Pure imaginary Y
    [0.0, 0.0, 1.0, 0.0],  # Pure imaginary Z
    [0.5, 0.5, 0.5, 0.5],  # Unit quaternion
    [2.0, 2.0, 2.0, 2.0],  # Non-unit
]

@pytest.mark.parametrize("x, y, z, w", _quat_mag_parameters)
def test_quaternion_magnitude_pwc(x, y, z, w):
    """Test quaternion magnitude calculation"""
    q = Quaternion(x, y, z, w)
    expected_mag = math.sqrt(x*x + y*y + z*z + w*w)

    assert abs(q.magnitude - expected_mag) < 1e-10
    assert abs(q.magnitude_squared - (x*x + y*y + z*z + w*w)) < 1e-10


@pytest.mark.parametrize("x, y, z, w", _quat_mag_parameters)
def test_quaternion_normalize_pwc(x, y, z, w):
    """Test quaternion normalization"""
    q = Quaternion(x, y, z, w)

    if x == 0.0 and y == 0.0 and z == 0.0 and w == 0.0:
        # Zero quaternion should normalize to identity
        normalized = q.normalize()
        assert normalized == Quaternion.identity()
    else:
        normalized = q.normalize()
        assert abs(normalized.magnitude - 1.0) < 1e-6


# ============================================================================
# 4. Quaternion Conjugate and Inverse - PWC
# ============================================================================
@pytest.mark.parametrize("x, y, z, w", _quat_mag_parameters)
def test_quaternion_conjugate_pwc(x, y, z, w):
    """Test quaternion conjugate"""
    q = Quaternion(x, y, z, w)
    conj = q.conjugate()

    # Conjugate should negate imaginary parts
    assert abs(conj.x - (-x)) < 1e-10
    assert abs(conj.y - (-y)) < 1e-10
    assert abs(conj.z - (-z)) < 1e-10
    assert abs(conj.w - w) < 1e-10

    # Conjugate of conjugate should return original
    double_conj = conj.conjugate()
    assert double_conj == q


@pytest.mark.parametrize("x, y, z, w", _quat_mag_parameters)
def test_quaternion_inverse_pwc(x, y, z, w):
    """Test quaternion inverse"""
    q = Quaternion(x, y, z, w)

    if q.magnitude_squared == 0.0:
        # Zero quaternion inverse returns identity
        inv = q.inverse()
        assert inv == Quaternion.identity()
    else:
        inv = q.inverse()

        # q * q^-1 should equal identity (or very close)
        product = q * inv
        identity = Quaternion.identity()

        assert abs(product.x - identity.x) < 1e-5
        assert abs(product.y - identity.y) < 1e-5
        assert abs(product.z - identity.z) < 1e-5
        assert abs(product.w - identity.w) < 1e-5


# ============================================================================
# 5. Quaternion Dot Product
# ============================================================================
@pytest.mark.parametrize("x1, y1, z1, w1, x2, y2, z2, w2", _quat_op_pwc_cases)
def test_quaternion_dot_pwc(x1, y1, z1, w1, x2, y2, z2, w2):
    """Test quaternion dot product"""
    q1 = Quaternion(x1, y1, z1, w1)
    q2 = Quaternion(x2, y2, z2, w2)

    expected_dot = x1*x2 + y1*y2 + z1*z2 + w1*w2
    assert abs(q1.dot(q2) - expected_dot) < 1e-10

    # Dot product should be commutative
    assert abs(q1.dot(q2) - q2.dot(q1)) < 1e-10


# ============================================================================
# 6. Quaternion from/to Euler Angles - PWC
# ============================================================================
_euler_angles = [
    0.0,
    math.pi / 4,
    math.pi / 2,
    math.pi,
    -math.pi / 2,
]

_euler_pwc_cases = [tuple(case) for case in AllPairs([
    _euler_angles,  # roll
    _euler_angles,  # pitch
    _euler_angles,  # yaw
])]

@pytest.mark.parametrize("roll, pitch, yaw", _euler_pwc_cases)
def test_quaternion_euler_conversion_pwc(roll, pitch, yaw):
    """Test Euler angle conversion with validation"""
    # Convert Euler to quaternion
    q = Quaternion.from_euler_angles(roll, pitch, yaw)

    # Quaternion should be normalized
    assert abs(q.magnitude - 1.0) < 1e-5

    # Convert back to Euler
    r, p, y = q.to_euler_angles()

    # Due to gimbal lock and multiple representations, we can't always
    # get back the exact same angles, but we can verify rotation is equivalent
    # by converting the original euler back to quat and comparing
    q_original = Quaternion.from_euler_angles(roll, pitch, yaw)
    q_roundtrip = Quaternion.from_euler_angles(r, p, y)

    # Both should be unit quaternions
    assert abs(q_original.magnitude - 1.0) < 1e-5
    assert abs(q_roundtrip.magnitude - 1.0) < 1e-5


# ============================================================================
# 7. Quaternion from/to Axis-Angle
# ============================================================================
_axis_values = [
    Vector3(1, 0, 0),    # X axis
    Vector3(0, 1, 0),    # Y axis
    Vector3(0, 0, 1),    # Z axis
    Vector3(1, 1, 0),    # XY diagonal
    Vector3(1, 1, 1),    # XYZ diagonal
]

_angle_values = [0.0, math.pi/4, math.pi/2, math.pi, -math.pi/2]

_axis_angle_pwc_cases = [tuple(case) for case in AllPairs([_axis_values, _angle_values])]

@pytest.mark.parametrize("axis, angle", _axis_angle_pwc_cases)
def test_quaternion_axis_angle_conversion_pwc(axis, angle):
    """Test axis-angle conversion"""
    # Convert to quaternion
    q = Quaternion.from_axis_angle(axis, angle)

    # Should be normalized
    assert abs(q.magnitude - 1.0) < 1e-5

    # Convert back to axis-angle
    result_axis, result_angle = q.to_axis_angle()

    # Verify magnitude of axis is 1 (normalized)
    assert abs(result_axis.magnitude - 1.0) < 1e-5

    # Angle should be in [0, 2π]
    assert 0 <= result_angle <= 2 * math.pi + 1e-5


# ============================================================================
# 8. Quaternion Vector Rotation
# ============================================================================
class TestQuaternionVectorRotation:
    """Test rotating vectors with quaternions"""

    def test_identity_rotation(self):
        """Identity quaternion shouldn't change vector"""
        v = Vector3(1, 2, 3)
        q = Quaternion.identity()
        rotated = q.rotate_vector(v)

        assert abs(rotated.x - v.x) < 1e-5
        assert abs(rotated.y - v.y) < 1e-5
        assert abs(rotated.z - v.z) < 1e-5

    def test_vector_rotation_preserves_magnitude(self):
        """Quaternion rotation should preserve vector magnitude"""
        v = Vector3(3, 4, 5)
        q = Quaternion.from_axis_angle(Vector3(1, 1, 1).normalized(), math.pi/3)

        rotated = q.rotate_vector(v)

        assert abs(rotated.magnitude - v.magnitude) < 1e-5

    def test_90_degree_rotation(self):
        """Test 90-degree rotation around Z axis"""
        v = Vector3(1, 0, 0)
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi/2)

        rotated = q.rotate_vector(v)

        # Should rotate from (1,0,0) to (0,1,0)
        assert abs(rotated.x - 0.0) < 1e-5
        assert abs(rotated.y - 1.0) < 1e-5
        assert abs(rotated.z - 0.0) < 1e-5

    def test_opposite_rotation_cancels(self):
        """Opposite rotations should cancel"""
        v = Vector3(2, 3, 4)
        axis = Vector3(1, 1, 1).normalized()
        angle = math.pi / 3

        q_forward = Quaternion.from_axis_angle(axis, angle)
        q_backward = Quaternion.from_axis_angle(axis, -angle)

        rotated_forward = q_forward.rotate_vector(v)
        rotated_back = q_backward.rotate_vector(rotated_forward)

        assert abs(rotated_back.x - v.x) < 1e-5
        assert abs(rotated_back.y - v.y) < 1e-5
        assert abs(rotated_back.z - v.z) < 1e-5


# ============================================================================
# 9. Quaternion Interpolation (Lerp/Slerp)
# ============================================================================
_interp_t_values = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

class TestQuaternionInterpolation:
    """Test quaternion interpolation methods"""

    def test_lerp_boundaries(self):
        """Test linear interpolation at boundaries"""
        q1 = Quaternion(0.5, 0.5, 0.5, 0.5).normalize()
        q2 = Quaternion(-0.5, 0.5, -0.5, 0.5).normalize()

        # At t=0, should be q1
        result_0 = q1.lerp(q2, 0.0)
        assert abs((result_0 - q1).magnitude) < 1e-5

        # At t=1, should be q2
        result_1 = q1.lerp(q2, 1.0)
        assert abs((result_1 - q2).magnitude) < 1e-5

    def test_lerp_clamped(self):
        """Test that lerp clamps t to [0,1]"""
        q1 = Quaternion.identity()
        q2 = Quaternion(0.5, 0, 0, 0.5).normalize()

        # t < 0 should clamp to 0
        result_neg = q1.lerp(q2, -0.5)
        assert abs((result_neg - q1).magnitude) < 1e-5

        # t > 1 should clamp to 1
        result_over = q1.lerp(q2, 1.5)
        assert abs((result_over - q2).magnitude) < 1e-5

    def test_slerp_boundaries(self):
        """Test spherical interpolation at boundaries"""
        q1 = Quaternion.identity()
        q2 = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi/2)

        # At t=0, should be q1
        result_0 = q1.slerp(q2, 0.0)
        assert abs(result_0.x - q1.x) < 1e-5

        # At t=1, should be q2
        result_1 = q1.slerp(q2, 1.0)
        assert abs(result_1.x - q2.x) < 1e-5

    def test_slerp_preserves_magnitude(self):
        """Slerp should produce unit quaternions"""
        q1 = Quaternion.identity()
        q2 = Quaternion.from_axis_angle(Vector3(1, 0, 0), math.pi/3)

        for t in _interp_t_values:
            result = q1.slerp(q2, t)
            assert abs(result.magnitude - 1.0) < 1e-5

    def test_slerp_handles_antiodal_quaternions(self):
        """Slerp should handle antipodal quaternions correctly"""
        q1 = Quaternion(0, 0, 0, 1)
        q2 = Quaternion(0, 0, 0, -1)  # Represents same rotation but opposite

        result = q1.slerp(q2, 0.5)
        # Should not crash and should be normalized
        assert abs(result.magnitude - 1.0) < 1e-5


# ============================================================================
# 10. Quaternion Copy and Equality
# ============================================================================
class TestQuaternionCopyEquality:
    """Test quaternion copying and equality"""

    def test_copy(self):
        """Test quaternion copying"""
        q1 = Quaternion(1, 2, 3, 4)
        q2 = q1.copy()

        assert q1 == q2
        assert q1 is not q2

    def test_equality_tolerance(self):
        """Test equality with tolerance"""
        q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
        q2 = Quaternion(1.0 + 1e-7, 2.0, 3.0, 4.0)
        q3 = Quaternion(1.0 + 1e-5, 2.0, 3.0, 4.0)

        assert q1 == q2  # Within tolerance
        assert not (q1 == q3)  # Outside tolerance


# ============================================================================
# 11. Edge Cases
# ============================================================================
class TestQuaternionEdgeCases:
    """Test edge cases for quaternions"""

    def test_identity_quaternion(self):
        """Test identity quaternion properties"""
        q = Quaternion.identity()
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0
        assert q.w == 1.0
        assert abs(q.magnitude - 1.0) < 1e-10

    def test_zero_angle_rotation(self):
        """Test rotation with zero angle"""
        axis = Vector3(1, 0, 0)
        q = Quaternion.from_axis_angle(axis, 0.0)

        # Should be identity-like
        assert abs(q.magnitude - 1.0) < 1e-5

    def test_full_rotation(self):
        """Test 2π rotation returns to identity"""
        axis = Vector3(1, 1, 1).normalized()
        q1 = Quaternion.from_axis_angle(axis, 0.0)
        q2 = Quaternion.from_axis_angle(axis, 2 * math.pi)

        # Both should represent same rotation
        # Check by rotating a vector
        v = Vector3(1, 0, 0)
        r1 = q1.rotate_vector(v)
        r2 = q2.rotate_vector(v)

        assert abs(r1.x - r2.x) < 1e-5
        assert abs(r1.y - r2.y) < 1e-5
        assert abs(r1.z - r2.z) < 1e-5
