"""
Blackbox tests for Quaternion using Equivalence Partitioning (EP),
Boundary Analysis (BA), and Error Guessing (EG).
"""
import math
import pytest
from engine.math.quaternion import Quaternion
from engine.math.vector3 import Vector3


# ============================================================
# Constructor and Identity Tests
# ============================================================

class TestQuaternionConstructor:

    def test_default_constructor(self):
        """EP: Default quaternion is identity (0,0,0,1)"""
        q = Quaternion()
        assert q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 1.0

    def test_custom_values(self):
        """EP: Custom component values"""
        q = Quaternion(1, 2, 3, 4)
        assert q.x == 1.0 and q.y == 2.0 and q.z == 3.0 and q.w == 4.0

    def test_identity_factory(self):
        """EP: Identity factory method"""
        q = Quaternion.identity()
        assert q == Quaternion(0, 0, 0, 1)

    def test_integer_inputs_to_float(self):
        """EP: Integer inputs converted to float"""
        q = Quaternion(1, 2, 3, 4)
        assert isinstance(q.w, float)


# ============================================================
# Arithmetic Tests
# ============================================================

class TestQuaternionArithmetic:

    def test_add(self):
        """EP: Quaternion addition"""
        result = Quaternion(1, 2, 3, 4) + Quaternion(5, 6, 7, 8)
        assert result == Quaternion(6, 8, 10, 12)

    def test_subtract(self):
        """EP: Quaternion subtraction"""
        result = Quaternion(5, 6, 7, 8) - Quaternion(1, 2, 3, 4)
        assert result == Quaternion(4, 4, 4, 4)

    def test_subtract_self(self):
        """BA: Subtract self gives zero quaternion"""
        q = Quaternion(1, 2, 3, 4)
        result = q - q
        assert result == Quaternion(0, 0, 0, 0)

    def test_multiply_quaternions(self):
        """EP: Quaternion-quaternion multiplication (Hamilton product)"""
        # Identity * any = any
        q = Quaternion(1, 2, 3, 4)
        result = Quaternion.identity() * q
        assert result == q

    def test_multiply_by_scalar(self):
        """EP: Scalar multiplication"""
        result = Quaternion(1, 2, 3, 4) * 2
        assert result == Quaternion(2, 4, 6, 8)

    def test_multiply_by_zero_scalar(self):
        """BA: Scalar multiply by zero"""
        result = Quaternion(1, 2, 3, 4) * 0
        assert result == Quaternion(0, 0, 0, 0)

    def test_rmul_scalar(self):
        """EP: Reverse scalar multiply"""
        result = 3 * Quaternion(1, 2, 3, 4)
        assert result == Quaternion(3, 6, 9, 12)

    def test_multiply_identity_is_identity(self):
        """BA: Identity * identity = identity"""
        result = Quaternion.identity() * Quaternion.identity()
        assert result == Quaternion.identity()


# ============================================================
# Equality Tests
# ============================================================

class TestQuaternionEquality:

    def test_equal(self):
        """EP: Equal quaternions"""
        assert Quaternion(1, 2, 3, 4) == Quaternion(1, 2, 3, 4)

    def test_not_equal(self):
        """EP: Unequal quaternions"""
        assert not (Quaternion(1, 2, 3, 4) == Quaternion(5, 6, 7, 8))

    def test_within_tolerance(self):
        """BA: Within 1e-6 tolerance"""
        assert Quaternion(1, 2, 3, 4) == Quaternion(1 + 1e-7, 2, 3, 4)

    def test_outside_tolerance(self):
        """BA: Outside 1e-6 tolerance"""
        assert not (Quaternion(1, 2, 3, 4) == Quaternion(1 + 1e-5, 2, 3, 4))


# ============================================================
# Magnitude and Normalize Tests
# ============================================================

class TestQuaternionMagnitude:

    def test_identity_magnitude(self):
        """EP: Identity quaternion has magnitude 1"""
        assert abs(Quaternion.identity().magnitude - 1.0) < 1e-6

    def test_magnitude_known(self):
        """EP: Known magnitude: (0,0,0,2) -> 2"""
        assert abs(Quaternion(0, 0, 0, 2).magnitude - 2.0) < 1e-6

    def test_magnitude_squared(self):
        """EP: Squared magnitude"""
        q = Quaternion(1, 2, 3, 4)
        expected = 1 + 4 + 9 + 16
        assert abs(q.magnitude_squared - expected) < 1e-6

    def test_normalize(self):
        """EP: Normalized quaternion has magnitude 1"""
        q = Quaternion(1, 2, 3, 4)
        result = q.normalize()
        assert abs(result.magnitude - 1.0) < 1e-6

    def test_normalize_identity_unchanged(self):
        """BA: Normalizing identity returns identity"""
        result = Quaternion.identity().normalize()
        assert result == Quaternion.identity()

    def test_normalize_zero_returns_identity(self):
        """BA: Normalizing zero quaternion returns identity"""
        result = Quaternion(0, 0, 0, 0).normalize()
        assert result == Quaternion.identity()

    def test_normalized_alias(self):
        """EP: normalized() matches normalize()"""
        q = Quaternion(1, 2, 3, 4)
        assert q.normalized() == q.normalize()


# ============================================================
# Conjugate and Inverse Tests
# ============================================================

class TestQuaternionConjugateInverse:

    def test_conjugate(self):
        """EP: Conjugate negates imaginary parts"""
        q = Quaternion(1, 2, 3, 4)
        result = q.conjugate()
        assert result == Quaternion(-1, -2, -3, 4)

    def test_conjugate_identity(self):
        """BA: Conjugate of identity is identity"""
        assert Quaternion.identity().conjugate() == Quaternion.identity()

    def test_inverse_of_unit_quaternion(self):
        """EP: Inverse of unit quaternion equals conjugate"""
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 4)
        inv = q.inverse()
        conj = q.conjugate()
        assert abs(inv.x - conj.x) < 1e-5
        assert abs(inv.y - conj.y) < 1e-5
        assert abs(inv.z - conj.z) < 1e-5
        assert abs(inv.w - conj.w) < 1e-5

    def test_inverse_times_original_is_identity(self):
        """EP: q * q^-1 = identity"""
        q = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 3)
        result = q * q.inverse()
        identity = Quaternion.identity()
        assert abs(result.x - identity.x) < 1e-5
        assert abs(result.y - identity.y) < 1e-5
        assert abs(result.z - identity.z) < 1e-5
        assert abs(result.w - identity.w) < 1e-5

    def test_inverse_zero_returns_identity(self):
        """BA: Inverse of zero quaternion returns identity"""
        result = Quaternion(0, 0, 0, 0).inverse()
        assert result == Quaternion.identity()


# ============================================================
# Dot Product Tests
# ============================================================

class TestQuaternionDot:

    def test_dot_with_self(self):
        """EP: Dot with self equals magnitude squared"""
        q = Quaternion(1, 2, 3, 4)
        assert abs(q.dot(q) - q.magnitude_squared) < 1e-6

    def test_dot_identity_with_identity(self):
        """BA: dot(identity, identity) = 1"""
        assert abs(Quaternion.identity().dot(Quaternion.identity()) - 1.0) < 1e-6

    def test_dot_zero(self):
        """BA: Dot with zero quaternion = 0"""
        assert Quaternion(1, 2, 3, 4).dot(Quaternion(0, 0, 0, 0)) == 0.0


# ============================================================
# Rotation Tests
# ============================================================

class TestQuaternionRotation:

    def test_rotate_vector_by_identity(self):
        """BA: Rotating by identity leaves vector unchanged"""
        v = Vector3(1, 2, 3)
        result = Quaternion.identity().rotate_vector(v)
        assert abs(result.x - v.x) < 1e-5
        assert abs(result.y - v.y) < 1e-5
        assert abs(result.z - v.z) < 1e-5

    def test_rotate_vector_90_around_z(self):
        """EP: Rotate (1,0,0) by 90 degrees around Z -> (0,1,0)"""
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        result = q.rotate_vector(Vector3(1, 0, 0))
        assert abs(result.x - 0) < 1e-5
        assert abs(result.y - 1) < 1e-5
        assert abs(result.z - 0) < 1e-5

    def test_rotate_vector_180_around_y(self):
        """EP: Rotate (1,0,0) by 180 degrees around Y -> (-1,0,0)"""
        q = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi)
        result = q.rotate_vector(Vector3(1, 0, 0))
        assert abs(result.x - (-1)) < 1e-5
        assert abs(result.z - 0) < 1e-4

    def test_rotate_zero_vector(self):
        """BA: Rotating zero vector gives zero vector"""
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 4)
        result = q.rotate_vector(Vector3.zero())
        assert abs(result.x) < 1e-5 and abs(result.y) < 1e-5 and abs(result.z) < 1e-5

    def test_rotate_360_returns_original(self):
        """BA: Full 360 rotation returns original vector"""
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), 2 * math.pi)
        v = Vector3(3, 4, 5)
        result = q.rotate_vector(v)
        assert abs(result.x - v.x) < 1e-4
        assert abs(result.y - v.y) < 1e-4
        assert abs(result.z - v.z) < 1e-4


# ============================================================
# Euler Angle Conversion Tests
# ============================================================

class TestQuaternionEuler:

    def test_identity_to_euler(self):
        """BA: Identity quaternion gives (0,0,0) euler angles"""
        roll, pitch, yaw = Quaternion.identity().to_euler_angles()
        assert abs(roll) < 1e-6 and abs(pitch) < 1e-6 and abs(yaw) < 1e-6

    def test_euler_roundtrip(self):
        """EP: from_euler -> to_euler should roundtrip"""
        roll, pitch, yaw = 0.3, 0.4, 0.5
        q = Quaternion.from_euler_angles(roll, pitch, yaw)
        r2, p2, y2 = q.to_euler_angles()
        assert abs(r2 - roll) < 1e-5
        assert abs(p2 - pitch) < 1e-5
        assert abs(y2 - yaw) < 1e-5

    def test_euler_gimbal_lock(self):
        """BA: Pitch at +/- pi/2 (gimbal lock edge case)"""
        q = Quaternion.from_euler_angles(0, math.pi / 2, 0)
        _, pitch, _ = q.to_euler_angles()
        assert abs(pitch - math.pi / 2) < 1e-5


# ============================================================
# Axis-Angle Conversion Tests
# ============================================================

class TestQuaternionAxisAngle:

    def test_from_axis_angle_z_90(self):
        """EP: 90 degrees around Z axis"""
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        assert abs(q.magnitude - 1.0) < 1e-6

    def test_from_axis_angle_zero_angle(self):
        """BA: Zero angle gives identity-like quaternion"""
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), 0)
        assert abs(q.w - 1.0) < 1e-6
        assert abs(q.x) < 1e-6 and abs(q.y) < 1e-6 and abs(q.z) < 1e-6

    def test_to_axis_angle_roundtrip(self):
        """EP: from_axis_angle -> to_axis_angle roundtrip"""
        axis_in = Vector3(0, 1, 0)
        angle_in = math.pi / 3
        q = Quaternion.from_axis_angle(axis_in, angle_in)
        axis_out, angle_out = q.to_axis_angle()
        assert abs(angle_out - angle_in) < 1e-5

    def test_to_axis_angle_identity(self):
        """BA: Identity quaternion gives 0 angle"""
        _, angle = Quaternion.identity().to_axis_angle()
        assert abs(angle) < 1e-6


# ============================================================
# Lerp and Slerp Tests
# ============================================================

class TestQuaternionInterpolation:

    def test_lerp_at_zero(self):
        """BA: lerp t=0 returns start (normalized)"""
        a = Quaternion.identity()
        b = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        result = a.lerp(b, 0)
        assert abs(result.w - a.w) < 1e-5

    def test_lerp_at_one(self):
        """BA: lerp t=1 returns end (normalized)"""
        a = Quaternion.identity()
        b = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        result = a.lerp(b, 1)
        assert abs(result.x - b.x) < 1e-5 and abs(result.w - b.w) < 1e-5

    def test_lerp_clamps_negative(self):
        """BA: lerp t<0 clamped"""
        a = Quaternion.identity()
        b = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        result = a.lerp(b, -1)
        assert abs(result.w - a.normalize().w) < 1e-5

    def test_lerp_clamps_above_one(self):
        """BA: lerp t>1 clamped"""
        a = Quaternion.identity()
        b = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        result = a.lerp(b, 2)
        assert abs(result.x - b.x) < 1e-5

    def test_slerp_at_zero(self):
        """BA: slerp t=0 returns start"""
        a = Quaternion.identity()
        b = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        result = a.slerp(b, 0)
        assert abs(result.w - a.w) < 1e-4

    def test_slerp_at_one(self):
        """BA: slerp t=1 returns end"""
        a = Quaternion.identity()
        b = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        result = a.slerp(b, 1)
        assert abs(result.x - b.x) < 1e-4

    def test_slerp_midpoint_magnitude(self):
        """EP: slerp result should have magnitude 1"""
        a = Quaternion.identity()
        b = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        result = a.slerp(b, 0.5)
        assert abs(result.magnitude - 1.0) < 1e-5

    def test_slerp_nearly_identical_uses_lerp(self):
        """BA: When quaternions are very similar (dot > 0.9995), slerp falls back to lerp"""
        a = Quaternion.identity()
        b = Quaternion.from_axis_angle(Vector3(0, 0, 1), 0.0001)  # Very small rotation
        result = a.slerp(b, 0.5)
        assert abs(result.magnitude - 1.0) < 1e-5


# ============================================================
# Rotation Matrix Conversion Tests
# ============================================================

class TestQuaternionRotationMatrix:

    def test_identity_matrix(self):
        """EP: Identity rotation matrix gives identity quaternion"""
        identity_matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        q = Quaternion.from_rotation_matrix(identity_matrix)
        assert abs(q.w - 1.0) < 1e-5
        assert abs(q.x) < 1e-5 and abs(q.y) < 1e-5 and abs(q.z) < 1e-5

    def test_90_degree_z_rotation_matrix(self):
        """EP: 90-degree Z rotation matrix"""
        # Rotation matrix for 90 deg around Z: [0,-1,0, 1,0,0, 0,0,1]
        matrix = [0, -1, 0, 1, 0, 0, 0, 0, 1]
        q = Quaternion.from_rotation_matrix(matrix)
        # Should be equivalent to from_axis_angle(Z, pi/2)
        expected = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        assert abs(abs(q.dot(expected)) - 1.0) < 1e-4


# ============================================================
# Error Guessing
# ============================================================

class TestQuaternionErrorGuessing:

    def test_copy_independent(self):
        """EG: Copy is independent"""
        q = Quaternion(1, 2, 3, 4)
        c = q.copy()
        c.x = 99
        assert q.x == 1.0

    def test_double_rotation(self):
        """EG: Two 90-degree rotations = one 180-degree rotation"""
        q90 = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        q180 = q90 * q90
        v = Vector3(1, 0, 0)
        result = q180.rotate_vector(v)
        assert abs(result.x - (-1)) < 1e-4
        assert abs(result.y - 0) < 1e-4

    def test_str_format(self):
        """EP: String format"""
        s = str(Quaternion(1, 2, 3, 4))
        assert "Quaternion" in s

    def test_repr_matches_str(self):
        """EP: repr matches str"""
        q = Quaternion(1, 2, 3, 4)
        assert repr(q) == str(q)

    def test_quaternion_multiplication_not_commutative(self):
        """EG: Quaternion multiplication is NOT commutative in general"""
        a = Quaternion.from_axis_angle(Vector3(1, 0, 0), math.pi / 4)
        b = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 4)
        ab = a * b
        ba = b * a
        # They should generally NOT be equal
        is_equal = (abs(ab.x - ba.x) < 1e-5 and abs(ab.y - ba.y) < 1e-5 and
                    abs(ab.z - ba.z) < 1e-5 and abs(ab.w - ba.w) < 1e-5)
        assert not is_equal
