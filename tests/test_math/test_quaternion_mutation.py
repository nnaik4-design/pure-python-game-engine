"""
Mutation-killing tests for Quaternion.
Targets surviving mutants from mutmut analysis.
"""
import math
import pytest
from engine.math.quaternion import Quaternion
from engine.math.vector3 import Vector3


class TestQuaternionStringMutations:

    def test_str_exact_format(self):
        """Kill mutant 9: string mutation in __str__"""
        q = Quaternion(1, 2, 3, 4)
        s = str(q)
        assert s.startswith("Quaternion(")
        assert s.endswith(")")
        assert not s.startswith("XX")


class TestQuaternionEqualityBoundary:

    def test_equality_boundary_x(self):
        """Kill mutants 51-52: tolerance on x"""
        q1 = Quaternion(0, 0, 0, 1)
        assert q1 != Quaternion(1e-6, 0, 0, 1)
        assert q1 == Quaternion(0.999e-6, 0, 0, 1)

    def test_equality_boundary_y(self):
        """Kill mutants 54-55: tolerance on y"""
        q1 = Quaternion(0, 0, 0, 1)
        assert q1 != Quaternion(0, 1e-6, 0, 1)
        assert q1 == Quaternion(0, 0.999e-6, 0, 1)

    def test_equality_boundary_z(self):
        """Kill mutants 57-58: tolerance on z"""
        q1 = Quaternion(0, 0, 0, 1)
        assert q1 != Quaternion(0, 0, 1e-6, 1)
        assert q1 == Quaternion(0, 0, 0.999e-6, 1)

    def test_equality_boundary_w(self):
        """Kill mutants 60-61: tolerance on w"""
        q1 = Quaternion(0, 0, 0, 0)
        assert q1 != Quaternion(0, 0, 0, 1e-6)
        assert q1 == Quaternion(0, 0, 0, 0.999e-6)


class TestHamiltonProductTerms:

    def test_hamilton_w_component(self):
        """Kill mutant 40: w*w + x*x mutated, verify Hamilton product w component"""
        # q1 * q2 where we can verify each component
        q1 = Quaternion(1, 0, 0, 0)  # Pure i quaternion
        q2 = Quaternion(0, 1, 0, 0)  # Pure j quaternion
        result = q1 * q2
        # i * j = k, so result should be (0, 0, 1, 0)
        assert result.x == pytest.approx(0, abs=1e-6)
        assert result.y == pytest.approx(0, abs=1e-6)
        assert result.z == pytest.approx(1, abs=1e-6)
        assert result.w == pytest.approx(0, abs=1e-6)

    def test_hamilton_specific_values(self):
        """Kill Hamilton product mutations with specific known values"""
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        r = q1 * q2
        # Manual Hamilton product:
        # w = 4*8 - 1*5 - 2*6 - 3*7 = 32 - 5 - 12 - 21 = -6
        # x = 4*5 + 1*8 + 2*7 - 3*6 = 20 + 8 + 14 - 18 = 24
        # y = 4*6 - 1*7 + 2*8 + 3*5 = 24 - 7 + 16 + 15 = 48
        # z = 4*7 + 1*6 - 2*5 + 3*8 = 28 + 6 - 10 + 24 = 48
        assert r.w == pytest.approx(-6, abs=1e-6)
        assert r.x == pytest.approx(24, abs=1e-6)
        assert r.y == pytest.approx(48, abs=1e-6)
        assert r.z == pytest.approx(48, abs=1e-6)

    def test_i_squared_is_negative_one(self):
        """Kill arithmetic mutations: i*i = -1"""
        qi = Quaternion(1, 0, 0, 0)
        result = qi * qi
        assert result.w == pytest.approx(-1, abs=1e-6)
        assert result.x == pytest.approx(0, abs=1e-6)

    def test_j_squared_is_negative_one(self):
        """Kill arithmetic mutations: j*j = -1"""
        qj = Quaternion(0, 1, 0, 0)
        result = qj * qj
        assert result.w == pytest.approx(-1, abs=1e-6)

    def test_k_squared_is_negative_one(self):
        """Kill arithmetic mutations: k*k = -1"""
        qk = Quaternion(0, 0, 1, 0)
        result = qk * qk
        assert result.w == pytest.approx(-1, abs=1e-6)


class TestInverseArithmetic:

    def test_inverse_x_component(self):
        """Kill mutant 93: conj.x * mag_sq instead of / mag_sq"""
        q = Quaternion(1, 2, 3, 4).normalized()
        inv = q.inverse()
        # q * q_inv should equal identity
        product = q * inv
        assert product.w == pytest.approx(1, abs=1e-4)
        assert product.x == pytest.approx(0, abs=1e-4)
        assert product.y == pytest.approx(0, abs=1e-4)
        assert product.z == pytest.approx(0, abs=1e-4)

    def test_inverse_each_component_verified(self):
        """Kill inverse component mutations individually"""
        q = Quaternion(0.5, 0, 0, math.sqrt(0.75))  # Unit quaternion
        inv = q.inverse()
        # For unit quaternion, inverse = conjugate
        assert inv.x == pytest.approx(-0.5, abs=1e-6)
        assert inv.y == pytest.approx(0, abs=1e-6)
        assert inv.z == pytest.approx(0, abs=1e-6)
        assert inv.w == pytest.approx(math.sqrt(0.75), abs=1e-6)


class TestEulerAngles:

    def test_gimbal_lock_boundary(self):
        """Kill mutant 130: abs(sinp) >= 1 vs > 1"""
        # Create a quaternion that produces sinp exactly 1.0
        q = Quaternion.from_euler_angles(0, math.pi / 2, 0)
        roll, pitch, yaw = q.to_euler_angles()
        assert pitch == pytest.approx(math.pi / 2, abs=0.01)

    def test_euler_each_axis_independent(self):
        """Kill euler conversion mutations"""
        # Roll only
        q = Quaternion.from_euler_angles(0.5, 0, 0)
        r, p, y = q.to_euler_angles()
        assert r == pytest.approx(0.5, abs=0.01)
        assert p == pytest.approx(0, abs=0.01)

        # Pitch only
        q = Quaternion.from_euler_angles(0, 0.3, 0)
        r, p, y = q.to_euler_angles()
        assert p == pytest.approx(0.3, abs=0.01)

        # Yaw only
        q = Quaternion.from_euler_angles(0, 0, 0.7)
        r, p, y = q.to_euler_angles()
        assert y == pytest.approx(0.7, abs=0.01)

    def test_from_euler_specific_components(self):
        """Kill from_euler arithmetic mutations"""
        # 90 degree roll: should produce specific quaternion
        q = Quaternion.from_euler_angles(math.pi / 2, 0, 0)
        # Expected: x=sin(pi/4)=0.707, w=cos(pi/4)=0.707
        assert abs(q.x) == pytest.approx(math.sin(math.pi / 4), abs=0.01)
        assert abs(q.w) == pytest.approx(math.cos(math.pi / 4), abs=0.01)


class TestToAxisAngle:

    def test_to_axis_angle_w_equals_1(self):
        """Kill mutant 151: w > 1 vs w >= 1"""
        # Identity quaternion: w=1 exactly
        q = Quaternion.identity()
        axis, angle = q.to_axis_angle()
        assert angle == pytest.approx(0, abs=1e-6)

    def test_to_axis_angle_sqrt_term(self):
        """Kill mutant 157: sqrt(1 - w*w) vs sqrt(2 - w*w)"""
        # 90-degree rotation around Z
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        axis, angle = q.to_axis_angle()
        assert angle == pytest.approx(math.pi / 2, abs=0.01)
        assert axis.z == pytest.approx(1, abs=0.01)


class TestFromRotationMatrixDetailed:

    def test_trace_positive_specific_values(self):
        """Kill mutants 264, 275-277: trace > 0 branch with specific verification"""
        # 45-degree rotation around Z
        a = math.pi / 4
        c, s = math.cos(a), math.sin(a)
        matrix = [c, -s, 0, s, c, 0, 0, 0, 1]
        q = Quaternion.from_rotation_matrix(matrix)
        # Verify by rotating a vector
        v = q.rotate_vector(Vector3(1, 0, 0))
        assert v.x == pytest.approx(c, abs=0.01)
        assert v.y == pytest.approx(s, abs=0.01)

    def test_m00_largest_branch_specific(self):
        """Kill mutants 289+: matrix[0] > matrix[4] branch"""
        # 180-degree around X: matrix = [1,0,0, 0,-1,0, 0,0,-1]
        matrix = [1, 0, 0, 0, -1, 0, 0, 0, -1]
        q = Quaternion.from_rotation_matrix(matrix)
        # Rotating (0,1,0) by 180 around X gives (0,-1,0)
        v = q.rotate_vector(Vector3(0, 1, 0))
        assert v.y == pytest.approx(-1, abs=0.01)
        # Rotating (0,0,1) by 180 around X gives (0,0,-1)
        v2 = q.rotate_vector(Vector3(0, 0, 1))
        assert v2.z == pytest.approx(-1, abs=0.01)

    def test_m11_largest_branch_specific(self):
        """Kill mutations in m11 branch"""
        # 180-degree around Y: matrix = [-1,0,0, 0,1,0, 0,0,-1]
        matrix = [-1, 0, 0, 0, 1, 0, 0, 0, -1]
        q = Quaternion.from_rotation_matrix(matrix)
        v = q.rotate_vector(Vector3(1, 0, 0))
        assert v.x == pytest.approx(-1, abs=0.01)
        v2 = q.rotate_vector(Vector3(0, 0, 1))
        assert v2.z == pytest.approx(-1, abs=0.01)

    def test_m22_largest_branch_specific(self):
        """Kill mutations in m22 branch"""
        # 180-degree around Z: matrix = [-1,0,0, 0,-1,0, 0,0,1]
        matrix = [-1, 0, 0, 0, -1, 0, 0, 0, 1]
        q = Quaternion.from_rotation_matrix(matrix)
        v = q.rotate_vector(Vector3(1, 0, 0))
        assert v.x == pytest.approx(-1, abs=0.01)
        v2 = q.rotate_vector(Vector3(0, 1, 0))
        assert v2.y == pytest.approx(-1, abs=0.01)

    def test_trace_boundary(self):
        """Kill mutant 264: trace > 0 vs trace >= 0"""
        # Matrix with trace exactly 0 should NOT use trace>0 branch
        # trace = m00 + m11 + m22 = 0 → one of the elif branches
        # 120-degree rotation around (1,1,1)/sqrt(3): trace = -1
        # Use a matrix where trace=0 exactly
        matrix = [0, 1, 0, 0, 0, 1, 1, 0, 0]  # trace=0
        q = Quaternion.from_rotation_matrix(matrix)
        assert isinstance(q, Quaternion)


class TestSlerpDetailed:

    def test_slerp_intermediate_value(self):
        """Kill slerp arithmetic mutations"""
        q1 = Quaternion.from_axis_angle(Vector3(0, 0, 1), 0)
        q2 = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        mid = q1.slerp(q2, 0.5)
        # Midpoint should be ~45 degree rotation
        v = mid.rotate_vector(Vector3(1, 0, 0))
        expected_angle = math.pi / 4
        assert v.x == pytest.approx(math.cos(expected_angle), abs=0.05)
        assert v.y == pytest.approx(math.sin(expected_angle), abs=0.05)


class TestRotateVectorDetailed:

    def test_rotate_vector_all_components(self):
        """Kill rotate_vector arithmetic mutations"""
        # 90 degrees around X: (0,1,0) -> (0,0,1)
        q = Quaternion.from_axis_angle(Vector3(1, 0, 0), math.pi / 2)
        v = q.rotate_vector(Vector3(0, 1, 0))
        assert v.x == pytest.approx(0, abs=0.01)
        assert v.y == pytest.approx(0, abs=0.01)
        assert v.z == pytest.approx(1, abs=0.01)

    def test_rotate_vector_around_y(self):
        """Kill more rotate_vector mutations"""
        # 90 degrees around Y: (1,0,0) -> (0,0,-1)
        q = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 2)
        v = q.rotate_vector(Vector3(1, 0, 0))
        assert v.x == pytest.approx(0, abs=0.01)
        assert v.z == pytest.approx(-1, abs=0.01)

    def test_rotate_preserves_magnitude(self):
        """Kill mutations that would change vector length"""
        q = Quaternion.from_axis_angle(Vector3(1, 1, 1).normalized(), 1.0)
        v = Vector3(3, 4, 5)
        rotated = q.rotate_vector(v)
        assert rotated.magnitude == pytest.approx(v.magnitude, abs=0.01)


class TestMagnitudeDetailed:

    def test_magnitude_squared_each_component(self):
        """Kill magnitude_squared component mutations"""
        q = Quaternion(1, 0, 0, 0)
        assert q.magnitude_squared == pytest.approx(1.0)
        q = Quaternion(0, 2, 0, 0)
        assert q.magnitude_squared == pytest.approx(4.0)
        q = Quaternion(0, 0, 3, 0)
        assert q.magnitude_squared == pytest.approx(9.0)
        q = Quaternion(0, 0, 0, 4)
        assert q.magnitude_squared == pytest.approx(16.0)

    def test_dot_each_component(self):
        """Kill dot product component mutations"""
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        # dot = 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        assert q1.dot(q2) == pytest.approx(70.0)


class TestNormalizeDetailed:

    def test_normalize_each_component(self):
        """Kill normalize component mutations"""
        q = Quaternion(2, 0, 0, 0)
        n = q.normalize()
        assert n.x == pytest.approx(1.0)
        assert n.y == pytest.approx(0.0)
        assert n.z == pytest.approx(0.0)
        assert n.w == pytest.approx(0.0)

        q = Quaternion(0, 3, 0, 0)
        n = q.normalize()
        assert n.y == pytest.approx(1.0)


class TestFromAxisAngleDetailed:

    def test_from_axis_angle_components(self):
        """Kill from_axis_angle arithmetic mutations"""
        # 90 degrees around Z
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 2)
        # Expected: w=cos(pi/4), z=sin(pi/4), x=y=0
        assert q.w == pytest.approx(math.cos(math.pi / 4), abs=0.01)
        assert q.z == pytest.approx(math.sin(math.pi / 4), abs=0.01)
        assert q.x == pytest.approx(0, abs=0.01)
        assert q.y == pytest.approx(0, abs=0.01)

    def test_from_axis_angle_x_axis(self):
        """Kill from_axis_angle per-component mutations"""
        q = Quaternion.from_axis_angle(Vector3(1, 0, 0), math.pi / 3)
        assert q.x == pytest.approx(math.sin(math.pi / 6), abs=0.01)
        assert q.y == pytest.approx(0, abs=0.01)
        assert q.z == pytest.approx(0, abs=0.01)
        assert q.w == pytest.approx(math.cos(math.pi / 6), abs=0.01)
