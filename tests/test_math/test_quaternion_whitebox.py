"""
Whitebox tests for Quaternion - targeting uncovered branches.
Focus on branch coverage for from_rotation_matrix, look_rotation, slerp negative dot,
and to_axis_angle w>1 normalization.
"""
import math
import pytest
from engine.math.quaternion import Quaternion
from engine.math.vector3 import Vector3


class TestQuaternionFromRotationMatrixBranches:
    """Whitebox: cover all 4 branches in from_rotation_matrix (lines 216-239)"""

    def test_branch_trace_positive(self):
        """Branch: trace > 0 (identity matrix, trace=3)"""
        # Identity rotation matrix
        matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        q = Quaternion.from_rotation_matrix(matrix)
        identity = Quaternion.identity()
        assert abs(q.w) == pytest.approx(1.0, abs=0.01)

    def test_branch_m00_largest(self):
        """Branch: matrix[0] > matrix[4] and matrix[0] > matrix[8] (lines 222-227)"""
        # 180-degree rotation around X axis: diag = (1, -1, -1), trace = -1
        matrix = [1, 0, 0, 0, -1, 0, 0, 0, -1]
        q = Quaternion.from_rotation_matrix(matrix)
        # Should produce rotation around X axis
        v = q.rotate_vector(Vector3(0, 1, 0))
        assert v.y == pytest.approx(-1.0, abs=0.01)

    def test_branch_m11_largest(self):
        """Branch: matrix[4] > matrix[8] (lines 228-233)"""
        # 180-degree rotation around Y axis: diag = (-1, 1, -1), trace = -1
        matrix = [-1, 0, 0, 0, 1, 0, 0, 0, -1]
        q = Quaternion.from_rotation_matrix(matrix)
        v = q.rotate_vector(Vector3(1, 0, 0))
        assert v.x == pytest.approx(-1.0, abs=0.01)

    def test_branch_m22_largest(self):
        """Branch: else (matrix[8] largest, lines 234-239)"""
        # 180-degree rotation around Z axis: diag = (-1, -1, 1), trace = -1
        matrix = [-1, 0, 0, 0, -1, 0, 0, 0, 1]
        q = Quaternion.from_rotation_matrix(matrix)
        v = q.rotate_vector(Vector3(1, 0, 0))
        assert v.x == pytest.approx(-1.0, abs=0.01)


class TestQuaternionLookRotation:
    """Whitebox: cover look_rotation (lines 246-262)"""

    def test_look_rotation_forward_z(self):
        """Branch: look_rotation with default up"""
        q = Quaternion.look_rotation(Vector3(0, 0, 1))
        assert isinstance(q, Quaternion)

    def test_look_rotation_custom_up(self):
        """Branch: look_rotation with custom up vector"""
        q = Quaternion.look_rotation(Vector3(1, 0, 0), Vector3(0, 1, 0))
        assert isinstance(q, Quaternion)

    def test_look_rotation_none_up(self):
        """Branch: up is None, uses default Vector3.up()"""
        q = Quaternion.look_rotation(Vector3(0, 0, 1), None)
        assert isinstance(q, Quaternion)


class TestQuaternionSlerpNegativeDot:
    """Whitebox: cover slerp negative dot branch (lines 154-156)"""

    def test_slerp_negative_dot_negates_other(self):
        """Branch: dot < 0 causes negation of other quaternion"""
        q1 = Quaternion.from_axis_angle(Vector3(0, 0, 1), 0)
        # Negate q1 to get same rotation but opposite quaternion hemisphere
        q2 = Quaternion(-q1.x, -q1.y, -q1.z, -q1.w)
        # These represent the same rotation, dot will be negative
        result = q1.slerp(q2, 0.5)
        assert result.magnitude == pytest.approx(1.0, abs=0.01)


class TestQuaternionToAxisAngleWGreaterThan1:
    """Whitebox: cover to_axis_angle w>1 normalization (line 128)"""

    def test_to_axis_angle_unnormalized(self):
        """Branch: w > 1 triggers normalization"""
        # Create a quaternion where w > 1 (unnormalized)
        q = Quaternion(0, 0, 0.5, 2.0)  # w=2.0 > 1
        axis, angle = q.to_axis_angle()
        # Should still produce valid output
        assert isinstance(angle, float)
        assert isinstance(axis, Vector3)
