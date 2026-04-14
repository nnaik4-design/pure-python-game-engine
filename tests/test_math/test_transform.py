"""
Blackbox tests for Transform using Equivalence Partitioning (EP),
Boundary Analysis (BA), and Error Guessing (EG).
"""
import math
import pytest
from engine.math.transform import Transform
from engine.math.vector2 import Vector2


# ============================================================
# Constructor Tests
# ============================================================

class TestTransformConstructor:

    def test_default_constructor(self):
        """EP: Default transform has zero position, zero rotation, unit scale"""
        t = Transform()
        assert t.position == Vector2.zero()
        assert t.rotation == 0.0
        assert t.scale == Vector2.one()

    def test_custom_position(self):
        """EP: Custom position"""
        t = Transform(position=Vector2(10, 20))
        assert t.position == Vector2(10, 20)

    def test_custom_rotation(self):
        """EP: Custom rotation"""
        t = Transform(rotation=math.pi / 4)
        assert t.rotation == math.pi / 4

    def test_custom_scale(self):
        """EP: Custom scale"""
        t = Transform(scale=Vector2(2, 3))
        assert t.scale == Vector2(2, 3)

    def test_negative_position(self):
        """EP: Negative position partition"""
        t = Transform(position=Vector2(-5, -10))
        assert t.position == Vector2(-5, -10)

    def test_zero_scale(self):
        """BA: Zero scale boundary"""
        t = Transform(scale=Vector2(0, 0))
        assert t.scale == Vector2(0, 0)

    def test_negative_scale(self):
        """EP: Negative scale (mirroring)"""
        t = Transform(scale=Vector2(-1, -1))
        assert t.scale == Vector2(-1, -1)


# ============================================================
# Parent-Child Hierarchy Tests
# ============================================================

class TestTransformHierarchy:

    def test_no_parent_by_default(self):
        """EP: Default transform has no parent"""
        t = Transform()
        assert t.parent is None

    def test_no_children_by_default(self):
        """EP: Default transform has no children"""
        t = Transform()
        assert t.children == []

    def test_set_parent(self):
        """EP: Setting parent registers child"""
        parent = Transform()
        child = Transform()
        child.parent = parent
        assert child.parent is parent
        assert child in parent.children

    def test_remove_parent(self):
        """EP: Setting parent to None removes from old parent"""
        parent = Transform()
        child = Transform()
        child.parent = parent
        child.parent = None
        assert child.parent is None
        assert child not in parent.children

    def test_reparent(self):
        """EP: Changing parent removes from old parent and adds to new"""
        parent1 = Transform()
        parent2 = Transform()
        child = Transform()
        child.parent = parent1
        child.parent = parent2
        assert child not in parent1.children
        assert child in parent2.children

    def test_multiple_children(self):
        """EP: Parent can have multiple children"""
        parent = Transform()
        c1 = Transform()
        c2 = Transform()
        c3 = Transform()
        c1.parent = parent
        c2.parent = parent
        c3.parent = parent
        assert len(parent.children) == 3

    def test_children_list_is_copy(self):
        """EG: children property returns a copy, not the internal list"""
        parent = Transform()
        child = Transform()
        child.parent = parent
        children_list = parent.children
        children_list.clear()
        assert len(parent.children) == 1  # Internal list unaffected


# ============================================================
# World Position Tests
# ============================================================

class TestTransformWorldPosition:

    def test_world_position_no_parent(self):
        """BA: Without parent, world position = local position"""
        t = Transform(position=Vector2(5, 10))
        assert t.world_position == Vector2(5, 10)

    def test_world_position_with_parent_offset(self):
        """EP: Child position adds to parent position"""
        parent = Transform(position=Vector2(10, 20))
        child = Transform(position=Vector2(5, 5))
        child.parent = parent
        result = child.world_position
        assert abs(result.x - 15) < 1e-6 and abs(result.y - 25) < 1e-6

    def test_world_position_with_parent_rotation(self):
        """EP: Parent rotation affects child world position"""
        parent = Transform(position=Vector2(0, 0), rotation=math.pi / 2)
        child = Transform(position=Vector2(1, 0))
        child.parent = parent
        result = child.world_position
        # Rotating (1,0) by 90 degrees gives (0,1)
        assert abs(result.x - 0) < 1e-6 and abs(result.y - 1) < 1e-6

    def test_world_position_with_parent_scale(self):
        """EP: Parent scale affects child world position"""
        parent = Transform(position=Vector2(0, 0), scale=Vector2(2, 3))
        child = Transform(position=Vector2(5, 10))
        child.parent = parent
        result = child.world_position
        assert abs(result.x - 10) < 1e-6 and abs(result.y - 30) < 1e-6

    def test_world_position_nested_hierarchy(self):
        """EP: Three-level hierarchy (grandparent -> parent -> child)"""
        grandparent = Transform(position=Vector2(10, 0))
        parent = Transform(position=Vector2(5, 0))
        child = Transform(position=Vector2(3, 0))
        parent.parent = grandparent
        child.parent = parent
        result = child.world_position
        assert abs(result.x - 18) < 1e-6


# ============================================================
# World Rotation Tests
# ============================================================

class TestTransformWorldRotation:

    def test_world_rotation_no_parent(self):
        """BA: Without parent, world rotation = local rotation"""
        t = Transform(rotation=1.5)
        assert abs(t.world_rotation - 1.5) < 1e-6

    def test_world_rotation_additive(self):
        """EP: Child rotation adds to parent rotation"""
        parent = Transform(rotation=math.pi / 4)
        child = Transform(rotation=math.pi / 4)
        child.parent = parent
        assert abs(child.world_rotation - math.pi / 2) < 1e-6

    def test_world_rotation_zero_parent(self):
        """BA: Parent with zero rotation doesn't affect child"""
        parent = Transform(rotation=0)
        child = Transform(rotation=math.pi / 3)
        child.parent = parent
        assert abs(child.world_rotation - math.pi / 3) < 1e-6


# ============================================================
# World Scale Tests
# ============================================================

class TestTransformWorldScale:

    def test_world_scale_no_parent(self):
        """BA: Without parent, world scale = local scale"""
        t = Transform(scale=Vector2(3, 4))
        assert t.world_scale == Vector2(3, 4)

    def test_world_scale_multiplicative(self):
        """EP: Scales multiply in hierarchy"""
        parent = Transform(scale=Vector2(2, 3))
        child = Transform(scale=Vector2(4, 5))
        child.parent = parent
        result = child.world_scale
        assert result == Vector2(8, 15)

    def test_world_scale_unit_parent(self):
        """BA: Unit parent scale doesn't affect child"""
        parent = Transform(scale=Vector2.one())
        child = Transform(scale=Vector2(3, 7))
        child.parent = parent
        assert child.world_scale == Vector2(3, 7)


# ============================================================
# Translation, Rotation, Scale Methods
# ============================================================

class TestTransformMethods:

    def test_translate(self):
        """EP: Translate moves position by delta"""
        t = Transform(position=Vector2(5, 10))
        t.translate(Vector2(3, -2))
        assert t.position == Vector2(8, 8)

    def test_translate_zero(self):
        """BA: Translating by zero doesn't change position"""
        t = Transform(position=Vector2(5, 10))
        t.translate(Vector2.zero())
        assert t.position == Vector2(5, 10)

    def test_rotate_method(self):
        """EP: Rotate adds to rotation"""
        t = Transform(rotation=math.pi / 4)
        t.rotate(math.pi / 4)
        assert abs(t.rotation - math.pi / 2) < 1e-6

    def test_rotate_by_zero(self):
        """BA: Rotating by zero doesn't change rotation"""
        t = Transform(rotation=1.0)
        t.rotate(0)
        assert abs(t.rotation - 1.0) < 1e-6

    def test_rotate_negative(self):
        """EP: Negative rotation (clockwise)"""
        t = Transform(rotation=math.pi)
        t.rotate(-math.pi / 2)
        assert abs(t.rotation - math.pi / 2) < 1e-6

    def test_scale_by(self):
        """EP: Scale by factor multiplies components"""
        t = Transform(scale=Vector2(2, 3))
        t.scale_by(Vector2(4, 5))
        assert t.scale == Vector2(8, 15)

    def test_scale_by_one(self):
        """BA: Scaling by (1,1) doesn't change scale"""
        t = Transform(scale=Vector2(3, 7))
        t.scale_by(Vector2.one())
        assert t.scale == Vector2(3, 7)

    def test_scale_by_zero(self):
        """BA: Scaling by zero zeroes the scale"""
        t = Transform(scale=Vector2(3, 7))
        t.scale_by(Vector2(0, 0))
        assert t.scale == Vector2(0, 0)


# ============================================================
# Look At Tests
# ============================================================

class TestTransformLookAt:

    def test_look_at_right(self):
        """EP: Looking at a point to the right -> rotation 0"""
        t = Transform(position=Vector2(0, 0))
        t.look_at(Vector2(10, 0))
        assert abs(t.rotation - 0) < 1e-6

    def test_look_at_up(self):
        """EP: Looking at a point above -> rotation -pi/2"""
        t = Transform(position=Vector2(0, 0))
        t.look_at(Vector2(0, -10))
        assert abs(t.rotation - (-math.pi / 2)) < 1e-6

    def test_look_at_left(self):
        """EP: Looking at point to the left -> rotation pi"""
        t = Transform(position=Vector2(0, 0))
        t.look_at(Vector2(-10, 0))
        assert abs(abs(t.rotation) - math.pi) < 1e-6


# ============================================================
# Forward and Right Direction Tests
# ============================================================

class TestTransformDirection:

    def test_forward_default(self):
        """BA: Default rotation (0) forward is (1, 0)"""
        t = Transform()
        fwd = t.forward()
        assert abs(fwd.x - 1) < 1e-6 and abs(fwd.y - 0) < 1e-6

    def test_forward_90_degrees(self):
        """EP: 90 degree rotation forward is (0, 1)"""
        t = Transform(rotation=math.pi / 2)
        fwd = t.forward()
        assert abs(fwd.x - 0) < 1e-6 and abs(fwd.y - 1) < 1e-6

    def test_right_default(self):
        """BA: Default right is (0, 1) (90 degrees from forward)"""
        t = Transform()
        r = t.right()
        assert abs(r.x - 0) < 1e-6 and abs(r.y - 1) < 1e-6


# ============================================================
# Transform Point Tests
# ============================================================

class TestTransformPoint:

    def test_transform_point_identity(self):
        """BA: Identity transform doesn't change point"""
        t = Transform()
        result = t.transform_point(Vector2(5, 10))
        assert abs(result.x - 5) < 1e-6 and abs(result.y - 10) < 1e-6

    def test_transform_point_translation_only(self):
        """EP: Translation offsets the point"""
        t = Transform(position=Vector2(10, 20))
        result = t.transform_point(Vector2(5, 5))
        assert abs(result.x - 15) < 1e-6 and abs(result.y - 25) < 1e-6

    def test_transform_point_scale_only(self):
        """EP: Scale multiplies the point"""
        t = Transform(scale=Vector2(2, 3))
        result = t.transform_point(Vector2(5, 10))
        assert abs(result.x - 10) < 1e-6 and abs(result.y - 30) < 1e-6

    def test_transform_point_rotation_only(self):
        """EP: Rotation rotates the point"""
        t = Transform(rotation=math.pi / 2)
        result = t.transform_point(Vector2(1, 0))
        assert abs(result.x - 0) < 1e-6 and abs(result.y - 1) < 1e-6

    def test_inverse_transform_point_roundtrip(self):
        """EG: transform_point then inverse_transform_point returns original"""
        t = Transform(position=Vector2(10, 20), rotation=0.5, scale=Vector2(2, 3))
        original = Vector2(5, 7)
        world = t.transform_point(original)
        local = t.inverse_transform_point(world)
        assert abs(local.x - original.x) < 1e-5
        assert abs(local.y - original.y) < 1e-5

    def test_transform_origin(self):
        """BA: Transforming the origin gives the transform's world position"""
        t = Transform(position=Vector2(10, 20))
        result = t.transform_point(Vector2.zero())
        assert abs(result.x - 10) < 1e-6 and abs(result.y - 20) < 1e-6


# ============================================================
# 3D Mode Tests
# ============================================================

class TestTransform3D:

    def test_3d_disabled_by_default(self):
        """EP: 3D mode is disabled by default"""
        t = Transform()
        assert t.quaternion_rotation is None

    def test_enable_3d(self):
        """EP: Enabling 3D sets identity quaternion"""
        t = Transform()
        t.enable_3d()
        q = t.quaternion_rotation
        assert q is not None
        assert abs(q.w - 1.0) < 1e-6

    def test_disable_3d(self):
        """EP: Disabling 3D clears quaternion"""
        t = Transform()
        t.enable_3d()
        t.disable_3d()
        assert t.quaternion_rotation is None

    def test_set_quaternion_enables_3d(self):
        """EG: Setting quaternion_rotation auto-enables 3D mode"""
        from engine.math.quaternion import Quaternion
        t = Transform()
        t.quaternion_rotation = Quaternion.identity()
        assert t.quaternion_rotation is not None

    def test_set_rotation_from_quaternion(self):
        """EP: Setting 2D rotation from quaternion uses yaw"""
        from engine.math.quaternion import Quaternion
        from engine.math.vector3 import Vector3
        q = Quaternion.from_axis_angle(Vector3(0, 0, 1), math.pi / 4)
        t = Transform()
        t.set_rotation_from_quaternion(q)
        assert abs(t.rotation - math.pi / 4) < 1e-5

    def test_set_rotation_from_none_quaternion(self):
        """BA: Passing None quaternion doesn't crash"""
        t = Transform()
        t.set_rotation_from_quaternion(None)  # Should not raise

    def test_get_quaternion_from_rotation(self):
        """EP: Get quaternion from 2D rotation"""
        t = Transform(rotation=math.pi / 4)
        q = t.get_quaternion_from_rotation()
        assert abs(q.magnitude - 1.0) < 1e-5


# ============================================================
# String Representation
# ============================================================

class TestTransformString:

    def test_str(self):
        """EP: String contains position, rotation, scale info"""
        t = Transform(position=Vector2(1, 2), rotation=math.pi / 2, scale=Vector2(3, 4))
        s = str(t)
        assert "Transform" in s
        assert "90.0" in s  # 90 degrees


# ============================================================
# Error Guessing
# ============================================================

class TestTransformErrorGuessing:

    def test_translate_multiple_times(self):
        """EG: Multiple translations accumulate"""
        t = Transform(position=Vector2(0, 0))
        t.translate(Vector2(1, 0))
        t.translate(Vector2(0, 1))
        t.translate(Vector2(1, 1))
        assert t.position == Vector2(2, 2)

    def test_world_position_is_copy(self):
        """EG: Modifying world_position doesn't affect transform"""
        t = Transform(position=Vector2(5, 10))
        wp = t.world_position
        wp.x = 999
        assert t.position == Vector2(5, 10)

    def test_deeply_nested_hierarchy(self):
        """EG: 5-level deep hierarchy computes correctly"""
        transforms = [Transform(position=Vector2(1, 0)) for _ in range(5)]
        for i in range(1, 5):
            transforms[i].parent = transforms[i - 1]
        # Each adds 1 to x: total world x = 5
        result = transforms[4].world_position
        assert abs(result.x - 5) < 1e-6
