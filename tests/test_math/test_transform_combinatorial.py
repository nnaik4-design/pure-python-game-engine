import pytest
import math
from allpairspy import AllPairs
from engine.math.transform import Transform
from engine.math.vector2 import Vector2


# ============================================================================
# 1. Transform Point Transformation - PWC
# ============================================================================
_transform_parameters = [
    [0.0, 5.0, -5.0],           # tx (position x)
    [0.0, 5.0, -5.0],           # ty (position y)
    [0.0, math.pi/4, math.pi/2],  # rotation
    [0.5, 1.0, 2.0],            # sx (scale x)
    [0.5, 1.0, 2.0],            # sy (scale y)
    [-5.0, 0.0, 5.0],           # px (point x)
    [-5.0, 0.0, 5.0],           # py (point y)
]
_transform_pwc_cases = [tuple(case) for case in AllPairs(_transform_parameters)]

@pytest.mark.parametrize("tx, ty, rot, sx, sy, px, py", _transform_pwc_cases)
def test_transform_point_pwc(tx, ty, rot, sx, sy, px, py):
    """Test local to world point transformation with round-trip validation"""
    transform = Transform(position=Vector2(tx, ty), rotation=rot, scale=Vector2(sx, sy))
    point = Vector2(px, py)

    # Transform local point to world
    world_point = transform.transform_point(point)

    # Transform world point back to local
    local_point = transform.inverse_transform_point(world_point)

    # Round trip should return to original point
    assert abs(local_point.x - point.x) < 1e-4, f"X mismatch: {local_point.x} vs {point.x}"
    assert abs(local_point.y - point.y) < 1e-4, f"Y mismatch: {local_point.y} vs {point.y}"


# ============================================================================
# 2. Transform Scale Impact - PWC
# ============================================================================
@pytest.mark.parametrize("tx, ty, rot, sx, sy, px, py", _transform_pwc_cases)
def test_transform_scale_impact_pwc(tx, ty, rot, sx, sy, px, py):
    """Test that scaling affects point transformation correctly"""
    transform = Transform(position=Vector2(tx, ty), rotation=rot, scale=Vector2(sx, sy))
    point = Vector2(px, py)

    # Transform the point
    world_point = transform.transform_point(point)

    # If point is zero, world point should be at transform position
    if px == 0.0 and py == 0.0:
        assert abs(world_point.x - tx) < 1e-10
        assert abs(world_point.y - ty) < 1e-10

    # For non-zero points, distance from transform position should reflect scale
    # The scaling should proportionally affect the distance
    if not (px == 0.0 and py == 0.0):
        distance_local = math.sqrt(px*px + py*py)
        distance_world = (world_point - Vector2(tx, ty)).magnitude

        # For uniform scaling, distance should scale uniformly
        # For non-uniform scaling, the effect is more complex (component-wise)
        if abs(abs(sx) - abs(sy)) < 1e-6:  # Uniform scaling
            # Account for scaling: scaled distance ≈ local distance * scale
            avg_scale = (abs(sx) + abs(sy)) / 2
            if px != 0 or py != 0:
                expected_scaled_mag = distance_local * avg_scale
                # Rotation doesn't change magnitude, but scale does
                assert abs(distance_world - expected_scaled_mag) < distance_world * 0.5 or abs(sx) < 1e-6


# ============================================================================
# 3. Transform Negative Scale - Edge Case
# ============================================================================
class TestTransformNegativeScale:
    """Test negative scale behavior (reflection)"""

    def test_negative_scale_x(self):
        """Test reflection on X axis"""
        transform = Transform(position=Vector2(0, 0), rotation=0, scale=Vector2(-1.0, 1.0))
        point = Vector2(1.0, 0.0)

        world_point = transform.transform_point(point)
        # X should be negated (reflected)
        assert abs(world_point.x - (-1.0)) < 1e-6
        assert abs(world_point.y - 0.0) < 1e-6

    def test_negative_scale_y(self):
        """Test reflection on Y axis"""
        transform = Transform(position=Vector2(0, 0), rotation=0, scale=Vector2(1.0, -1.0))
        point = Vector2(0.0, 1.0)

        world_point = transform.transform_point(point)
        assert abs(world_point.x - 0.0) < 1e-6
        assert abs(world_point.y - (-1.0)) < 1e-6

    def test_negative_both_scales(self):
        """Test reflection on both axes (180° rotation via scale)"""
        transform = Transform(position=Vector2(0, 0), rotation=0, scale=Vector2(-1.0, -1.0))
        point = Vector2(1.0, 2.0)

        world_point = transform.transform_point(point)
        assert abs(world_point.x - (-1.0)) < 1e-6
        assert abs(world_point.y - (-2.0)) < 1e-6


# ============================================================================
# 4. Transform Rotation - PWC
# ============================================================================
_rotation_test_cases = [
    (0.0, Vector2(1.0, 0.0), Vector2(1.0, 0.0)),     # No rotation
    (math.pi/2, Vector2(1.0, 0.0), Vector2(0.0, 1.0)),  # 90° rotation
    (math.pi, Vector2(1.0, 0.0), Vector2(-1.0, 0.0)), # 180° rotation
    (-math.pi/2, Vector2(1.0, 0.0), Vector2(0.0, -1.0)), # -90° rotation
]

class TestTransformRotation:
    """Test rotation transformations"""

    @pytest.mark.parametrize("angle, point, expected", _rotation_test_cases)
    def test_rotation_correctness(self, angle, point, expected):
        """Test that rotations produce correct results"""
        transform = Transform(position=Vector2(0, 0), rotation=angle, scale=Vector2(1.0, 1.0))
        result = transform.transform_point(point)

        assert abs(result.x - expected.x) < 1e-5, f"X mismatch for angle {angle}"
        assert abs(result.y - expected.y) < 1e-5, f"Y mismatch for angle {angle}"

    def test_rotation_accumulation(self):
        """Test that sequential rotations accumulate"""
        angle1 = math.pi / 4
        angle2 = math.pi / 4

        transform = Transform(position=Vector2(0, 0), rotation=angle1 + angle2, scale=Vector2(1.0, 1.0))
        point = Vector2(1.0, 0.0)

        result = transform.transform_point(point)

        # Should rotate 90° total
        expected_x = 0.0
        expected_y = 1.0
        assert abs(result.x - expected_x) < 1e-5
        assert abs(result.y - expected_y) < 1e-5


# ============================================================================
# 5. Transform Hierarchy - Parent/Child - PWC
# ============================================================================
PARENT_STATES = [
    {"pos": (0, 0), "rot": 0, "scale": (1, 1)},
    {"pos": (10, 5), "rot": 0, "scale": (1, 1)},
    {"pos": (0, 0), "rot": math.pi/2, "scale": (1, 1)},
    {"pos": (0, 0), "rot": 0, "scale": (2, 2)},
]

CHILD_STATES = [
    {"pos": (0, 0), "rot": 0, "scale": (1, 1)},
    {"pos": (5, 0), "rot": 0, "scale": (1, 1)},
    {"pos": (0, 0), "rot": math.pi/2, "scale": (1, 1)},
    {"pos": (0, 0), "rot": 0, "scale": (2, 2)},
]

_hierarchy_pwc_cases = [(p_idx, c_idx) for p_idx, c_idx in AllPairs([range(len(PARENT_STATES)), range(len(CHILD_STATES))])]

@pytest.mark.parametrize("p_idx, c_idx", _hierarchy_pwc_cases)
def test_hierarchy_world_position_pwc(p_idx, c_idx):
    """Test that child world position correctly incorporates parent transform"""
    p_state = PARENT_STATES[p_idx]
    c_state = CHILD_STATES[c_idx]

    parent = Transform(
        position=Vector2(*p_state["pos"]),
        rotation=p_state["rot"],
        scale=Vector2(*p_state["scale"])
    )

    child = Transform(
        position=Vector2(*c_state["pos"]),
        rotation=c_state["rot"],
        scale=Vector2(*c_state["scale"])
    )
    child.parent = parent

    # Child's world position should account for parent transform
    world_pos = child.world_position

    # Verify it's different from local position (unless parent is identity)
    local_pos = child.position
    if not (p_state["pos"] == (0, 0) and p_state["rot"] == 0 and p_state["scale"] == (1, 1)):
        # Non-identity parent should affect world position
        # Exception: when child is at origin (0,0), rotation/scale don't move it
        # Only parent's position affects it
        if c_state["pos"] == (0, 0):
            # Child at origin: world position should be parent's position
            assert abs(world_pos.x - p_state["pos"][0]) < 1e-10
            assert abs(world_pos.y - p_state["pos"][1]) < 1e-10
        else:
            # Child not at origin: non-identity parent should change world position
            assert abs(world_pos.x - local_pos.x) > 1e-10 or abs(world_pos.y - local_pos.y) > 1e-10


@pytest.mark.parametrize("p_idx, c_idx", _hierarchy_pwc_cases)
def test_hierarchy_world_rotation_pwc(p_idx, c_idx):
    """Test that child world rotation accumulates with parent"""
    p_state = PARENT_STATES[p_idx]
    c_state = CHILD_STATES[c_idx]

    parent = Transform(
        position=Vector2(*p_state["pos"]),
        rotation=p_state["rot"],
        scale=Vector2(*p_state["scale"])
    )

    child = Transform(
        position=Vector2(*c_state["pos"]),
        rotation=c_state["rot"],
        scale=Vector2(*c_state["scale"])
    )
    child.parent = parent

    # World rotation should be parent rotation + child rotation
    world_rot = child.world_rotation
    expected_rot = p_state["rot"] + c_state["rot"]

    # Normalize angles to compare
    world_rot_normalized = math.atan2(math.sin(world_rot), math.cos(world_rot))
    expected_rot_normalized = math.atan2(math.sin(expected_rot), math.cos(expected_rot))

    assert abs(world_rot_normalized - expected_rot_normalized) < 1e-5


@pytest.mark.parametrize("p_idx, c_idx", _hierarchy_pwc_cases)
def test_hierarchy_world_scale_pwc(p_idx, c_idx):
    """Test that child world scale multiplies with parent scale"""
    p_state = PARENT_STATES[p_idx]
    c_state = CHILD_STATES[c_idx]

    parent = Transform(
        position=Vector2(*p_state["pos"]),
        rotation=p_state["rot"],
        scale=Vector2(*p_state["scale"])
    )

    child = Transform(
        position=Vector2(*c_state["pos"]),
        rotation=c_state["rot"],
        scale=Vector2(*c_state["scale"])
    )
    child.parent = parent

    # World scale should be parent scale * child scale
    world_scale = child.world_scale
    expected_x = p_state["scale"][0] * c_state["scale"][0]
    expected_y = p_state["scale"][1] * c_state["scale"][1]

    assert abs(world_scale.x - expected_x) < 1e-5
    assert abs(world_scale.y - expected_y) < 1e-5


# ============================================================================
# 6. Transform Methods - Translation, Rotation, Scaling
# ============================================================================
class TestTransformMethods:
    """Test Transform manipulation methods"""

    def test_translate(self):
        """Test translation method"""
        transform = Transform(position=Vector2(5, 5), rotation=0, scale=Vector2(1, 1))
        transform.translate(Vector2(3, -2))

        assert transform.position.x == 8.0
        assert transform.position.y == 3.0

    def test_rotate(self):
        """Test rotation method"""
        transform = Transform(position=Vector2(0, 0), rotation=0, scale=Vector2(1, 1))
        transform.rotate(math.pi / 2)

        assert abs(transform.rotation - math.pi/2) < 1e-10

    def test_scale_by(self):
        """Test scaling method"""
        transform = Transform(position=Vector2(0, 0), rotation=0, scale=Vector2(2, 3))
        transform.scale_by(Vector2(0.5, 2))

        assert abs(transform.scale.x - 1.0) < 1e-10
        assert abs(transform.scale.y - 6.0) < 1e-10

    def test_look_at(self):
        """Test look_at method"""
        transform = Transform(position=Vector2(0, 0), rotation=0, scale=Vector2(1, 1))
        transform.look_at(Vector2(1, 0))

        # Should look at right (angle 0)
        assert abs(transform.rotation - 0.0) < 1e-5

        transform.look_at(Vector2(0, 1))
        # Should look down (angle π/2)
        assert abs(transform.rotation - math.pi/2) < 1e-5

    def test_forward_direction(self):
        """Test forward direction method"""
        transform = Transform(position=Vector2(0, 0), rotation=0, scale=Vector2(1, 1))
        forward = transform.forward()

        # Should point right
        assert abs(forward.x - 1.0) < 1e-5
        assert abs(forward.y - 0.0) < 1e-5

        transform.rotation = math.pi / 2
        forward = transform.forward()

        # Should point down
        assert abs(forward.x - 0.0) < 1e-5
        assert abs(forward.y - 1.0) < 1e-5

    def test_right_direction(self):
        """Test right direction method"""
        transform = Transform(position=Vector2(0, 0), rotation=0, scale=Vector2(1, 1))
        right = transform.right()

        # Should point down (π/2 from right)
        assert abs(right.x - 0.0) < 1e-5
        assert abs(right.y - 1.0) < 1e-5


# ============================================================================
# 7. Multi-Level Hierarchy
# ============================================================================
class TestMultiLevelHierarchy:
    """Test transformations with multiple hierarchy levels"""

    def test_three_level_hierarchy(self):
        """Test transform with grandparent, parent, and child"""
        grandparent = Transform(position=Vector2(10, 0), rotation=0, scale=Vector2(2, 1))
        parent = Transform(position=Vector2(5, 0), rotation=0, scale=Vector2(1, 1))
        child = Transform(position=Vector2(1, 0), rotation=0, scale=Vector2(1, 1))

        parent.parent = grandparent
        child.parent = parent

        # Child world position should be: 10 + (5 * 2) + (1 * 2 * 1) = 10 + 10 + 2 = 22
        world_pos = child.world_position
        expected_x = 10 + 5 * 2 + 1 * 2
        assert abs(world_pos.x - expected_x) < 1e-5

    def test_reparenting(self):
        """Test changing parent"""
        parent1 = Transform(position=Vector2(5, 0), rotation=0, scale=Vector2(1, 1))
        parent2 = Transform(position=Vector2(10, 0), rotation=0, scale=Vector2(1, 1))
        child = Transform(position=Vector2(1, 0), rotation=0, scale=Vector2(1, 1))

        child.parent = parent1
        world_pos_1 = child.world_position

        child.parent = parent2
        world_pos_2 = child.world_position

        # World positions should be different
        assert abs(world_pos_1.x - world_pos_2.x) > 1e-10

    def test_hierarchy_children_list(self):
        """Test that parent tracks children"""
        parent = Transform(position=Vector2(0, 0))
        child1 = Transform(position=Vector2(1, 0))
        child2 = Transform(position=Vector2(2, 0))

        child1.parent = parent
        child2.parent = parent

        # Parent should have both children
        assert len(parent.children) == 2
        assert child1 in parent.children
        assert child2 in parent.children

        # Removing a child
        child1.parent = None
        assert len(parent.children) == 1
        assert child1 not in parent.children


# ============================================================================
# 8. Edge Cases
# ============================================================================
class TestTransformEdgeCases:
    """Test edge cases for transforms"""

    def test_zero_scale(self):
        """Test transform with zero scale"""
        transform = Transform(position=Vector2(5, 5), rotation=0, scale=Vector2(0, 0))
        point = Vector2(10, 10)

        world_point = transform.transform_point(point)

        # Zero scale should collapse to position
        assert abs(world_point.x - 5.0) < 1e-6
        assert abs(world_point.y - 5.0) < 1e-6

    def test_very_small_scale(self):
        """Test with very small scale"""
        transform = Transform(position=Vector2(0, 0), rotation=0, scale=Vector2(1e-10, 1e-10))
        point = Vector2(1, 1)

        world_point = transform.transform_point(point)
        local_point = transform.inverse_transform_point(world_point)

        # Round trip should still work
        assert abs(local_point.x - 1.0) < 1e-5
        assert abs(local_point.y - 1.0) < 1e-5

    def test_large_rotation_wrapping(self):
        """Test that large rotations wrap correctly"""
        transform1 = Transform(position=Vector2(0, 0), rotation=math.pi)
        transform2 = Transform(position=Vector2(0, 0), rotation=-math.pi)

        point = Vector2(1, 0)
        result1 = transform1.transform_point(point)
        result2 = transform2.transform_point(point)

        # Both should produce same result (π and -π represent same rotation)
        assert abs(result1.x - result2.x) < 1e-5
        assert abs(result1.y - result2.y) < 1e-5

    def test_no_parent(self):
        """Test that transforms work without parent"""
        transform = Transform(position=Vector2(5, 5), rotation=math.pi/4, scale=Vector2(2, 2))

        # Should not crash when accessing world properties
        assert transform.world_position is not None
        assert transform.world_rotation is not None
        assert transform.world_scale is not None

        # Without parent, world properties = local properties
        assert abs(transform.world_position.x - transform.position.x) < 1e-10
        assert abs(transform.world_rotation - transform.rotation) < 1e-10
