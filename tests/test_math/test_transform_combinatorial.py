import pytest
import math
import itertools
from allpairspy import AllPairs
from engine.math.transform import Transform
from engine.math.vector2 import Vector2

# --- Test Values ---
COORD_VALUES = [0.0, 1.0, -1.0, 5.0, -5.0]
ANGLES = [0.0, math.pi/4, math.pi/2, math.pi]
SCALES = [1.0, 2.0, -1.0, 0.5]


def get_transform_point_pairs():
    # tx, ty, rot, sx, sy, px, py 
    # (7 parameters -> tx, ty, r, sx, sy for transform; px, py for point)
    return list(AllPairs([
        COORD_VALUES, COORD_VALUES, ANGLES, SCALES, SCALES, COORD_VALUES, COORD_VALUES
    ]))

@pytest.mark.parametrize("tx, ty, rot, sx, sy, px, py", get_transform_point_pairs())
def test_transform_point_pwc(tx, ty, rot, sx, sy, px, py):
    transform = Transform(position=Vector2(tx, ty), rotation=rot, scale=Vector2(sx, sy))
    point = Vector2(px, py)
    
    world_point = transform.transform_point(point)
    local_point = transform.inverse_transform_point(world_point)
    
    # Validation: round trip
    assert abs(local_point.x - point.x) < 1e-4
    assert abs(local_point.y - point.y) < 1e-4


# ECC values for parent-child relationship (Each Choice Coverage)
# Options: Translate, Rotate, Scale
PARENT_STATES = [
    {"pos": (0, 0), "rot": 0, "scale": (1, 1)},
    {"pos": (10, 5), "rot": 0, "scale": (1, 1)},
    {"pos": (0, 0), "rot": math.pi/2, "scale": (1, 1)},
    {"pos": (0, 0), "rot": 0, "scale": (2, 2)}
]

CHILD_STATES = [
    {"pos": (0, 0), "rot": 0, "scale": (1, 1)},
    {"pos": (10, 5), "rot": 0, "scale": (1, 1)},
    {"pos": (0, 0), "rot": math.pi/2, "scale": (1, 1)},
    {"pos": (0, 0), "rot": 0, "scale": (2, 2)}
]

def get_ecc_hierarchy():
    # Just generating pairs of parent type and child type 
    # Since they are dictionaries, we return index pairs
    return list(AllPairs([range(len(PARENT_STATES)), range(len(CHILD_STATES))]))

@pytest.mark.parametrize("p_idx, c_idx", get_ecc_hierarchy())
def test_hierarchy_ecc(p_idx, c_idx):
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
    
    # Simply verify the hierarchy logic doesn't crash and returns valid properties
    w_pos = child.world_position
    w_rot = child.world_rotation
    w_scale = child.world_scale
    
    assert isinstance(w_pos, Vector2)
    assert isinstance(w_rot, float)
    assert isinstance(w_scale, Vector2)

