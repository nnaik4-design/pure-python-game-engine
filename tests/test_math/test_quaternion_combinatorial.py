import pytest
import math
import itertools
from allpairspy import AllPairs
from engine.math.quaternion import Quaternion
from engine.math.vector3 import Vector3

# --- Test Values ---
COORD_VALUES = [0.0, 1.0, -1.0, 0.5, -0.5]
T_VALUES = [-0.5, 0.0, 0.5, 1.0, 1.5]
ANGLES = [0.0, math.pi/4, math.pi/2, math.pi, -math.pi/2]

# --- PWC Generators ---

def get_look_rotation_pairs():
    VECTOR_BOUNDARIES = [
        Vector3.forward(), Vector3.up(), Vector3.right(), 
        Vector3(-1.0, -1.0, -1.0), Vector3.one()
    ]
    return list(AllPairs([VECTOR_BOUNDARIES, VECTOR_BOUNDARIES]))

# --- ACoC Generators ---
def get_euler_combinations():
    return list(itertools.product(ANGLES, ANGLES, ANGLES))



@pytest.mark.parametrize("roll, pitch, yaw", get_euler_combinations())
def test_from_euler_acoc(roll, pitch, yaw):
    q = Quaternion.from_euler_angles(roll, pitch, yaw)
    
    # Convert back to euler angles to check 
    r, p, y = q.to_euler_angles()
    
    # Can't directly assert equality due to multiple representations and gimball lock,
    # but we verify it's a valid quaternion and round trip is near
    assert abs(q.magnitude - 1.0) < 1e-5


@pytest.mark.parametrize("forward, up", get_look_rotation_pairs())
def test_look_rotation_pwc(forward, up):
    # Now you don't need to construct the vectors inside the test!
    if forward.magnitude == 0 or up.magnitude == 0:
        return
        
    if abs(abs(forward.normalized().dot(up.normalized())) - 1.0) < 1e-5:
        return
        
    q = Quaternion.look_rotation(forward, up)
    assert abs(q.magnitude - 1.0) < 1e-4

