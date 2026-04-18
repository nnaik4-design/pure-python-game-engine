"""
Combinatorial testing for the Scene module.
Applies Each Choice Coverage (ECC) to systematically test the update lifecycle hierarchy.
"""
import pytest
from engine.scene.scene import Scene
from engine.scene.game_object import GameObject, Component

class MockUpdateComponent(Component):
    """Helper component to track if update was executed."""
    def __init__(self):
        super().__init__()
        self.was_updated = False

    def update(self, delta_time: float):
        self.was_updated = True

# Partitions identified (Building blocks for ECC):
# scene_active: [True (Active), False (Inactive)]
# obj_active: [True (Active), False (Inactive)]
# obj_destroyed: [False (Alive), True (Destroyed)]
# comp_active: [True (Active), False (Inactive)]

@pytest.mark.parametrize(
    "scene_active, obj_active, obj_destroyed, comp_active, expected_update_call",
    [
        # Combination 1: Covers Scene-True, Obj-True, Destroyed-False, Comp-True
        # This is the "Happy Path" where everything is valid.
        (True, True, False, True, True),
        
        # Combination 2: Covers Scene-False. Re-uses Obj-True, Destroyed-False, Comp-True
        # Fails at the Scene level.
        (False, True, False, True, False),
        
        # Combination 3: Covers Obj-False. Re-uses Scene-True, Destroyed-False, Comp-True
        # Fails at the GameObject level.
        (True, False, False, True, False),
        
        # Combination 4: Covers Destroyed-True, Comp-False. Re-uses Scene-True, Obj-True
        # Fails at the Component/Destroyed level.
        (True, True, True, False, False),
    ]
)
def test_scene_update_hierarchy_ecc(scene_active, obj_active, obj_destroyed, comp_active, expected_update_call):
    """
    Combinatorial testing: Each Choice Coverage (ECC) for the Scene update loop.
    Ensures every state partition of the hierarchy is executed at least once across 4 minimal test cases.
    """
    # 1. Setup hierarchy
    scene = Scene()
    obj = GameObject("EccObject")
    comp = MockUpdateComponent()
    
    obj.add_component(comp)
    scene.add_object(obj)
    
    # 2. Apply combinatorial state
    scene.set_active(scene_active)
    obj.set_active(obj_active)
    comp.is_active = comp_active
    
    if obj_destroyed:
        obj.destroy()
        
    # 3. Execute method under test
    scene.update(0.016)
    
    # 4. Validate output
    assert comp.was_updated == expected_update_call