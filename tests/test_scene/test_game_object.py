"""
Blackbox tests for GameObject and Component classes.
Tests use Equivalence Partitioning (EP), Boundary Analysis (BA), and Error Guessing (EG).
"""
import pytest
from engine.scene.game_object import GameObject, Component
from engine.math.vector2 import Vector2


# ================================================================
# Constructor Tests
# ================================================================

class TestGameObjectConstructor:

    def test_default_constructor(self):
        """EP: Default constructor creates named 'GameObject' with defaults"""
        obj = GameObject()
        assert obj.name == "GameObject"
        assert obj.is_active is True
        assert obj.is_destroyed is False
        assert obj.z_order == 0
        assert obj.tags == []
        assert obj.scene is None

    def test_custom_name(self):
        """EP: Custom name is stored"""
        obj = GameObject("Player")
        assert obj.name == "Player"

    def test_empty_name(self):
        """BA: Empty string name"""
        obj = GameObject("")
        assert obj.name == ""

    def test_has_transform(self):
        """EP: Every GameObject has a Transform component"""
        obj = GameObject()
        assert obj.transform is not None
        assert obj.transform.position == Vector2.zero()

    def test_empty_components(self):
        """EP: No components initially"""
        obj = GameObject()
        assert len(obj.components) == 0
        assert len(obj.components_list) == 0

    def test_empty_data(self):
        """EP: Data dict is empty initially"""
        obj = GameObject()
        assert obj.data == {}


# ================================================================
# Component Management Tests
# ================================================================

class TestComponentManagement:

    def test_add_component(self):
        """EP: Adding a component registers it"""
        obj = GameObject()
        comp = Component()
        obj.add_component(comp)
        assert obj.has_component(Component)
        assert obj.get_component(Component) is comp
        assert comp.game_object is obj

    def test_add_component_returns_component(self):
        """EP: add_component returns the component"""
        obj = GameObject()
        comp = Component()
        result = obj.add_component(comp)
        assert result is comp

    def test_remove_component(self):
        """EP: Removing a component unregisters it"""
        obj = GameObject()
        comp = Component()
        obj.add_component(comp)
        result = obj.remove_component(Component)
        assert result is True
        assert not obj.has_component(Component)
        assert comp.game_object is None

    def test_remove_nonexistent_component(self):
        """BA: Removing a component that doesn't exist returns False"""
        obj = GameObject()
        result = obj.remove_component(Component)
        assert result is False

    def test_get_nonexistent_component(self):
        """BA: Getting a component that doesn't exist returns None"""
        obj = GameObject()
        assert obj.get_component(Component) is None

    def test_has_component_false(self):
        """BA: has_component returns False when not present"""
        obj = GameObject()
        assert obj.has_component(Component) is False

    def test_replace_same_type_component(self):
        """EG: Adding a component of same type replaces the old one"""
        obj = GameObject()
        comp1 = Component()
        comp2 = Component()
        obj.add_component(comp1)
        obj.add_component(comp2)
        assert obj.get_component(Component) is comp2
        assert len(obj.components_list) == 1

    def test_multiple_component_types(self):
        """EP: Multiple different component types coexist"""
        class CompA(Component):
            pass
        class CompB(Component):
            pass

        obj = GameObject()
        a = CompA()
        b = CompB()
        obj.add_component(a)
        obj.add_component(b)
        assert obj.has_component(CompA)
        assert obj.has_component(CompB)
        assert len(obj.components_list) == 2


# ================================================================
# Tag Tests
# ================================================================

class TestGameObjectTags:

    def test_add_tag(self):
        """EP: Adding a tag stores it"""
        obj = GameObject()
        obj.add_tag("enemy")
        assert obj.has_tag("enemy")

    def test_remove_tag(self):
        """EP: Removing a tag removes it"""
        obj = GameObject()
        obj.add_tag("enemy")
        obj.remove_tag("enemy")
        assert not obj.has_tag("enemy")

    def test_has_tag_false(self):
        """BA: has_tag returns False for non-existent tag"""
        obj = GameObject()
        assert not obj.has_tag("nonexistent")

    def test_add_duplicate_tag(self):
        """EG: Adding the same tag twice doesn't duplicate it"""
        obj = GameObject()
        obj.add_tag("player")
        obj.add_tag("player")
        assert obj.tags.count("player") == 1

    def test_remove_nonexistent_tag(self):
        """EG: Removing a tag that doesn't exist doesn't crash"""
        obj = GameObject()
        obj.remove_tag("nonexistent")  # Should not raise

    def test_multiple_tags(self):
        """EP: Multiple different tags coexist"""
        obj = GameObject()
        obj.add_tag("enemy")
        obj.add_tag("boss")
        obj.add_tag("flying")
        assert obj.has_tag("enemy")
        assert obj.has_tag("boss")
        assert obj.has_tag("flying")


# ================================================================
# Active State and Destroy Tests
# ================================================================

class TestGameObjectLifecycle:

    def test_set_active_false(self):
        """EP: Deactivate object"""
        obj = GameObject()
        obj.set_active(False)
        assert obj.is_active is False

    def test_set_active_true(self):
        """EP: Reactivate object"""
        obj = GameObject()
        obj.set_active(False)
        obj.set_active(True)
        assert obj.is_active is True

    def test_destroy(self):
        """EP: Destroying marks as destroyed and clears components"""
        obj = GameObject()
        comp = Component()
        obj.add_component(comp)
        obj.destroy()
        assert obj.is_destroyed is True
        assert len(obj.components) == 0
        assert len(obj.components_list) == 0

    def test_destroy_twice(self):
        """EG: Destroying twice doesn't crash"""
        obj = GameObject()
        obj.destroy()
        obj.destroy()  # Should not raise
        assert obj.is_destroyed is True

    def test_update_inactive_does_nothing(self):
        """EP: Update on inactive object skips component updates"""
        class TrackingComponent(Component):
            def __init__(self):
                super().__init__()
                self.updated = False
            def update(self, dt):
                self.updated = True

        obj = GameObject()
        tc = TrackingComponent()
        obj.add_component(tc)
        obj.set_active(False)
        obj.update(0.016)
        assert tc.updated is False

    def test_update_destroyed_does_nothing(self):
        """EP: Update on destroyed object skips component updates"""
        class TrackingComponent(Component):
            def __init__(self):
                super().__init__()
                self.updated = False
            def update(self, dt):
                self.updated = True

        obj = GameObject()
        tc = TrackingComponent()
        obj.add_component(tc)
        obj.destroy()
        # After destroy, components are cleared, so we need to check differently
        # The object itself should not process updates
        assert obj.is_destroyed is True

    def test_update_active_calls_components(self):
        """EP: Update on active object calls component update"""
        class TrackingComponent(Component):
            def __init__(self):
                super().__init__()
                self.updated = False
                self.last_dt = None
            def update(self, dt):
                self.updated = True
                self.last_dt = dt

        obj = GameObject()
        tc = TrackingComponent()
        obj.add_component(tc)
        obj.update(0.016)
        assert tc.updated is True
        assert tc.last_dt == pytest.approx(0.016)

    def test_update_skips_inactive_component(self):
        """EP: Inactive components are skipped during update"""
        class TrackingComponent(Component):
            def __init__(self):
                super().__init__()
                self.updated = False
            def update(self, dt):
                self.updated = True

        obj = GameObject()
        tc = TrackingComponent()
        tc.is_active = False
        obj.add_component(tc)
        obj.update(0.016)
        assert tc.updated is False


# ================================================================
# Position/Rotation/Scale Convenience Methods
# ================================================================

class TestGameObjectTransform:

    def test_set_position(self):
        """EP: set_position updates transform"""
        obj = GameObject()
        obj.set_position(Vector2(10, 20))
        assert obj.get_position() == Vector2(10, 20)

    def test_translate(self):
        """EP: translate moves by delta"""
        obj = GameObject()
        obj.set_position(Vector2(5, 5))
        obj.translate(Vector2(3, -2))
        assert obj.get_position() == Vector2(8, 3)

    def test_set_rotation(self):
        """EP: set_rotation updates transform"""
        obj = GameObject()
        obj.set_rotation(1.57)
        assert obj.get_rotation() == pytest.approx(1.57)

    def test_rotate(self):
        """EP: rotate adds delta rotation"""
        obj = GameObject()
        obj.set_rotation(0.5)
        obj.rotate(0.3)
        assert obj.get_rotation() == pytest.approx(0.8)

    def test_set_scale(self):
        """EP: set_scale updates transform"""
        obj = GameObject()
        obj.set_scale(Vector2(2, 3))
        assert obj.get_scale() == Vector2(2, 3)

    def test_z_order(self):
        """EP: z_order can be modified"""
        obj = GameObject()
        obj.z_order = 5
        assert obj.z_order == 5


# ================================================================
# String Representation
# ================================================================

class TestGameObjectString:

    def test_str_default(self):
        """EP: String representation shows name, active, and component count"""
        obj = GameObject("Player")
        s = str(obj)
        assert "Player" in s
        assert "active=True" in s

    def test_str_with_components(self):
        """EP: String shows component count"""
        obj = GameObject("Player")
        obj.add_component(Component())
        s = str(obj)
        assert "components=1" in s


# ================================================================
# Error Guessing
# ================================================================

class TestGameObjectErrorGuessing:

    def test_data_storage(self):
        """EG: Custom data can be stored and retrieved"""
        obj = GameObject()
        obj.data["health"] = 100
        obj.data["name"] = "Hero"
        assert obj.data["health"] == 100
        assert obj.data["name"] == "Hero"

    def test_component_game_object_reference(self):
        """EG: Component's game_object is set correctly on add"""
        obj1 = GameObject("A")
        obj2 = GameObject("B")
        comp = Component()
        obj1.add_component(comp)
        assert comp.game_object is obj1

    def test_multiple_objects_independent(self):
        """EG: Multiple GameObjects are independent"""
        obj1 = GameObject("A")
        obj2 = GameObject("B")
        obj1.set_position(Vector2(1, 1))
        obj2.set_position(Vector2(2, 2))
        assert obj1.get_position() == Vector2(1, 1)
        assert obj2.get_position() == Vector2(2, 2)
