"""
Blackbox tests for Scene class.
Tests use Equivalence Partitioning (EP), Boundary Analysis (BA), and Error Guessing (EG).
"""
import pytest
from engine.scene.scene import Scene
from engine.scene.game_object import GameObject, Component
from engine.math.vector2 import Vector2


# ================================================================
# Constructor Tests
# ================================================================

class TestSceneConstructor:

    def test_default_constructor(self):
        """EP: Default scene has name and empty collections"""
        scene = Scene()
        assert scene.name == "Untitled Scene"
        assert scene.is_active is True
        assert len(scene.game_objects) == 0
        assert scene.get_object_count() == 0

    def test_custom_name(self):
        """EP: Custom name is stored"""
        scene = Scene("Level 1")
        assert scene.name == "Level 1"

    def test_empty_data(self):
        """EP: Data dict starts empty"""
        scene = Scene()
        assert scene.data == {}


# ================================================================
# Add/Remove Object Tests
# ================================================================

class TestSceneObjectManagement:

    def test_add_object(self):
        """EP: Adding an object stores it and sets scene reference"""
        scene = Scene()
        obj = GameObject("Player")
        scene.add_object(obj)
        assert scene.get_object_count() == 1
        assert obj.scene is scene

    def test_add_multiple_objects(self):
        """EP: Multiple objects can be added"""
        scene = Scene()
        for i in range(5):
            scene.add_object(GameObject(f"Obj{i}"))
        assert scene.get_object_count() == 5

    def test_add_duplicate_object_ignored(self):
        """EG: Adding same object twice doesn't duplicate"""
        scene = Scene()
        obj = GameObject("Player")
        scene.add_object(obj)
        scene.add_object(obj)
        assert scene.get_object_count() == 1

    def test_remove_object(self):
        """EP: Removing an object removes it and clears scene reference"""
        scene = Scene()
        obj = GameObject("Player")
        scene.add_object(obj)
        scene.remove_object(obj)
        assert scene.get_object_count() == 0
        assert obj.scene is None

    def test_remove_nonexistent_object(self):
        """EG: Removing object not in scene doesn't crash"""
        scene = Scene()
        obj = GameObject("Player")
        scene.remove_object(obj)  # Should not raise


# ================================================================
# Find Object Tests
# ================================================================

class TestSceneFindObjects:

    def test_find_object_by_name(self):
        """EP: Find object by name"""
        scene = Scene()
        obj = GameObject("Player")
        scene.add_object(obj)
        found = scene.find_object("Player")
        assert found is obj

    def test_find_nonexistent_name(self):
        """BA: Finding nonexistent name returns None"""
        scene = Scene()
        assert scene.find_object("Ghost") is None

    def test_find_objects_with_tag(self):
        """EP: Find all objects with a tag"""
        scene = Scene()
        e1 = GameObject("Enemy1")
        e1.add_tag("enemy")
        e2 = GameObject("Enemy2")
        e2.add_tag("enemy")
        p = GameObject("Player")
        p.add_tag("player")
        scene.add_object(e1)
        scene.add_object(e2)
        scene.add_object(p)
        enemies = scene.find_objects_with_tag("enemy")
        assert len(enemies) == 2
        assert e1 in enemies
        assert e2 in enemies

    def test_find_objects_with_nonexistent_tag(self):
        """BA: Finding with nonexistent tag returns empty list"""
        scene = Scene()
        result = scene.find_objects_with_tag("nonexistent")
        assert result == []

    def test_find_objects_of_type(self):
        """EP: Find objects of a specific type"""
        class EnemyObject(GameObject):
            pass

        scene = Scene()
        enemy = EnemyObject("Enemy")
        player = GameObject("Player")
        scene.add_object(enemy)
        scene.add_object(player)
        found = scene.find_objects_of_type(EnemyObject)
        assert len(found) == 1
        assert found[0] is enemy

    def test_find_objects_returns_copy(self):
        """EG: find_objects_with_tag returns a copy, not the internal list"""
        scene = Scene()
        obj = GameObject("Obj")
        obj.add_tag("test")
        scene.add_object(obj)
        result = scene.find_objects_with_tag("test")
        result.clear()  # Modifying returned list
        # Internal list should still have the object
        assert len(scene.find_objects_with_tag("test")) == 1


# ================================================================
# Update and Lifecycle Tests
# ================================================================

class TestSceneUpdate:

    def test_update_calls_object_updates(self):
        """EP: Scene update calls update on active objects"""
        class TrackingComponent(Component):
            def __init__(self):
                super().__init__()
                self.updated = False
            def update(self, dt):
                self.updated = True

        scene = Scene()
        obj = GameObject("Obj")
        tc = TrackingComponent()
        obj.add_component(tc)
        scene.add_object(obj)
        scene.update(0.016)
        assert tc.updated is True

    def test_update_inactive_scene_does_nothing(self):
        """EP: Inactive scene doesn't update objects"""
        class TrackingComponent(Component):
            def __init__(self):
                super().__init__()
                self.updated = False
            def update(self, dt):
                self.updated = True

        scene = Scene()
        obj = GameObject("Obj")
        tc = TrackingComponent()
        obj.add_component(tc)
        scene.add_object(obj)
        scene.set_active(False)
        scene.update(0.016)
        assert tc.updated is False

    def test_update_skips_inactive_objects(self):
        """EP: Inactive objects are skipped during scene update"""
        class TrackingComponent(Component):
            def __init__(self):
                super().__init__()
                self.updated = False
            def update(self, dt):
                self.updated = True

        scene = Scene()
        obj = GameObject("Obj")
        tc = TrackingComponent()
        obj.add_component(tc)
        obj.set_active(False)
        scene.add_object(obj)
        scene.update(0.016)
        assert tc.updated is False

    def test_cleanup_destroyed_objects(self):
        """EP: Destroyed objects are removed after update"""
        scene = Scene()
        obj = GameObject("Obj")
        scene.add_object(obj)
        obj.destroy()
        scene.update(0.016)
        assert scene.get_object_count() == 0

    def test_cleanup_all(self):
        """EP: cleanup destroys all objects"""
        scene = Scene()
        for i in range(3):
            scene.add_object(GameObject(f"Obj{i}"))
        scene.cleanup()
        assert scene.get_object_count() == 0


# ================================================================
# Active Object Count Tests
# ================================================================

class TestSceneObjectCount:

    def test_active_object_count(self):
        """EP: Count only active objects"""
        scene = Scene()
        obj1 = GameObject("A")
        obj2 = GameObject("B")
        obj3 = GameObject("C")
        obj2.set_active(False)
        scene.add_object(obj1)
        scene.add_object(obj2)
        scene.add_object(obj3)
        assert scene.get_active_object_count() == 2
        assert scene.get_object_count() == 3

    def test_active_count_all_inactive(self):
        """BA: All inactive returns 0"""
        scene = Scene()
        obj = GameObject("A")
        obj.set_active(False)
        scene.add_object(obj)
        assert scene.get_active_object_count() == 0

    def test_active_count_empty_scene(self):
        """BA: Empty scene returns 0"""
        scene = Scene()
        assert scene.get_active_object_count() == 0


# ================================================================
# Scene Set Active Tests
# ================================================================

class TestSceneActive:

    def test_set_active_false(self):
        """EP: Deactivate scene"""
        scene = Scene()
        scene.set_active(False)
        assert scene.is_active is False

    def test_set_active_true(self):
        """EP: Reactivate scene"""
        scene = Scene()
        scene.set_active(False)
        scene.set_active(True)
        assert scene.is_active is True


# ================================================================
# Error Guessing
# ================================================================

class TestSceneErrorGuessing:

    def test_scene_data_storage(self):
        """EG: Scene data dict works as expected"""
        scene = Scene()
        scene.data["level"] = 3
        scene.data["score"] = 1000
        assert scene.data["level"] == 3

    def test_add_tagged_object_indexes_tags(self):
        """EG: Tags on object before add_object are indexed"""
        scene = Scene()
        obj = GameObject("Obj")
        obj.add_tag("special")
        scene.add_object(obj)
        found = scene.find_objects_with_tag("special")
        assert len(found) == 1
        assert found[0] is obj

    def test_remove_object_cleans_up_name_index(self):
        """EG: Removing object cleans up name index"""
        scene = Scene()
        obj = GameObject("Player")
        scene.add_object(obj)
        scene.remove_object(obj)
        assert scene.find_object("Player") is None

    def test_remove_object_cleans_up_tag_index(self):
        """EG: Removing object cleans up tag index"""
        scene = Scene()
        obj = GameObject("Obj")
        obj.add_tag("enemy")
        scene.add_object(obj)
        scene.remove_object(obj)
        assert scene.find_objects_with_tag("enemy") == []

    def test_many_objects_performance(self):
        """EG: Scene handles many objects"""
        scene = Scene()
        for i in range(100):
            scene.add_object(GameObject(f"Obj{i}"))
        assert scene.get_object_count() == 100
        scene.update(0.016)
        assert scene.get_object_count() == 100
