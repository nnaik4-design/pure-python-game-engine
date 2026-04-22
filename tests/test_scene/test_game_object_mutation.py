"""
Mutation-killing tests for GameObject and Component.
Targets surviving mutants from mutmut analysis.
"""
import pytest
from engine.scene.game_object import GameObject, Component
from engine.scene.scene import Scene


class TestGameObjectMutationKillers:

    def test_component_game_object_initially_none(self):
        """Kill mutant 2: game_object initial value None mutated to '' """
        c = Component()
        assert c.game_object is None

    def test_game_object_scene_initially_none(self):
        """Kill mutant 17: scene initial value - verify it is None not a string"""
        go = GameObject("test")
        assert go.scene is None

    def test_str_starts_with_gameobject(self):
        """Kill mutant 49: __str__ format string mutation"""
        go = GameObject("player")
        s = str(go)
        assert s.startswith("GameObject(")
        assert s.endswith(")")
        assert "name='player'" in s
        assert "active=True" in s
        assert "components=0" in s

    def test_scene_remove_cleans_empty_tag_list(self):
        """Kill scene mutant 19: 'if not objects_by_tag[tag]' → 'if objects_by_tag[tag]'
        After removing the only object with a tag, the tag key should be deleted.
        """
        scene = Scene("test")
        go = GameObject("tagged")
        go.add_tag("enemy")
        scene.add_object(go)
        # Tag should exist in scene
        assert "enemy" in scene.objects_by_tag
        assert go in scene.objects_by_tag["enemy"]
        # Remove the object
        scene.remove_object(go)
        # Tag key should be cleaned up since no objects have it
        assert "enemy" not in scene.objects_by_tag
