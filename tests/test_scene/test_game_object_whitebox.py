"""
Whitebox tests for GameObject - targeting uncovered branches.
Focus on Component lifecycle (start/render/destroy), tag+scene interaction, render path.
"""
import pytest
from unittest.mock import MagicMock
from engine.scene.game_object import GameObject, Component
from engine.scene.scene import Scene
from engine.math.vector2 import Vector2


class TestComponentLifecycle:
    """Whitebox: cover Component.start(), render(), destroy() base methods (lines 22, 26, 30)"""

    def test_component_start_does_nothing(self):
        """Branch: base Component.start() is a no-op"""
        comp = Component()
        comp.start()  # Should not raise

    def test_component_render_does_nothing(self):
        """Branch: base Component.render() is a no-op"""
        comp = Component()
        renderer = MagicMock()
        comp.render(renderer)  # Should not raise

    def test_component_destroy_does_nothing(self):
        """Branch: base Component.destroy() is a no-op"""
        comp = Component()
        comp.destroy()  # Should not raise

    def test_component_update_does_nothing(self):
        """Branch: base Component.update() is a no-op"""
        comp = Component()
        comp.update(0.016)  # Should not raise


class TestGameObjectStartWithScene:
    """Whitebox: cover start() calling component.start() (lines 62-63)"""

    def test_start_calls_component_start(self):
        """Branch: start() iterates components and calls start()"""
        class TrackingComp(Component):
            def __init__(self):
                super().__init__()
                self.started = False
            def start(self):
                self.started = True

        obj = GameObject()
        tc = TrackingComp()
        obj.add_component(tc)
        obj.start()
        assert tc.started is True


class TestGameObjectRender:
    """Whitebox: cover render() method (lines 76-81)"""

    def test_render_calls_component_render(self):
        """Branch: render() calls render on active components"""
        class TrackingComp(Component):
            def __init__(self):
                super().__init__()
                self.rendered = False
            def render(self, renderer):
                self.rendered = True

        obj = GameObject()
        tc = TrackingComp()
        obj.add_component(tc)
        renderer = MagicMock()
        obj.render(renderer)
        assert tc.rendered is True

    def test_render_skips_inactive(self):
        """Branch: inactive object skips render"""
        class TrackingComp(Component):
            def __init__(self):
                super().__init__()
                self.rendered = False
            def render(self, renderer):
                self.rendered = True

        obj = GameObject()
        tc = TrackingComp()
        obj.add_component(tc)
        obj.set_active(False)
        obj.render(MagicMock())
        assert tc.rendered is False

    def test_render_skips_destroyed(self):
        """Branch: destroyed object skips render"""
        obj = GameObject()
        obj.destroy()
        obj.render(MagicMock())  # Should not raise

    def test_render_skips_inactive_component(self):
        """Branch: inactive component skipped during render"""
        class TrackingComp(Component):
            def __init__(self):
                super().__init__()
                self.rendered = False
            def render(self, renderer):
                self.rendered = True

        obj = GameObject()
        tc = TrackingComp()
        tc.is_active = False
        obj.add_component(tc)
        obj.render(MagicMock())
        assert tc.rendered is False


class TestAddComponentWithScene:
    """Whitebox: cover add_component when object is in scene (line 98)"""

    def test_add_component_to_scene_object_calls_start(self):
        """Branch: adding component to object already in scene calls start()"""
        class TrackingComp(Component):
            def __init__(self):
                super().__init__()
                self.started = False
            def start(self):
                self.started = True

        scene = Scene()
        obj = GameObject()
        scene.add_object(obj)
        tc = TrackingComp()
        obj.add_component(tc)
        assert tc.started is True


class TestTagsWithScene:
    """Whitebox: cover tag methods that interact with scene (lines 128-142)"""

    def test_add_tag_with_scene_new_tag_group(self):
        """Branch: add_tag creates new tag group in scene (line 128-130)"""
        scene = Scene()
        obj = GameObject()
        scene.add_object(obj)
        obj.add_tag("enemy")
        assert "enemy" in scene.objects_by_tag
        assert obj in scene.objects_by_tag["enemy"]

    def test_add_tag_with_scene_existing_tag_group(self):
        """Branch: add_tag appends to existing tag group"""
        scene = Scene()
        obj1 = GameObject("A")
        obj1.add_tag("enemy")
        scene.add_object(obj1)
        obj2 = GameObject("B")
        scene.add_object(obj2)
        obj2.add_tag("enemy")
        assert len(scene.objects_by_tag["enemy"]) == 2

    def test_remove_tag_with_scene(self):
        """Branch: remove_tag updates scene indices (lines 139-142)"""
        scene = Scene()
        obj = GameObject()
        scene.add_object(obj)
        obj.add_tag("enemy")
        obj.remove_tag("enemy")
        assert "enemy" not in scene.objects_by_tag

    def test_remove_tag_with_scene_other_objects_remain(self):
        """Branch: remove_tag from one object doesn't remove tag group if others have it"""
        scene = Scene()
        obj1 = GameObject("A")
        obj2 = GameObject("B")
        obj1.add_tag("enemy")
        obj2.add_tag("enemy")
        scene.add_object(obj1)
        scene.add_object(obj2)
        obj1.remove_tag("enemy")
        assert "enemy" in scene.objects_by_tag
        assert len(scene.objects_by_tag["enemy"]) == 1
