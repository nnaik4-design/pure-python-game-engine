"""
Whitebox tests for Scene - targeting uncovered branches.
Focus on render() method, initialize(), and edge cases in add/remove.
"""
import pytest
from unittest.mock import MagicMock
from engine.scene.scene import Scene
from engine.scene.game_object import GameObject, Component


class TestSceneInitialize:
    """Whitebox: cover initialize() (lines 22-23)"""

    def test_initialize_calls_start_on_all_objects(self):
        """Branch: initialize iterates objects and calls start()"""
        class TrackingComp(Component):
            def __init__(self):
                super().__init__()
                self.started = False
            def start(self):
                self.started = True

        scene = Scene()
        obj = GameObject()
        tc = TrackingComp()
        obj.add_component(tc)
        scene.add_object(obj)
        scene.initialize()
        assert tc.started is True

    def test_initialize_empty_scene(self):
        """Branch: initialize on empty scene doesn't crash"""
        scene = Scene()
        scene.initialize()  # Should not raise


class TestSceneRender:
    """Whitebox: cover render() method (lines 86-97)"""

    def test_render_sorts_by_z_order(self):
        """Branch: render sorts active objects by z_order"""
        render_order = []

        class OrderTracker(Component):
            def __init__(self, name):
                super().__init__()
                self._name = name
            def render(self, renderer):
                render_order.append(self._name)

        scene = Scene()
        # Add in wrong z-order
        obj_back = GameObject("Back")
        obj_back.z_order = 0
        obj_back.add_component(OrderTracker("back"))

        obj_front = GameObject("Front")
        obj_front.z_order = 10
        obj_front.add_component(OrderTracker("front"))

        obj_mid = GameObject("Mid")
        obj_mid.z_order = 5
        obj_mid.add_component(OrderTracker("mid"))

        scene.add_object(obj_front)
        scene.add_object(obj_back)
        scene.add_object(obj_mid)

        renderer = MagicMock()
        scene.render(renderer)
        assert render_order == ["back", "mid", "front"]

    def test_render_inactive_scene(self):
        """Branch: inactive scene doesn't render"""
        class TrackingComp(Component):
            def __init__(self):
                super().__init__()
                self.rendered = False
            def render(self, renderer):
                self.rendered = True

        scene = Scene()
        obj = GameObject()
        tc = TrackingComp()
        obj.add_component(tc)
        scene.add_object(obj)
        scene.set_active(False)
        scene.render(MagicMock())
        assert tc.rendered is False

    def test_render_skips_inactive_objects(self):
        """Branch: render skips inactive objects"""
        render_order = []

        class OrderTracker(Component):
            def __init__(self, name):
                super().__init__()
                self._name = name
            def render(self, renderer):
                render_order.append(self._name)

        scene = Scene()
        obj_active = GameObject("Active")
        obj_active.add_component(OrderTracker("active"))
        obj_inactive = GameObject("Inactive")
        obj_inactive.add_component(OrderTracker("inactive"))
        obj_inactive.set_active(False)

        scene.add_object(obj_active)
        scene.add_object(obj_inactive)
        scene.render(MagicMock())
        assert render_order == ["active"]


class TestSceneAddRemoveEdgeCases:
    """Whitebox: cover edge cases in add/remove (branches in lines 32, 48, 53-56)"""

    def test_add_object_without_name(self):
        """Branch: object with empty name (line 32 - name check)"""
        scene = Scene()
        obj = GameObject("")
        scene.add_object(obj)
        assert scene.get_object_count() == 1

    def test_remove_cleans_multiple_tags(self):
        """Branch: removing object cleans up multiple tags"""
        scene = Scene()
        obj = GameObject("Multi")
        obj.add_tag("a")
        obj.add_tag("b")
        scene.add_object(obj)
        scene.remove_object(obj)
        assert scene.find_objects_with_tag("a") == []
        assert scene.find_objects_with_tag("b") == []
