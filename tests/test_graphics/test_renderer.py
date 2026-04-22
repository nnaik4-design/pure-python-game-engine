import pytest
from unittest.mock import MagicMock
from engine.graphics.renderer import Renderer
from engine.math.vector2 import Vector2


@pytest.fixture
def renderer():
    canvas = MagicMock()
    canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
    return Renderer(canvas)


# 7. Test utility methods: get_size, get_center
def test_renderer_utility_methods(renderer):
    """Test getter methods return correct values"""
    size = renderer.get_size()
    assert isinstance(size, Vector2)
    assert size.x == 800
    assert size.y == 600

    center = renderer.get_center()
    assert isinstance(center, Vector2)
    assert center.x == 400
    assert center.y == 300


# 8. Test render layers
class TestRenderLayers:
    """Test render layer management and depth sorting"""

    def test_render_layer_set_and_track(self):
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
        renderer = Renderer(canvas)

        # Set different layers
        renderer.set_render_layer(0)
        assert renderer.current_layer == 0
        assert 0 in renderer.render_layers

        renderer.set_render_layer(5)
        assert renderer.current_layer == 5
        assert 5 in renderer.render_layers

        # Multiple layers should exist
        renderer.set_render_layer(2)
        assert 0 in renderer.render_layers
        assert 2 in renderer.render_layers
        assert 5 in renderer.render_layers

    def test_draw_text_with_anchors(self):
        """Test text drawing with various anchor points"""
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
        renderer = Renderer(canvas)

        anchors = ['center', 'nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se']
        position = Vector2(100, 100)
        text = "Test"

        for anchor in anchors:
            canvas.reset_mock()
            renderer.draw_text(position, text, anchor=anchor)
            canvas.create_text.assert_called_once()
            args, kwargs = canvas.create_text.call_args
            assert kwargs['anchor'] == anchor
            assert kwargs['text'] == text

class TestPostProcessingEdgeCases:
    """Test uncovered branches in post-processing effects"""

    def test_add_duplicate_post_processing_effect(self):
        """Test adding an effect that already exists (uncovered branch)"""
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
        renderer = Renderer(canvas)

        renderer.add_post_processing_effect("bloom")
        renderer.add_post_processing_effect("bloom")  # Add duplicate

        assert renderer.post_processing_effects.count("bloom") == 1

    def test_remove_post_processing_effect(self):
        """Test removing existing and non-existing effects (uncovered branch)"""
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
        renderer = Renderer(canvas)

        renderer.add_post_processing_effect("bloom")
        renderer.remove_post_processing_effect("bloom")
        assert "bloom" not in renderer.post_processing_effects

        # Try removing non-existent effect (shouldn't crash)
        renderer.remove_post_processing_effect("blur")
        assert len(renderer.post_processing_effects) == 0

    def test_set_render_layer_existing_layer(self):
        """Test setting a layer that already exists (uncovered branch)"""
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
        renderer = Renderer(canvas)

        renderer.set_render_layer(0)
        assert 0 in renderer.render_layers

        renderer.set_render_layer(0)  # Set same layer again
        assert renderer.current_layer == 0
        assert 0 in renderer.render_layers

    def test_draw_sprite_without_scale(self):
        """Test draw_sprite_from_atlas with scale=None (uncovered branch)"""
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
        renderer = Renderer(canvas)

        position = Vector2(100, 100)
        atlas_data = {'size': Vector2(32, 32), 'color': '#FFFFFF'}

        # Call with scale=None (default)
        renderer.draw_sprite_from_atlas(position, atlas_data=atlas_data, scale=None)

        # Should use original size
        renderer.canvas.create_rectangle.assert_called_once()
        args, kwargs = renderer.canvas.create_rectangle.call_args
        x1, y1, x2, y2 = args[:4]

        expected_x1 = position.x - 32 / 2
        expected_x2 = position.x + 32 / 2
        assert abs(x1 - expected_x1) < 1e-4
        assert abs(x2 - expected_x2) < 1e-4


# 10. Test drawing functions that aren't tested
class TestDrawingFunctions:
    """Test drawing functions with missing coverage"""

    def test_draw_line(self):
        """Test line drawing"""
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
        renderer = Renderer(canvas)

        start = Vector2(0, 0)
        end = Vector2(100, 100)
        color = "#FF0000"
        width = 2

        renderer.draw_line(start, end, color=color, width=width)

        renderer.canvas.create_line.assert_called_once()
        args, kwargs = renderer.canvas.create_line.call_args
        assert args[0] == start.x
        assert args[1] == start.y
        assert args[2] == end.x
        assert args[3] == end.y
        assert kwargs['fill'] == color
        assert kwargs['width'] == width

    def test_draw_polygon(self):
        """Test polygon drawing"""
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
        renderer = Renderer(canvas)

        points = [Vector2(0, 0), Vector2(100, 0), Vector2(100, 100), Vector2(0, 100)]
        color = "#00FF00"
        outline = "#FF0000"
        width = 2

        renderer.draw_polygon(points, color=color, outline=outline, width=width)

        renderer.canvas.create_polygon.assert_called_once()
        args, kwargs = renderer.canvas.create_polygon.call_args
        coords = args[0]

        assert len(coords) == 8  # 4 points * 2 coordinates
        assert kwargs['fill'] == color
        assert kwargs['outline'] == outline
        assert kwargs['width'] == width

    def test_clear_canvas(self):
        """Test clearing the canvas"""
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
        renderer = Renderer(canvas)

        renderer.clear()

        renderer.canvas.delete.assert_called_once_with("all")

    def test_apply_post_processing_each_effect(self):
        """Test each effect individually to ensure all branches are covered"""
        canvas = MagicMock()
        canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600

        # Test bloom effect
        renderer = Renderer(canvas)
        renderer.post_processing_effects = ["bloom"]
        renderer.apply_post_processing()

        # Test blur effect
        renderer = Renderer(canvas)
        renderer.post_processing_effects = ["blur"]
        renderer.apply_post_processing()

        # Test vintage effect
        renderer = Renderer(canvas)
        renderer.post_processing_effects = ["vintage"]
        renderer.apply_post_processing()

        # All should pass without error
        assert True