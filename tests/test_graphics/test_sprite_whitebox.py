"""
Whitebox tests for Sprite - targeting uncovered branches.
Focus on render() method and update() with atlas animation.
"""
import pytest
from unittest.mock import MagicMock, patch
from engine.graphics.sprite import Sprite, SpriteAnimation, SpriteAtlas
from engine.scene.game_object import GameObject
from engine.math.vector2 import Vector2


class TestSpriteUpdateWithAtlas:
    """Whitebox: cover update() with atlas animation (lines 225-230)"""

    def test_update_with_animation_and_atlas(self):
        """Branch: update with playing animation updates atlas sprite name"""
        sprite = Sprite()
        atlas = SpriteAtlas(Vector2(256, 256))
        atlas.add_sprite("walk_frame_0", Vector2(0, 0), Vector2(32, 32))
        atlas.add_sprite("walk_frame_1", Vector2(32, 0), Vector2(32, 32))
        sprite.set_sprite_atlas(atlas)
        sprite.add_animation("walk", [0, 1], frame_duration=0.1)
        sprite.play_animation("walk")
        sprite.update(0.05)  # Not enough to advance
        # Frame 0 - name should update to "walk_frame_0"
        assert sprite.current_sprite_name == "walk_frame_0"

    def test_update_animation_no_atlas(self):
        """Branch: update with animation but no atlas just updates frame"""
        sprite = Sprite()
        sprite.add_animation("walk", [0, 1, 2])
        sprite.play_animation("walk")
        sprite.update(0.15)
        # Animation advances but no atlas to update


class TestSpriteRender:
    """Whitebox: cover render() method (lines 234-297)"""

    def test_render_rectangle(self):
        """Branch: render rectangle shape calls draw_rectangle"""
        obj = GameObject()
        sprite = Sprite(color='#FF0000', size=Vector2(50, 50), shape='rectangle')
        obj.add_component(sprite)
        renderer = MagicMock()
        sprite.render(renderer)
        renderer.draw_rectangle.assert_called_once()

    def test_render_circle(self):
        """Branch: render circle shape calls draw_circle"""
        obj = GameObject()
        sprite = Sprite(color='#00FF00', size=Vector2(50, 50), shape='circle')
        obj.add_component(sprite)
        renderer = MagicMock()
        sprite.render(renderer)
        renderer.draw_circle.assert_called_once()

    def test_render_triangle(self):
        """Branch: render triangle shape calls draw_polygon"""
        obj = GameObject()
        sprite = Sprite(color='#0000FF', size=Vector2(50, 50), shape='triangle')
        obj.add_component(sprite)
        renderer = MagicMock()
        sprite.render(renderer)
        renderer.draw_polygon.assert_called_once()

    def test_render_invisible_skipped(self):
        """Branch: invisible sprite skips render"""
        obj = GameObject()
        sprite = Sprite()
        sprite.visible = False
        obj.add_component(sprite)
        renderer = MagicMock()
        sprite.render(renderer)
        renderer.draw_rectangle.assert_not_called()

    def test_render_no_game_object_skipped(self):
        """Branch: sprite without game_object skips render"""
        sprite = Sprite()
        renderer = MagicMock()
        sprite.render(renderer)
        renderer.draw_rectangle.assert_not_called()

    def test_render_with_atlas(self):
        """Branch: render uses atlas sprite data when available"""
        obj = GameObject()
        atlas = SpriteAtlas(Vector2(256, 256))
        atlas.add_sprite("player", Vector2(0, 0), Vector2(32, 32), '#FF0000')
        sprite = Sprite(size=Vector2(50, 50))
        sprite.set_sprite_atlas(atlas, "player")
        obj.add_component(sprite)
        renderer = MagicMock()
        sprite.render(renderer)
        renderer.draw_rectangle.assert_called_once()

    def test_render_with_tint(self):
        """Branch: render applies tint shader effect"""
        obj = GameObject()
        sprite = Sprite(color='#FFFFFF')
        sprite.set_tint('#FF0000')
        obj.add_component(sprite)
        renderer = MagicMock()
        sprite.render(renderer)
        renderer.draw_rectangle.assert_called_once()

    def test_render_with_rotation(self):
        """Branch: render with rotated transform"""
        import math
        obj = GameObject()
        obj.set_rotation(math.pi / 4)
        sprite = Sprite(size=Vector2(50, 50), shape='triangle')
        obj.add_component(sprite)
        renderer = MagicMock()
        sprite.render(renderer)
        renderer.draw_polygon.assert_called_once()

    def test_render_with_scale(self):
        """Branch: render with scaled transform"""
        obj = GameObject()
        obj.set_scale(Vector2(2, 3))
        sprite = Sprite(size=Vector2(50, 50))
        obj.add_component(sprite)
        renderer = MagicMock()
        sprite.render(renderer)
        renderer.draw_rectangle.assert_called_once()
        # Verify the actual_size passed accounts for scale
        call_args = renderer.draw_rectangle.call_args
        actual_size = call_args[0][1]  # Second positional arg
        assert actual_size.x == pytest.approx(100)
        assert actual_size.y == pytest.approx(150)
