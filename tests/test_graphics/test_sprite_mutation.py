"""
Mutation-killing tests for Sprite, SpriteAnimation, SpriteAtlas.
Targets surviving mutants from mutmut analysis.
"""
import pytest
from engine.graphics.sprite import Sprite, SpriteAnimation, SpriteAtlas
from engine.math.vector2 import Vector2


class TestSpriteAnimationMutationKillers:

    def test_frame_timer_accumulates(self):
        """Kill mutant 18: frame_timer += delta_time mutated to frame_timer = delta_time"""
        anim = SpriteAnimation("walk", [0, 1, 2], frame_duration=0.5)
        anim.play()
        # First update: add 0.2
        anim.update(0.2)
        assert anim.frame_timer == pytest.approx(0.2)
        # Second update: should accumulate to 0.4, not reset to 0.15
        anim.update(0.15)
        assert anim.frame_timer == pytest.approx(0.35)

    def test_frame_timer_gte_boundary(self):
        """Kill mutant 20: >= mutated to >"""
        anim = SpriteAnimation("walk", [0, 1, 2], frame_duration=0.5)
        anim.play()
        # Exactly at boundary should advance frame
        anim.update(0.5)
        assert anim.current_frame == 1

    def test_frame_timer_resets_to_zero(self):
        """Kill mutant 21: frame_timer = 0.0 mutated to 1.0"""
        anim = SpriteAnimation("walk", [0, 1, 2], frame_duration=0.5)
        anim.play()
        anim.update(0.5)  # triggers frame advance, timer resets
        assert anim.frame_timer == pytest.approx(0.0)


class TestSpriteAtlasMutationKillers:

    def test_add_sprite_default_color(self):
        """Kill mutant 46: default color '#FFFFFF' mutated"""
        atlas = SpriteAtlas(Vector2(256, 256))
        atlas.add_sprite("test", Vector2(0, 0), Vector2(32, 32))
        info = atlas.get_sprite_data("test")
        assert info['color'] == '#FFFFFF'

    def test_create_animation_horizontal_default(self):
        """Kill mutant 59: horizontal default True → False"""
        atlas = SpriteAtlas(Vector2(256, 256))
        # With horizontal=True (default), x position should advance
        names = atlas.create_animation_frames("walk", 3, Vector2(32, 32), Vector2(0, 0))
        assert len(names) == 3
        # Second frame should be at x=32 (horizontal), not y=32
        info1 = atlas.get_sprite_data(names[1])
        assert info1['position'].x == pytest.approx(32.0)
        assert info1['position'].y == pytest.approx(0.0)


class TestSpriteMutationKillers:

    def test_outline_color_initially_none(self):
        """Kill mutant 77: outline_color initial None → '' """
        s = Sprite('#FF0000')
        assert s.outline_color is None

    def test_current_animation_initially_none(self):
        """Kill mutant 85: current_animation initial None → '' """
        s = Sprite('#FF0000')
        assert s.current_animation is None

    def test_set_outline_default_width(self):
        """Kill mutant 96: set_outline default width=1 → 2"""
        s = Sprite('#FF0000')
        s.set_outline('#000000')  # use default width
        assert s.outline_width == 1

    def test_add_animation_default_duration(self):
        """Kill mutant 102: frame_duration default 0.1 → 1.1"""
        s = Sprite('#FF0000')
        s.add_animation("walk", [0, 1, 2])  # use default duration
        anim = s.animations["walk"]
        assert anim.frame_duration == pytest.approx(0.1)

    def test_add_shader_effect_stores_data(self):
        """Kill mutant 120: shader_effects[name] = effect_data → None"""
        s = Sprite('#FF0000')
        data = {"glow": 0.5, "color": "blue"}
        s.add_shader_effect("glow", data)
        assert s.shader_effects["glow"] is data
        assert s.shader_effects["glow"]["glow"] == 0.5

    def test_contains_point_circle_shape(self):
        """Kill mutant 130: 'circle' string mutation"""
        s = Sprite('#FF0000', size=Vector2(100, 100), shape='circle')
        from engine.scene.game_object import GameObject
        go = GameObject("test")
        go.add_component(s)
        # Point at center should be inside
        assert s.contains_point(Vector2(0, 0)) is True
        # Point far away should be outside
        assert s.contains_point(Vector2(200, 200)) is False

    def test_sprite_alpha_default(self):
        """Kill mutant: alpha default 1.0"""
        s = Sprite('#FF0000')
        assert s.alpha == 1.0

    def test_sprite_visible_default(self):
        """Kill mutant: visible default True"""
        s = Sprite('#FF0000')
        assert s.visible is True
