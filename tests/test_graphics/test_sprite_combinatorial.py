import pytest
import math
from unittest.mock import MagicMock
from allpairspy import AllPairs
from engine.graphics.sprite import Sprite, SpriteAnimation, SpriteAtlas
from engine.math.vector2 import Vector2
from engine.scene.game_object import GameObject

# Bug: SpriteAnimation.update didn't handle frame timer correctly when delta time exceeded frame duration, causing skipped frames and incorrect timing.
# 1. PWC for SpriteAnimation.update with comprehensive parameters
# is_playing (True, False), loop (True, False), indices ([], [0, 1]), frame_timer reaching duration (True, False)
_anim_parameters = [
    [True, False],  # is_playing
    [True, False],  # loop
    [True, False],  # has_indices
    [True, False],  # reaches_duration
]
_anim_pwc_cases = [tuple(case) for case in AllPairs(_anim_parameters)]

@pytest.mark.parametrize("is_playing, loop, has_indices, reaches_duration", _anim_pwc_cases)
def test_sprite_animation_update_pwc(is_playing, loop, has_indices, reaches_duration):
    """Test animation frame updates with state tracking"""
    indices = [0, 1] if has_indices else []
    anim = SpriteAnimation("test", indices, frame_duration=0.1, loop=loop)

    if is_playing:
        anim.play()

    if reaches_duration:
        delta = 0.15  # Exceeds frame_duration
    else:
        delta = 0.05  # Below frame_duration

    initial_frame = anim.current_frame
    initial_timer = anim.frame_timer
    result = anim.update(delta)

    if not has_indices:
        # No indices - should always return 0
        assert result == 0
    else:
        if not is_playing:
            # Not playing - should return first frame
            assert result == indices[0]
        else:
            # Playing animation
            if reaches_duration:
                # Timer exceeded duration - frame should advance
                # excess time should be carried over to next frame
                assert anim.frame_timer == (initial_timer + delta - 0.1)
                if initial_frame + 1 >= len(indices):
                    # Wrapped around
                    if loop:
                        assert anim.current_frame == 0
                        assert anim.is_playing is True
                    else:
                        # Non-looping - should stay at last frame and stop
                        assert anim.current_frame == len(indices) - 1
                        assert anim.is_playing is False
                else:
                    assert anim.current_frame == initial_frame + 1
            else:
                # Timer has not reached duration yet
                assert anim.frame_timer == initial_timer + delta
                assert anim.current_frame == initial_frame


# 2. PWC for Sprite property limits - clamping behavior
# Test boundary clamping for alpha, brightness, contrast
_limit_parameters = [
    [-0.5, 0.0, 0.5, 1.0, 1.5],  # alpha input
    [-1.0, 0.0, 1.0, 2.0, 2.5],  # brightness input
    [-1.0, 0.0, 1.0, 2.0, 2.5],  # contrast input
]
_limit_pwc_cases = [tuple(case) for case in AllPairs(_limit_parameters)]

@pytest.mark.parametrize("alpha_in, bright_in, cont_in", _limit_pwc_cases)
def test_sprite_property_clamping_pwc(alpha_in, bright_in, cont_in):
    """Test sprite property clamping to valid ranges"""
    sprite = Sprite()

    sprite.set_alpha(alpha_in)
    sprite.set_brightness(bright_in)
    sprite.set_contrast(cont_in)

    # Verify clamping for alpha [0.0, 1.0]
    assert sprite.alpha == max(0.0, min(1.0, alpha_in))
    assert 0.0 <= sprite.alpha <= 1.0

    # Verify clamping for brightness [0.0, 2.0]
    assert sprite.brightness == max(0.0, min(2.0, bright_in))
    assert 0.0 <= sprite.brightness <= 2.0

    # Verify clamping for contrast [0.0, 2.0]
    assert sprite.contrast == max(0.0, min(2.0, cont_in))
    assert 0.0 <= sprite.contrast <= 2.0

# sprite.contains_point didn't account for triagle shape, it uses and else clause that treats all non-circle shapes as rectangles.
# 3. PWC for contains_point with multiple test points
# shape ('circle', 'rectangle', 'triangle'), point_position (multiple), scale (1.0, 2.0)
_contains_parameters = [
    ['circle', 'rectangle', 'triangle'],  # shape
    [0.5, 1.5],  # scale
]
_contains_pwc_cases = [tuple(case) for case in AllPairs(_contains_parameters)]

def _point_in_triangle(point, v1, v2, v3):
    """Check if a point is inside a triangle using barycentric coordinates"""
    def sign(p1, p2, p3):
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)

    d1 = sign(point, v1, v2)
    d2 = sign(point, v2, v3)
    d3 = sign(point, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)



@pytest.mark.parametrize("shape, scale", _contains_pwc_cases)
def test_sprite_contains_point_pwc(shape, scale):
    """Test point containment with geometric validation"""
    sprite = Sprite(size=Vector2(100, 100), shape=shape)
    go = GameObject()
    go.transform.scale = Vector2(scale, scale)
    go.transform.position = Vector2(0, 0)
    sprite.game_object = go

    actual_size = Vector2(100 * scale, 100 * scale)

    # Test points at various positions
    test_cases = [
        (Vector2(0, 0), True, "center"),
        (Vector2(10, 10), True, "near center"),
        (Vector2(actual_size.x / 2 - 1, 0), True, "near edge"),
        (Vector2(actual_size.x / 2 + 1, 0), False, "just outside edge"),
        (Vector2(actual_size.x + 50, actual_size.y + 50), False, "far outside"),
    ]

    for point, expected, description in test_cases:
        result = sprite.contains_point(point)
        if shape == 'circle':
            radius = max(actual_size.x, actual_size.y) / 2
            distance = point.distance_to(Vector2(0, 0))
            expected = distance <= radius
        elif shape == 'rectangle':
            half_w = actual_size.x / 2
            half_h = actual_size.y / 2
            expected = (-half_w <= point.x <= half_w and -half_h <= point.y <= half_h)
        elif shape == 'triangle':
            # Triangle vertices at world position (0, 0)
            half_w = actual_size.x / 2
            half_h = actual_size.y / 2
            v1 = Vector2(0, -half_h)       # Top
            v2 = Vector2(-half_w, half_h)  # Bottom left
            v3 = Vector2(half_w, half_h)   # Bottom right
            expected = _point_in_triangle(point, v1, v2, v3)

        assert result == expected, f"{shape} - {description}: point {point}, expected {expected}, got {result}"


# 4. PWC for Sprite.render with multiple shapes and visual options
# visible (T/F), has_game_object (T/F), shape ('rectangle', 'circle', 'triangle'), has_outline (T/F)
_render_parameters = [
    [True, False],  # visible
    [True, False],  # has_game_object
    ['rectangle', 'circle', 'triangle'],  # shape
    [True, False],  # has_outline
]
_render_pwc_cases = [tuple(case) for case in AllPairs(_render_parameters)]

@pytest.mark.parametrize("visible, has_game_object, shape, has_outline", _render_pwc_cases)
def test_sprite_render_pwc(visible, has_game_object, shape, has_outline):
    """Test sprite rendering with visual property validation"""
    sprite = Sprite(color="#FF0000", shape=shape, size=Vector2(50, 50))
    sprite.visible = visible

    if has_outline:
        sprite.set_outline("#00FF00", width=2)

    if has_game_object:
        go = GameObject()
        go.transform.position = Vector2(100, 100)
        go.transform.scale = Vector2(1.0, 1.0)
        sprite.game_object = go

    renderer = MagicMock()

    sprite.render(renderer)

    if not visible or not has_game_object:
        # Should return early - no rendering
        renderer.draw_rectangle.assert_not_called()
        renderer.draw_circle.assert_not_called()
        renderer.draw_polygon.assert_not_called()
    else:
        # Should render based on shape
        if shape == 'circle':
            renderer.draw_circle.assert_called_once()
            args, kwargs = renderer.draw_circle.call_args
            # Verify circle position and radius
            circle_pos = args[0]
            radius = args[1]
            assert circle_pos.x == 100
            assert circle_pos.y == 100
            assert radius > 0
            if has_outline:
                assert args[3] == "#00FF00"  # outline_color
                assert args[4] == 2  # outline_width

        elif shape == 'triangle':
            renderer.draw_polygon.assert_called_once()
            args, kwargs = renderer.draw_polygon.call_args
            points = args[0]
            assert len(points) == 3, "Triangle should have 3 points"
            # Verify points form a valid triangle centered at sprite position
            center_x = sum(p.x for p in points) / 3
            center_y = sum(p.y for p in points) / 3
            assert abs(center_x - 100) < 1e-4
            assert abs(center_y - 100) < 1e-4
            if has_outline:
                assert args[2] == "#00FF00"  # outline_color
                assert args[3] == 2  # outline_width

        else:  # rectangle
            renderer.draw_rectangle.assert_called_once()
            args, kwargs = renderer.draw_rectangle.call_args
            rect_pos = args[0]
            rect_size = args[1]
            assert rect_pos.x == 100
            assert rect_pos.y == 100
            assert rect_size.x == 50
            assert rect_size.y == 50
            # Verify outline is passed when set
            if has_outline:
                assert args[4] == "#00FF00"  # outline_color
                assert args[5] == 2  # outline_width

# 5. SpriteAtlas functionality
class TestSpriteAtlas:
    """Test sprite atlas management"""

    def test_sprite_atlas_add_and_retrieve(self):
        """Test adding and retrieving sprites from atlas"""
        atlas = SpriteAtlas(Vector2(256, 256))

        atlas.add_sprite("sprite1", Vector2(0, 0), Vector2(32, 32), "#FF0000")
        data = atlas.get_sprite_data("sprite1")

        assert data is not None
        assert data['size'] == Vector2(32, 32)
        assert data['color'] == "#FF0000"
        assert data['uv_start'] == Vector2(0, 0)
        assert data['uv_end'] == Vector2(32/256, 32/256)

    def test_sprite_atlas_create_animation_frames(self):
        """Test creating animation frames from sprite sheet"""
        atlas = SpriteAtlas(Vector2(256, 256))

        frame_names = atlas.create_animation_frames("walk", 4, Vector2(32, 32), Vector2(0, 0), horizontal=True)

        assert len(frame_names) == 4
        assert frame_names[0] == "walk_frame_0"
        assert frame_names[3] == "walk_frame_3"

        # Verify frames are positioned correctly (horizontally)
        for i, name in enumerate(frame_names):
            data = atlas.get_sprite_data(name)
            expected_x = i * 32
            assert abs(data['position'].x - expected_x) < 1e-6


# 6. Sprite color and shader effects
class TestSpriteShaderEffects:
    """Test sprite visual effects"""

    def test_sprite_tint_and_brightness(self):
        """Test tint color and brightness effects"""
        sprite = Sprite(color="#FFFFFF")

        sprite.set_tint("#FF0000")
        assert sprite.tint_color == "#FF0000"

        sprite.set_brightness(1.5)
        assert sprite.brightness == 1.5

        sprite.set_brightness(2.5)  # Should clamp to 2.0
        assert sprite.brightness == 2.0

    def test_sprite_shader_effect_management(self):
        """Test adding and removing custom shader effects"""
        sprite = Sprite()

        sprite.add_shader_effect("distortion", {"strength": 0.5})
        assert "distortion" in sprite.shader_effects
        assert sprite.shader_effects["distortion"]["strength"] == 0.5

        sprite.remove_shader_effect("distortion")
        assert "distortion" not in sprite.shader_effects

        # Removing non-existent effect shouldn't error
        sprite.remove_shader_effect("nonexistent")
