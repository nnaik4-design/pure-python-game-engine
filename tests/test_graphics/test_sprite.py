"""
Blackbox tests for SpriteAnimation, SpriteAtlas, and Sprite classes.
Tests use Equivalence Partitioning (EP), Boundary Analysis (BA), and Error Guessing (EG).
"""
import pytest
from engine.graphics.sprite import SpriteAnimation, SpriteAtlas, Sprite
from engine.scene.game_object import GameObject
from engine.math.vector2 import Vector2


# ================================================================
# SpriteAnimation Constructor Tests
# ================================================================

class TestSpriteAnimationConstructor:

    def test_constructor(self):
        """EP: Animation initializes with correct defaults"""
        anim = SpriteAnimation("walk", [0, 1, 2, 3])
        assert anim.name == "walk"
        assert anim.frame_indices == [0, 1, 2, 3]
        assert anim.frame_duration == 0.1
        assert anim.loop is True
        assert anim.current_frame == 0
        assert anim.is_playing is False

    def test_custom_duration(self):
        """EP: Custom frame duration"""
        anim = SpriteAnimation("run", [0, 1], frame_duration=0.05)
        assert anim.frame_duration == 0.05

    def test_no_loop(self):
        """EP: Non-looping animation"""
        anim = SpriteAnimation("die", [0, 1, 2], loop=False)
        assert anim.loop is False


# ================================================================
# SpriteAnimation Playback Tests
# ================================================================

class TestSpriteAnimationPlayback:

    def test_play(self):
        """EP: Play starts the animation"""
        anim = SpriteAnimation("walk", [0, 1, 2])
        anim.play()
        assert anim.is_playing is True

    def test_stop(self):
        """EP: Stop resets animation"""
        anim = SpriteAnimation("walk", [0, 1, 2])
        anim.play()
        anim.current_frame = 2
        anim.stop()
        assert anim.is_playing is False
        assert anim.current_frame == 0
        assert anim.frame_timer == 0.0

    def test_pause(self):
        """EP: Pause stops without resetting"""
        anim = SpriteAnimation("walk", [0, 1, 2])
        anim.play()
        anim.current_frame = 1
        anim.pause()
        assert anim.is_playing is False
        assert anim.current_frame == 1  # Not reset

    def test_update_not_playing_returns_first_frame(self):
        """EP: Update when not playing returns first frame index"""
        anim = SpriteAnimation("walk", [5, 6, 7])
        result = anim.update(0.1)
        assert result == 5

    def test_update_advances_frame(self):
        """EP: Update with enough time advances frame"""
        anim = SpriteAnimation("walk", [0, 1, 2], frame_duration=0.1)
        anim.play()
        # Advance past first frame
        idx = anim.update(0.15)
        assert anim.current_frame == 1
        assert idx == 1

    def test_update_not_enough_time(self):
        """BA: Update with less than frame_duration stays on current frame"""
        anim = SpriteAnimation("walk", [0, 1, 2], frame_duration=0.1)
        anim.play()
        idx = anim.update(0.05)
        assert anim.current_frame == 0
        assert idx == 0

    def test_looping_wraps_around(self):
        """EP: Looping animation wraps back to first frame"""
        anim = SpriteAnimation("walk", [0, 1], frame_duration=0.1, loop=True)
        anim.play()
        anim.update(0.15)  # Goes to frame 1
        anim.update(0.15)  # Should wrap to frame 0
        assert anim.current_frame == 0

    def test_non_looping_stops_at_last_frame(self):
        """EP: Non-looping animation stops at last frame"""
        anim = SpriteAnimation("die", [0, 1, 2], frame_duration=0.1, loop=False)
        anim.play()
        anim.update(0.15)  # Frame 1
        anim.update(0.15)  # Frame 2
        anim.update(0.15)  # Past end - should stay at frame 2
        assert anim.current_frame == 2
        assert anim.is_playing is False

    def test_empty_frames_returns_zero(self):
        """BA: Empty frame_indices returns 0"""
        anim = SpriteAnimation("empty", [])
        result = anim.update(0.1)
        assert result == 0

    def test_single_frame_animation(self):
        """BA: Single frame animation stays on that frame"""
        anim = SpriteAnimation("static", [5], frame_duration=0.1)
        anim.play()
        idx = anim.update(0.05)
        assert idx == 5


# ================================================================
# SpriteAtlas Tests
# ================================================================

class TestSpriteAtlas:

    def test_constructor(self):
        """EP: Atlas has texture size and empty sprites"""
        atlas = SpriteAtlas(Vector2(256, 256))
        assert atlas.texture_size == Vector2(256, 256)
        assert len(atlas.sprites) == 0

    def test_add_sprite(self):
        """EP: Add sprite region to atlas"""
        atlas = SpriteAtlas(Vector2(256, 256))
        atlas.add_sprite("player", Vector2(0, 0), Vector2(32, 32), '#FF0000')
        data = atlas.get_sprite_data("player")
        assert data is not None
        assert data['position'] == Vector2(0, 0)
        assert data['size'] == Vector2(32, 32)
        assert data['color'] == '#FF0000'

    def test_uv_coordinates(self):
        """EP: UV coordinates calculated correctly"""
        atlas = SpriteAtlas(Vector2(256, 256))
        atlas.add_sprite("sprite", Vector2(64, 128), Vector2(32, 32))
        data = atlas.get_sprite_data("sprite")
        assert data['uv_start'].x == pytest.approx(64 / 256)
        assert data['uv_start'].y == pytest.approx(128 / 256)
        assert data['uv_end'].x == pytest.approx((64 + 32) / 256)
        assert data['uv_end'].y == pytest.approx((128 + 32) / 256)

    def test_get_nonexistent_sprite(self):
        """BA: Getting nonexistent sprite returns None"""
        atlas = SpriteAtlas(Vector2(256, 256))
        assert atlas.get_sprite_data("nonexistent") is None

    def test_create_animation_frames_horizontal(self):
        """EP: Create horizontal animation frames"""
        atlas = SpriteAtlas(Vector2(256, 256))
        names = atlas.create_animation_frames(
            "walk", 4, Vector2(32, 32), Vector2(0, 0), horizontal=True
        )
        assert len(names) == 4
        assert names[0] == "walk_frame_0"
        assert names[3] == "walk_frame_3"
        # Verify positions
        data0 = atlas.get_sprite_data("walk_frame_0")
        data1 = atlas.get_sprite_data("walk_frame_1")
        assert data0['position'] == Vector2(0, 0)
        assert data1['position'] == Vector2(32, 0)

    def test_create_animation_frames_vertical(self):
        """EP: Create vertical animation frames"""
        atlas = SpriteAtlas(Vector2(256, 256))
        names = atlas.create_animation_frames(
            "climb", 3, Vector2(32, 32), Vector2(0, 0), horizontal=False
        )
        assert len(names) == 3
        data0 = atlas.get_sprite_data("climb_frame_0")
        data1 = atlas.get_sprite_data("climb_frame_1")
        assert data0['position'] == Vector2(0, 0)
        assert data1['position'] == Vector2(0, 32)

    def test_create_zero_frames(self):
        """BA: Zero frame count creates no frames"""
        atlas = SpriteAtlas(Vector2(256, 256))
        names = atlas.create_animation_frames(
            "empty", 0, Vector2(32, 32), Vector2(0, 0)
        )
        assert names == []


# ================================================================
# Sprite Constructor Tests
# ================================================================

class TestSpriteConstructor:

    def test_default_constructor(self):
        """EP: Sprite defaults"""
        sprite = Sprite()
        assert sprite.color == '#FFFFFF'
        assert sprite.size == Vector2(50, 50)
        assert sprite.shape == 'rectangle'
        assert sprite.visible is True
        assert sprite.alpha == 1.0

    def test_custom_constructor(self):
        """EP: Custom color, size, shape"""
        sprite = Sprite(color='#FF0000', size=Vector2(100, 50), shape='circle')
        assert sprite.color == '#FF0000'
        assert sprite.size == Vector2(100, 50)
        assert sprite.shape == 'circle'

    def test_default_shader_values(self):
        """EP: Shader defaults"""
        sprite = Sprite()
        assert sprite.brightness == 1.0
        assert sprite.contrast == 1.0
        assert sprite.tint_color is None


# ================================================================
# Sprite Property Tests
# ================================================================

class TestSpriteProperties:

    def test_set_color(self):
        """EP: Set sprite color"""
        sprite = Sprite()
        sprite.set_color('#00FF00')
        assert sprite.color == '#00FF00'

    def test_set_size(self):
        """EP: Set sprite size"""
        sprite = Sprite()
        sprite.set_size(Vector2(100, 200))
        assert sprite.size == Vector2(100, 200)

    def test_set_outline(self):
        """EP: Set outline properties"""
        sprite = Sprite()
        sprite.set_outline('#000000', 2)
        assert sprite.outline_color == '#000000'
        assert sprite.outline_width == 2

    def test_set_alpha_normal(self):
        """EP: Set alpha in valid range"""
        sprite = Sprite()
        sprite.set_alpha(0.5)
        assert sprite.alpha == 0.5

    def test_set_alpha_clamp_high(self):
        """BA: Alpha clamped to 1.0"""
        sprite = Sprite()
        sprite.set_alpha(2.0)
        assert sprite.alpha == 1.0

    def test_set_alpha_clamp_low(self):
        """BA: Alpha clamped to 0.0"""
        sprite = Sprite()
        sprite.set_alpha(-1.0)
        assert sprite.alpha == 0.0

    def test_set_brightness_normal(self):
        """EP: Set brightness"""
        sprite = Sprite()
        sprite.set_brightness(1.5)
        assert sprite.brightness == 1.5

    def test_set_brightness_clamp_high(self):
        """BA: Brightness clamped to 2.0"""
        sprite = Sprite()
        sprite.set_brightness(5.0)
        assert sprite.brightness == 2.0

    def test_set_brightness_clamp_low(self):
        """BA: Brightness clamped to 0.0"""
        sprite = Sprite()
        sprite.set_brightness(-1.0)
        assert sprite.brightness == 0.0

    def test_set_contrast_normal(self):
        """EP: Set contrast"""
        sprite = Sprite()
        sprite.set_contrast(0.8)
        assert sprite.contrast == 0.8

    def test_set_contrast_clamp_high(self):
        """BA: Contrast clamped to 2.0"""
        sprite = Sprite()
        sprite.set_contrast(3.0)
        assert sprite.contrast == 2.0

    def test_set_contrast_clamp_low(self):
        """BA: Contrast clamped to 0.0"""
        sprite = Sprite()
        sprite.set_contrast(-0.5)
        assert sprite.contrast == 0.0

    def test_get_size_returns_copy(self):
        """EG: get_size returns a copy"""
        sprite = Sprite(size=Vector2(50, 50))
        size = sprite.get_size()
        size.x = 999
        assert sprite.size.x == 50


# ================================================================
# Sprite Animation Integration Tests
# ================================================================

class TestSpriteAnimationIntegration:

    def test_add_animation(self):
        """EP: Add animation to sprite"""
        sprite = Sprite()
        sprite.add_animation("walk", [0, 1, 2, 3], 0.1, True)
        assert "walk" in sprite.animations

    def test_play_animation(self):
        """EP: Play an added animation"""
        sprite = Sprite()
        sprite.add_animation("walk", [0, 1, 2])
        sprite.play_animation("walk")
        assert sprite.current_animation is not None
        assert sprite.current_animation.is_playing is True

    def test_play_nonexistent_animation(self):
        """EG: Playing nonexistent animation doesn't crash"""
        sprite = Sprite()
        sprite.play_animation("nonexistent")  # Should not raise

    def test_stop_animation(self):
        """EP: Stop current animation"""
        sprite = Sprite()
        sprite.add_animation("walk", [0, 1, 2])
        sprite.play_animation("walk")
        sprite.stop_animation()
        assert sprite.current_animation is None

    def test_stop_when_no_animation(self):
        """EG: Stop animation when none playing doesn't crash"""
        sprite = Sprite()
        sprite.stop_animation()  # Should not raise

    def test_play_switches_animation(self):
        """EP: Playing new animation stops the old one"""
        sprite = Sprite()
        sprite.add_animation("walk", [0, 1, 2])
        sprite.add_animation("run", [3, 4, 5])
        sprite.play_animation("walk")
        sprite.play_animation("run")
        assert sprite.current_animation.name == "run"


# ================================================================
# Sprite Atlas Integration Tests
# ================================================================

class TestSpriteAtlasIntegration:

    def test_set_sprite_atlas(self):
        """EP: Set atlas on sprite"""
        sprite = Sprite()
        atlas = SpriteAtlas(Vector2(256, 256))
        atlas.add_sprite("player", Vector2(0, 0), Vector2(32, 32))
        sprite.set_sprite_atlas(atlas, "player")
        assert sprite.sprite_atlas is atlas
        assert sprite.current_sprite_name == "player"

    def test_set_sprite_atlas_no_initial_sprite(self):
        """EP: Set atlas without initial sprite name"""
        sprite = Sprite()
        atlas = SpriteAtlas(Vector2(256, 256))
        sprite.set_sprite_atlas(atlas)
        assert sprite.sprite_atlas is atlas
        assert sprite.current_sprite_name is None

    def test_set_current_sprite(self):
        """EP: Set current sprite from atlas"""
        sprite = Sprite()
        atlas = SpriteAtlas(Vector2(256, 256))
        atlas.add_sprite("idle", Vector2(0, 0), Vector2(32, 32))
        atlas.add_sprite("run", Vector2(32, 0), Vector2(32, 32))
        sprite.set_sprite_atlas(atlas)
        sprite.set_current_sprite("run")
        assert sprite.current_sprite_name == "run"

    def test_set_nonexistent_current_sprite(self):
        """EG: Setting nonexistent sprite name doesn't change"""
        sprite = Sprite()
        atlas = SpriteAtlas(Vector2(256, 256))
        sprite.set_sprite_atlas(atlas)
        sprite.set_current_sprite("nonexistent")
        assert sprite.current_sprite_name is None


# ================================================================
# Shader Effects Tests
# ================================================================

class TestSpriteShaderEffects:

    def test_set_tint(self):
        """EP: Set tint color"""
        sprite = Sprite()
        sprite.set_tint('#FF0000')
        assert sprite.tint_color == '#FF0000'

    def test_add_shader_effect(self):
        """EP: Add custom shader effect"""
        sprite = Sprite()
        sprite.add_shader_effect("glow", {"intensity": 0.5})
        assert "glow" in sprite.shader_effects

    def test_remove_shader_effect(self):
        """EP: Remove shader effect"""
        sprite = Sprite()
        sprite.add_shader_effect("glow", {"intensity": 0.5})
        sprite.remove_shader_effect("glow")
        assert "glow" not in sprite.shader_effects

    def test_remove_nonexistent_effect(self):
        """EG: Removing nonexistent effect doesn't crash"""
        sprite = Sprite()
        sprite.remove_shader_effect("nonexistent")  # Should not raise

    def test_blend_colors_with_tint(self):
        """EP: _apply_shader_effects with tint blends at factor=0.5 (returns base)"""
        sprite = Sprite()
        sprite.set_tint('#FF0000')
        result = sprite._apply_shader_effects('#FFFFFF')
        # _blend_colors uses factor=0.5, and returns color1 when factor <= 0.5
        assert result == '#FFFFFF'

    def test_blend_colors_no_tint(self):
        """EP: _apply_shader_effects without tint returns base"""
        sprite = Sprite()
        result = sprite._apply_shader_effects('#FFFFFF')
        assert result == '#FFFFFF'


# ================================================================
# Contains Point Tests
# ================================================================

class TestSpriteContainsPoint:

    def test_contains_point_no_game_object(self):
        """BA: No game_object returns False"""
        sprite = Sprite(size=Vector2(50, 50))
        assert sprite.contains_point(Vector2(0, 0)) is False

    def test_rectangle_contains_center(self):
        """EP: Point at center is inside rectangle sprite"""
        obj = GameObject("Test")
        sprite = Sprite(size=Vector2(100, 100), shape='rectangle')
        obj.add_component(sprite)
        assert sprite.contains_point(Vector2(0, 0)) is True

    def test_rectangle_contains_edge(self):
        """BA: Point at edge is inside rectangle"""
        obj = GameObject("Test")
        sprite = Sprite(size=Vector2(100, 100), shape='rectangle')
        obj.add_component(sprite)
        assert sprite.contains_point(Vector2(50, 50)) is True

    def test_rectangle_outside(self):
        """EP: Point outside rectangle returns False"""
        obj = GameObject("Test")
        sprite = Sprite(size=Vector2(100, 100), shape='rectangle')
        obj.add_component(sprite)
        assert sprite.contains_point(Vector2(200, 200)) is False

    def test_circle_contains_center(self):
        """EP: Point at center is inside circle sprite"""
        obj = GameObject("Test")
        sprite = Sprite(size=Vector2(100, 100), shape='circle')
        obj.add_component(sprite)
        assert sprite.contains_point(Vector2(0, 0)) is True

    def test_circle_outside(self):
        """EP: Point far outside circle returns False"""
        obj = GameObject("Test")
        sprite = Sprite(size=Vector2(100, 100), shape='circle')
        obj.add_component(sprite)
        assert sprite.contains_point(Vector2(200, 200)) is False

    def test_contains_with_position_offset(self):
        """EP: Contains accounts for object position"""
        obj = GameObject("Test")
        obj.set_position(Vector2(100, 100))
        sprite = Sprite(size=Vector2(50, 50), shape='rectangle')
        obj.add_component(sprite)
        assert sprite.contains_point(Vector2(100, 100)) is True
        assert sprite.contains_point(Vector2(0, 0)) is False

    def test_contains_with_scale(self):
        """EP: Contains accounts for object scale"""
        obj = GameObject("Test")
        obj.set_scale(Vector2(2, 2))
        sprite = Sprite(size=Vector2(50, 50), shape='rectangle')
        obj.add_component(sprite)
        # Scaled size is 100x100, so point at (49, 49) should be inside
        assert sprite.contains_point(Vector2(49, 49)) is True


# ================================================================
# Error Guessing
# ================================================================

class TestSpriteErrorGuessing:

    def test_sprite_update_no_animation(self):
        """EG: Update with no animation doesn't crash"""
        sprite = Sprite()
        sprite.update(0.016)  # Should not raise

    def test_sprite_visible_false(self):
        """EG: Invisible sprite can still check contains_point"""
        obj = GameObject("Test")
        sprite = Sprite(size=Vector2(100, 100))
        sprite.visible = False
        obj.add_component(sprite)
        # contains_point doesn't check visibility
        assert sprite.contains_point(Vector2(0, 0)) is True

    def test_multiple_sprites_on_different_objects(self):
        """EG: Multiple sprites on different objects are independent"""
        obj1 = GameObject("A")
        obj2 = GameObject("B")
        s1 = Sprite(color='#FF0000')
        s2 = Sprite(color='#00FF00')
        obj1.add_component(s1)
        obj2.add_component(s2)
        assert s1.color != s2.color
        assert s1.game_object is obj1
        assert s2.game_object is obj2





class TestSpriteAnimationStateMachine:
    """Test animation state transitions and playback control"""

    def test_animation_play_stop_pause(self):
        """Test play, stop, and pause state transitions"""
        anim = SpriteAnimation("test", [0, 1, 2], frame_duration=0.1, loop=True)

        # Initial state
        assert anim.is_playing is False
        assert anim.current_frame == 0
        assert anim.frame_timer == 0.0

        # Play
        anim.play()
        assert anim.is_playing is True

        # Update and verify frame advances
        frame_before = anim.current_frame
        anim.update(0.15)
        assert anim.current_frame > frame_before

        # Pause (stop)
        anim.pause()
        assert anim.is_playing is False

        # Stop (resets)
        anim.stop()
        assert anim.is_playing is False
        assert anim.current_frame == 0
        assert anim.frame_timer == 0.0

    def test_animation_looping_behavior(self):
        """Test looping vs non-looping animations"""
        # Non-looping animation
        anim_no_loop = SpriteAnimation("no_loop", [0, 1], frame_duration=0.1, loop=False)
        anim_no_loop.play()

        # Update through all frames and beyond
        anim_no_loop.update(0.25)
        assert anim_no_loop.current_frame == 1  # Stays at last frame
        assert anim_no_loop.is_playing is False  # Stops playing

        # Looping animation
        anim_loop = SpriteAnimation("loop", [0, 1], frame_duration=0.1, loop=True)
        anim_loop.play()

        # Update through all frames and beyond
        anim_loop.update(0.25)
        assert anim_loop.current_frame == 0  # Wraps back to start
        assert anim_loop.is_playing is True  # Continues playing