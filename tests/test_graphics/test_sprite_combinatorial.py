import pytest
from unittest.mock import MagicMock
from engine.graphics.sprite import Sprite, SpriteAnimation, SpriteAtlas
from engine.math.vector2 import Vector2
from engine.scene.game_object import GameObject

# 1. PWC for SpriteAnimation.update
# is_playing (True, False), loop (True, False), indices ([], [0, 1])
# frame_timer reaching duration (True, False)
@pytest.mark.parametrize("is_playing, loop, has_indices, reaches_duration", [
    (False, True, True, False),
    (True, False, False, False),
    (True, True, True, True),
    (False, False, False, True),
    (True, False, True, False),
])
def test_sprite_animation_update_pwc(is_playing, loop, has_indices, reaches_duration):
    indices = [0, 1] if has_indices else []
    anim = SpriteAnimation("test", indices, frame_duration=0.1, loop=loop)
    
    if is_playing:
        anim.play()
        
    if reaches_duration:
        delta = 0.15
    else:
        delta = 0.05
    
    initial_frame = anim.current_frame
    result = anim.update(delta)
    
    if not is_playing or not has_indices:
        assert anim.frame_timer == (0.0 if not is_playing else delta)
        expected = indices[0] if indices else 0
        assert result == expected
    else:
        if reaches_duration:
            assert anim.frame_timer == 0.0
            if is_playing and has_indices:
                assert anim.current_frame == (initial_frame + 1)
        else:
            assert anim.frame_timer == delta
            assert anim.current_frame == initial_frame

# 2. PWC for Sprite.contains_point
# shape ('circle', 'rectangle'), point_status ('inside', 'outside'), world_scale (1.0, 2.0)
@pytest.mark.parametrize("shape, point_status, scale", [
    ('circle', 'inside', 1.0),
    ('rectangle', 'outside', 1.0),
    ('circle', 'outside', 2.0),
    ('rectangle', 'inside', 2.0),
])
def test_sprite_contains_point_pwc(shape, point_status, scale):
    sprite = Sprite(size=Vector2(100, 100), shape=shape)
    go = GameObject()
    go.transform.scale = Vector2(scale, scale)
    go.transform.position = Vector2(0, 0)
    sprite.game_object = go
    
    # Calculate point 
    if point_status == 'inside':
        point = Vector2(10, 10)  # Always inside for scale >= 1.0 and size 100
    else:
        point = Vector2(250, 250) # Always outside for scale <= 2.0 and size 100
    
    result = sprite.contains_point(point)
    expected = (point_status == 'inside')
    assert result == expected


# 3. ECC for Sprite limits
@pytest.mark.parametrize("alpha_in, alpha_exp, bright_in, bright_exp, cont_in, cont_exp", [
    (-0.5, 0.0, -1.0, 0.0, -1.0, 0.0),
    (0.5, 0.5, 1.0, 1.0, 1.0, 1.0),
    (1.5, 1.0, 2.5, 2.0, 2.5, 2.0),
])
def test_sprite_limits_ecc(alpha_in, alpha_exp, bright_in, bright_exp, cont_in, cont_exp):
    sprite = Sprite()
    sprite.set_alpha(alpha_in)
    sprite.set_brightness(bright_in)
    sprite.set_contrast(cont_in)
    
    assert sprite.alpha == alpha_exp
    assert sprite.brightness == bright_exp
    assert sprite.contrast == cont_exp

# 4. PWC for Sprite.render
# visible (True, False), has_game_object (True, False), has_atlas (True, False), shape ('rectangle', 'circle', 'triangle')
@pytest.mark.parametrize("visible, has_game_object, has_atlas, shape", [
    (False, True, False, 'rectangle'),
    (True, False, False, 'circle'),
    (True, True, True, 'triangle'),    # Tests visible + game object + atlas + triangle
    (False, False, True, 'circle'),
    (True, True, False, 'circle'),     # Tests visible + game object + no atlas + circle
    (True, True, False, 'rectangle')   # Tests visible + game object + no atlas + rectangle
])
def test_sprite_render_pwc(visible, has_game_object, has_atlas, shape):
    sprite = Sprite(shape=shape)
    sprite.visible = visible
    
    if has_game_object:
        go = GameObject()
        sprite.game_object = go
        
    renderer = MagicMock()
    
    if has_atlas:
        atlas = SpriteAtlas(Vector2(256, 256))
        atlas.add_sprite("sprite_a", Vector2(0, 0), Vector2(32, 32))
        sprite.set_sprite_atlas(atlas, "sprite_a")
    
    sprite.render(renderer)
    
    if not visible or not has_game_object:
        # Should return early
        renderer.draw_rectangle.assert_not_called()
        renderer.draw_circle.assert_not_called()
        renderer.draw_polygon.assert_not_called()
    else:
        # Should render based on shape
        if shape == 'circle':
            renderer.draw_circle.assert_called_once()
            renderer.draw_rectangle.assert_not_called()
            renderer.draw_polygon.assert_not_called()
        elif shape == 'triangle':
            renderer.draw_polygon.assert_called_once()
            renderer.draw_rectangle.assert_not_called()
            renderer.draw_circle.assert_not_called()
        else:
            renderer.draw_rectangle.assert_called_once()
            renderer.draw_circle.assert_not_called()
            renderer.draw_polygon.assert_not_called()
            
            # Verify sizes if atlas was used or not (only for rectangle here to keep it simple)
            args, kwargs = renderer.draw_rectangle.call_args
            if has_atlas:
                assert args[1] == Vector2(32, 32)
            else:
                assert args[1] == sprite.size
