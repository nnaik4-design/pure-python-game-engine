import pytest
from unittest.mock import MagicMock
from engine.graphics.renderer import Renderer
from engine.math.vector2 import Vector2

@pytest.fixture
def renderer():
    canvas = MagicMock()
    canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
    return Renderer(canvas)

# 1. PWC for draw_rectangle
# Parameters: rotation (0, 45), outline (None, '#FF0000'), width (1, 5)
# Using Orthogonal Array / Pairwise combinations manually
@pytest.mark.parametrize("rotation, outline, width", [
    (0, None, 1),
    (0, '#FF0000', 5),
    (45, None, 5),
    (45, '#FF0000', 1),
])
def test_draw_rectangle_pwc(renderer, rotation, outline, width):
    renderer.draw_rectangle(
        position=Vector2(100, 100),
        size=Vector2(50, 50),
        color="#00FF00",
        rotation=rotation,
        outline=outline,
        width=width
    )
    if rotation == 0:
        renderer.canvas.create_rectangle.assert_called_once()
        args, kwargs = renderer.canvas.create_rectangle.call_args
        assert kwargs['outline'] == (outline or '#00FF00')
        assert kwargs['width'] == width
    else:
        renderer.canvas.create_polygon.assert_called_once()
        args, kwargs = renderer.canvas.create_polygon.call_args
        assert kwargs['outline'] == (outline or '#00FF00')
        assert kwargs['width'] == width

# 2. ACoC for set_shader
# Parameters: enabled (True, False), shader_in_active (True, False)
@pytest.mark.parametrize("enabled, shader_in_active_before, expected_in_active_after", [
    (True, True, True),
    (True, False, True),
    (False, True, False),
    (False, False, False),
])
def test_set_shader_acoc(renderer, enabled, shader_in_active_before, expected_in_active_after):
    shader_name = "test_shader"
    if shader_in_active_before:
        renderer.active_shaders.append(shader_name)
        
    renderer.set_shader(shader_name, enabled=enabled)
    
    is_active = shader_name in renderer.active_shaders
    assert is_active == expected_in_active_after

# 3. ECC for draw_sprite_from_atlas
# Parameters: atlas_data (None, Valid), scale (None, Valid)
@pytest.mark.parametrize("has_atlas, has_scale", [
    (False, False),
    (True, False),
    (True, True),
])
def test_draw_sprite_from_atlas_ecc(renderer, has_atlas, has_scale):
    atlas_data = {
        'size': Vector2(32, 32),
        'color': '#FFFFFF'
    } if has_atlas else None
    
    scale = Vector2(2, 2) if has_scale else None
    
    result = renderer.draw_sprite_from_atlas(Vector2(0, 0), atlas_data=atlas_data, scale=scale)
    
    if not has_atlas:
        assert result is None
        renderer.canvas.create_rectangle.assert_not_called()
    else:
        renderer.canvas.create_rectangle.assert_called_once()
        args, kwargs = renderer.canvas.create_rectangle.call_args
        if has_scale:
            # Check scaled rectangle bounds (x2-x1 = 64)
            assert args[2] - args[0] == 64
        else:
            # Check normal size bounds (x2-x1 = 32)
            assert args[2] - args[0] == 32

# 4. PWC for apply_post_processing
@pytest.mark.parametrize("effects", [
    (["bloom"]),
    (["blur", "vintage"]),
    (["bloom", "unknown_effect"]),
    ([])
])
def test_apply_post_processing_pwc(renderer, effects):
    renderer.post_processing_effects = effects
    
    # Mock the effect methods to verify they get called
    renderer._apply_bloom_effect = MagicMock()
    renderer._apply_blur_effect = MagicMock()
    renderer._apply_vintage_effect = MagicMock()
    
    renderer.apply_post_processing()
    
    if "bloom" in effects:
        renderer._apply_bloom_effect.assert_called_once()
    else:
        renderer._apply_bloom_effect.assert_not_called()
        
    if "blur" in effects:
        renderer._apply_blur_effect.assert_called_once()
    else:
        renderer._apply_blur_effect.assert_not_called()
        
    if "vintage" in effects:
        renderer._apply_vintage_effect.assert_called_once()
    else:
        renderer._apply_vintage_effect.assert_not_called()
