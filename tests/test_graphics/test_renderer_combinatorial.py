import pytest
import math
from unittest.mock import MagicMock
from itertools import product
from allpairspy import AllPairs
from engine.graphics.renderer import Renderer
from engine.math.vector2 import Vector2

@pytest.fixture
def renderer():
    canvas = MagicMock()
    canvas.__getitem__.side_effect = lambda k: 800 if k == 'width' else 600
    return Renderer(canvas)

# 1. PWC for draw_rectangle with position, size, and rotation
# Parameters: rotation (0, 45, 90), position variations, size variations
_rect_parameters = [
    [0, 45, 90],  # rotation
    [(0, 0), (100, 100), (400, 300)],  # position (as tuples for serialization)
    [(20, 20), (50, 50), (100, 80)],  # size (width, height)
    [None, '#FF0000'],  # outline
    [1, 3],  # width
]
_rect_pwc_cases = [tuple(case) for case in AllPairs(_rect_parameters)]

@pytest.mark.parametrize("rotation, pos_tuple, size_tuple, outline, width", _rect_pwc_cases)
def test_draw_rectangle_pwc(renderer, rotation, pos_tuple, size_tuple, outline, width):
    """Test rectangle drawing with geometric validation"""
    position = Vector2(pos_tuple[0], pos_tuple[1])
    size = Vector2(size_tuple[0], size_tuple[1])
    color = "#00FF00"

    renderer.draw_rectangle(
        position=position,
        size=size,
        color=color,
        rotation=rotation,
        outline=outline,
        width=width
    )

    # Verify correct method was called
    if rotation == 0:
        renderer.canvas.create_rectangle.assert_called_once()
        args, kwargs = renderer.canvas.create_rectangle.call_args

        # Validate coordinates are calculated correctly for non-rotated rectangle
        x1, y1, x2, y2 = args[:4]
        expected_x1 = position.x - size.x / 2
        expected_y1 = position.y - size.y / 2
        expected_x2 = position.x + size.x / 2
        expected_y2 = position.y + size.y / 2

        assert abs(x1 - expected_x1) < 1e-6, f"x1 mismatch: {x1} vs {expected_x1}"
        assert abs(y1 - expected_y1) < 1e-6, f"y1 mismatch: {y1} vs {expected_y1}"
        assert abs(x2 - expected_x2) < 1e-6, f"x2 mismatch: {x2} vs {expected_x2}"
        assert abs(y2 - expected_y2) < 1e-6, f"y2 mismatch: {y2} vs {expected_y2}"

        # Verify parameters
        assert kwargs['fill'] == color
        assert kwargs['outline'] == (outline or color)
        assert kwargs['width'] == width
    else:
        # Rotated rectangle - should use polygon
        renderer.canvas.create_polygon.assert_called_once()
        args, kwargs = renderer.canvas.create_polygon.call_args

        # Validate polygon has 4 corners (8 coordinates for 4 points)
        coords = args[0]
        assert len(coords) == 8, f"Expected 8 coordinates for rectangle, got {len(coords)}"

        # Verify corners form a valid quadrilateral centered at position
        points = [Vector2(coords[i], coords[i+1]) for i in range(0, 8, 2)]

        # Calculate center of polygon
        center = Vector2(
            sum(p.x for p in points) / 4,
            sum(p.y for p in points) / 4
        )
        assert abs(center.x - position.x) < 1e-5, f"Rotated rect center X off: {center.x} vs {position.x}"
        assert abs(center.y - position.y) < 1e-5, f"Rotated rect center Y off: {center.y} vs {position.y}"

        # All corners should be roughly same distance from center
        distances = [p.distance_to(center) for p in points]
        expected_dist = math.sqrt((size.x/2)**2 + (size.y/2)**2)
        for dist in distances:
            assert abs(dist - expected_dist) < 1e-4, f"Corner distance {dist} doesn't match expected {expected_dist}"

        assert kwargs['fill'] == color
        assert kwargs['outline'] == (outline or color)
        assert kwargs['width'] == width

# 2. PWC for draw_circle with position, radius, and color validation
# Parameters: radius (10, 50, 100), position (edge, center, custom), outline (None, color)
_circle_parameters = [
    [10, 50, 100],  # radius
    [(50, 50), (400, 300)],  # position
    [None, '#FF0000'],  # outline
    [1, 2],  # width
]
_circle_pwc_cases = [tuple(case) for case in AllPairs(_circle_parameters)]

@pytest.mark.parametrize("radius, pos_tuple, outline, width", _circle_pwc_cases)
def test_draw_circle_pwc(renderer, radius, pos_tuple, outline, width):
    """Test circle drawing with geometric validation"""
    position = Vector2(pos_tuple[0], pos_tuple[1])
    color = "#0000FF"

    renderer.draw_circle(
        position=position,
        radius=radius,
        color=color,
        outline=outline,
        width=width
    )

    renderer.canvas.create_oval.assert_called_once()
    args, kwargs = renderer.canvas.create_oval.call_args

    # Validate bounding box coordinates
    x1, y1, x2, y2 = args[:4]
    expected_x1 = position.x - radius
    expected_y1 = position.y - radius
    expected_x2 = position.x + radius
    expected_y2 = position.y + radius

    assert abs(x1 - expected_x1) < 1e-6
    assert abs(y1 - expected_y1) < 1e-6
    assert abs(x2 - expected_x2) < 1e-6
    assert abs(y2 - expected_y2) < 1e-6

    # Verify width of bounding box is 2*radius
    assert abs((x2 - x1) - 2 * radius) < 1e-6
    assert abs((y2 - y1) - 2 * radius) < 1e-6

    assert kwargs['fill'] == color
    assert kwargs['outline'] == (outline or color)
    assert kwargs['width'] == width

# 3. ACoC for set_shader - test shader management with multiple shaders
# Parameters: enabled (True, False), shader_in_active (True, False)
_shader_enabled = [True, False]
_shader_in_active = [True, False]
_shader_acoc_cases = [
    (enabled, active, enabled)  # expected_in_active_after equals the enabled flag
    for enabled, active in product(_shader_enabled, _shader_in_active)
]

@pytest.mark.parametrize("enabled, shader_in_active_before, expected_in_active_after", _shader_acoc_cases)
def test_set_shader_acoc(renderer, enabled, shader_in_active_before, expected_in_active_after):
    """Test shader management with state verification"""
    shader_name = "test_shader"
    if shader_in_active_before:
        renderer.active_shaders.append(shader_name)

    renderer.set_shader(shader_name, enabled=enabled)

    is_active = shader_name in renderer.active_shaders
    assert is_active == expected_in_active_after

    # Verify no duplicates when enabling multiple times
    renderer.set_shader(shader_name, enabled=True)
    assert renderer.active_shaders.count(shader_name) <= 1, "Shader should not be duplicated"

    # Verify disabling removes it completely
    renderer.set_shader(shader_name, enabled=False)
    assert shader_name not in renderer.active_shaders


# 4. PWC for set_shader_uniform and post-processing
# Parameters: multiple shaders, different uniform values
_uniform_parameters = [
    ['color', 'intensity', 'offset'],  # uniform names
    [0.0, 0.5, 1.0],  # uniform values
    [True, False],  # has_post_effect
]
_uniform_pwc_cases = [tuple(case) for case in AllPairs(_uniform_parameters)]

@pytest.mark.parametrize("uniform_name, uniform_value, has_effect", _uniform_pwc_cases)
def test_shader_uniforms_pwc(renderer, uniform_name, uniform_value, has_effect):
    """Test shader uniform setting and post-processing effects"""
    renderer.set_shader_uniform(uniform_name, uniform_value)

    assert uniform_name in renderer.shader_uniforms
    assert renderer.shader_uniforms[uniform_name] == uniform_value

    if has_effect:
        renderer.add_post_processing_effect("bloom")
        assert "bloom" in renderer.post_processing_effects

        # Verify we can set multiple uniforms
        renderer.set_shader_uniform("other", 0.8)
        assert len(renderer.shader_uniforms) >= 2

# 5. PWC for draw_sprite_from_atlas with scale and rotation
# Parameters: has_atlas, scale_x, scale_y, position_x, position_y
_atlas_parameters = [
    [True, False],  # has_atlas
    [1.0, 2.0],  # scale
    [(0, 0), (100, 100)],  # position
]
_atlas_pwc_cases = [tuple(case) for case in AllPairs(_atlas_parameters)]

@pytest.mark.parametrize("has_atlas, scale, pos_tuple", _atlas_pwc_cases)
def test_draw_sprite_from_atlas_pwc(renderer, has_atlas, scale, pos_tuple):
    """Test sprite atlas rendering with geometric validation"""
    position = Vector2(pos_tuple[0], pos_tuple[1])
    atlas_data = {
        'size': Vector2(32, 32),
        'color': '#FFFFFF'
    } if has_atlas else None

    scale_vec = Vector2(scale, scale)

    result = renderer.draw_sprite_from_atlas(position, atlas_data=atlas_data, scale=scale_vec)

    if not has_atlas:
        assert result is None
        renderer.canvas.create_rectangle.assert_not_called()
    else:
        renderer.canvas.create_rectangle.assert_called_once()
        args, kwargs = renderer.canvas.create_rectangle.call_args

        # Validate position and size calculations
        x1, y1, x2, y2 = args[:4]

        expected_size_x = 32 * scale
        expected_size_y = 32 * scale
        expected_x1 = position.x - expected_size_x / 2
        expected_y1 = position.y - expected_size_y / 2
        expected_x2 = position.x + expected_size_x / 2
        expected_y2 = position.y + expected_size_y / 2

        assert abs(x1 - expected_x1) < 1e-4, f"X1 mismatch: {x1} vs {expected_x1}"
        assert abs(y1 - expected_y1) < 1e-4, f"Y1 mismatch: {y1} vs {expected_y1}"
        assert abs(x2 - expected_x2) < 1e-4, f"X2 mismatch: {x2} vs {expected_x2}"
        assert abs(y2 - expected_y2) < 1e-4, f"Y2 mismatch: {y2} vs {expected_y2}"

        assert kwargs['fill'] == '#FFFFFF'


# 6. PWC for post-processing effects
# Parameters: bloom (T/F), blur (T/F), vintage (T/F)
_effects_parameters = [
    [True, False],  # bloom
    [True, False],  # blur
    [True, False],  # vintage
]
_effects_pwc_cases = [tuple(case) for case in AllPairs(_effects_parameters)]
_effects_test_cases = []
_effect_names = ['bloom', 'blur', 'vintage']
for case in _effects_pwc_cases:
    effects = [name for name, present in zip(_effect_names, case) if present]
    _effects_test_cases.append((effects,))

@pytest.mark.parametrize("effects", _effects_test_cases)
def test_apply_post_processing_pwc(renderer, effects):
    """Test post-processing effect management and ordering"""
    renderer.post_processing_effects = effects

    renderer.apply_post_processing()

    # Verify effects list is preserved
    for effect in effects:
        assert effect in renderer.post_processing_effects
