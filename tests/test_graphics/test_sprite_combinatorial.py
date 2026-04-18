"""
Combinatorial testing for the Sprite module.
Applies Each Choice Coverage (ECC) to systematically test SpriteAtlas.create_animation_frames
without combinatorial explosion.
"""
import pytest
from engine.graphics.sprite import SpriteAtlas
from engine.math.vector2 import Vector2

# Partitions identified (Building blocks for ECC):
# base_name: ["walk" (Normal), "" (Empty)]
# frame_count: [5 (Multiple), 1 (Single), 0 (Zero)]
# frame_size: [Vector2(32, 32) (Normal), Vector2(0, 0) (Zero boundary)]
# start_position: [Vector2(0, 0) (Origin), Vector2(100, 100) (Offset)]
# horizontal: [True, False]

@pytest.mark.parametrize(
    "base_name, frame_count, frame_size, start_position, horizontal, expected_len",
    [
        # Combination 1: Covers Normal Name, Multiple Frames, Normal Size, Origin Start, Horizontal
        ("walk", 5, Vector2(32, 32), Vector2(0, 0), True, 5),
        
        # Combination 2: Covers Empty Name, Single Frame, Zero Size, Offset Start, Vertical
        ("", 1, Vector2(0, 0), Vector2(100, 100), False, 1),
        
        # Combination 3: Covers Zero Frames. Re-uses Normal Name, Normal Size, Origin Start, Horizontal
        ("run", 0, Vector2(16, 16), Vector2(0, 0), True, 0),
    ]
)
def test_create_animation_frames_ecc(base_name, frame_count, frame_size, start_position, horizontal, expected_len):
    """
    Combinatorial testing: Each Choice Coverage (ECC) for SpriteAtlas.create_animation_frames.
    Ensures every parameter partition is executed at least once across 3 minimal test cases.
    """
    atlas = SpriteAtlas(Vector2(512, 512))
    
    # Execute the method under test
    generated_names = atlas.create_animation_frames(
        base_name, frame_count, frame_size, start_position, horizontal
    )
    
    # 1. Validate the correct number of frames was generated
    assert len(generated_names) == expected_len
    
    # 2. Validate internal atlas state matches expected output
    assert len(atlas.sprites) == expected_len
    
    if expected_len > 0:
        # Validate the naming convention is adhered to
        first_frame_name = f"{base_name}_frame_0"
        assert first_frame_name in generated_names
        
        # Validate positioning logic
        first_frame_data = atlas.get_sprite_data(first_frame_name)
        assert first_frame_data['position'] == start_position
        
        if expected_len > 1:
            second_frame_name = f"{base_name}_frame_1"
            second_frame_data = atlas.get_sprite_data(second_frame_name)
            
            # Ensure the orientation (horizontal vs vertical) stepped the coordinates correctly
            if horizontal:
                assert second_frame_data['position'].x == start_position.x + frame_size.x
                assert second_frame_data['position'].y == start_position.y
            else:
                assert second_frame_data['position'].x == start_position.x
                assert second_frame_data['position'].y == start_position.y + frame_size.y