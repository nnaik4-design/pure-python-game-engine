"""
Whitebox tests for Sound/SoundGenerator - targeting uncovered branches.
Focus on play_sound paths and initialize_default_sounds.
"""
import pytest
from unittest.mock import patch
from engine.audio.sound_generator import Sound, SoundGenerator


class TestSoundGeneratorPlaySound:
    """Whitebox: cover play_sound branches (lines 174-197)"""

    def test_play_registered_bullet_sound(self):
        """Branch: play registered bullet sound triggers play_pattern"""
        sg = SoundGenerator()
        sound = sg.create_bullet_sound()
        sg.register_sound(sound)
        sg.play_sound("bullet")  # Should not crash

    def test_play_registered_explosion_sound(self):
        """Branch: play registered explosion sound"""
        sg = SoundGenerator()
        sound = sg.create_explosion_sound()
        sg.register_sound(sound)
        sg.play_sound("explosion")  # Should not crash
        # Wait for thread to finish
        if sg.current_thread:
            sg.current_thread.join(timeout=2)

    def test_play_registered_engine_sound(self):
        """Branch: play registered engine sound"""
        sg = SoundGenerator()
        sound = sg.create_engine_sound()
        sg.register_sound(sound)
        sg.play_sound("engine")  # Should not crash
        if sg.current_thread:
            sg.current_thread.join(timeout=2)

    def test_play_custom_named_sound(self):
        """Branch: play custom-named sound (else branch in play_pattern)"""
        sg = SoundGenerator()
        sound = Sound("custom")
        sound.generate_tone(440, 0.1)
        sg.register_sound(sound)
        sg.play_sound("custom")
        if sg.current_thread:
            sg.current_thread.join(timeout=2)


class TestInitializeDefaultSounds:
    """Whitebox: cover initialize_default_sounds (lines 214-226)"""

    def test_initialize_default_sounds(self):
        """Branch: creates and registers bullet, explosion, engine sounds"""
        sg = SoundGenerator()
        sg.initialize_default_sounds()
        assert "bullet" in sg.sounds
        assert "explosion" in sg.sounds
        assert "engine" in sg.sounds
        assert len(sg.sounds["bullet"].samples) > 0
        assert len(sg.sounds["explosion"].samples) > 0
        assert len(sg.sounds["engine"].samples) > 0
