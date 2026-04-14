"""
Blackbox tests for Sound and SoundGenerator classes.
Tests use Equivalence Partitioning (EP), Boundary Analysis (BA), and Error Guessing (EG).
"""
import math
import pytest
from engine.audio.sound_generator import Sound, SoundGenerator


# ================================================================
# Sound Constructor Tests
# ================================================================

class TestSoundConstructor:

    def test_constructor(self):
        """EP: Sound initializes with name and defaults"""
        s = Sound("test")
        assert s.name == "test"
        assert s.samples == []
        assert s.sample_rate == 22050
        assert s.duration == 0.0


# ================================================================
# Tone Generation Tests
# ================================================================

class TestSoundToneGeneration:

    def test_generate_sine_tone(self):
        """EP: Sine wave generates correct number of samples"""
        s = Sound("sine")
        s.generate_tone(440, 1.0, 'sine', 0.5)
        assert s.duration == 1.0
        assert len(s.samples) == 22050  # sample_rate * duration

    def test_generate_square_tone(self):
        """EP: Square wave generates samples"""
        s = Sound("square")
        s.generate_tone(440, 0.5, 'square', 0.5)
        assert len(s.samples) == int(22050 * 0.5)
        # Square wave samples should be +amplitude or -amplitude
        for sample in s.samples:
            assert abs(sample) == pytest.approx(0.5, abs=0.01)

    def test_generate_sawtooth_tone(self):
        """EP: Sawtooth wave generates samples"""
        s = Sound("saw")
        s.generate_tone(440, 0.1, 'sawtooth', 0.5)
        assert len(s.samples) > 0

    def test_generate_triangle_tone(self):
        """EP: Triangle wave generates samples"""
        s = Sound("tri")
        s.generate_tone(440, 0.1, 'triangle', 0.5)
        assert len(s.samples) > 0

    def test_generate_noise(self):
        """EP: Noise generates random samples within amplitude range"""
        s = Sound("noise")
        s.generate_tone(440, 0.1, 'noise', 0.5)
        assert len(s.samples) > 0
        for sample in s.samples:
            assert -0.5 <= sample <= 0.5

    def test_unknown_wave_type_zero_samples(self):
        """BA: Unknown wave type produces zero-valued samples"""
        s = Sound("unknown")
        s.generate_tone(440, 0.1, 'nonexistent', 0.5)
        assert all(sample == 0 for sample in s.samples)

    def test_sine_amplitude(self):
        """BA: Sine wave samples bounded by amplitude"""
        s = Sound("sine")
        s.generate_tone(440, 0.5, 'sine', 0.3)
        for sample in s.samples:
            assert -0.3 <= sample <= 0.3 + 1e-9

    def test_zero_duration(self):
        """BA: Zero duration produces no samples"""
        s = Sound("zero")
        s.generate_tone(440, 0.0, 'sine', 0.5)
        assert len(s.samples) == 0

    def test_very_short_duration(self):
        """BA: Very short duration produces at least some samples"""
        s = Sound("short")
        s.generate_tone(440, 0.001, 'sine', 0.5)
        expected = int(22050 * 0.001)
        assert len(s.samples) == expected

    def test_high_frequency(self):
        """EP: High frequency generates without error"""
        s = Sound("high")
        s.generate_tone(10000, 0.1, 'sine', 0.5)
        assert len(s.samples) > 0

    def test_low_frequency(self):
        """EP: Low frequency generates without error"""
        s = Sound("low")
        s.generate_tone(20, 0.1, 'sine', 0.5)
        assert len(s.samples) > 0

    def test_zero_amplitude(self):
        """BA: Zero amplitude produces all zero samples"""
        s = Sound("silent")
        s.generate_tone(440, 0.1, 'sine', 0.0)
        for sample in s.samples:
            assert sample == pytest.approx(0.0, abs=1e-10)


# ================================================================
# Sweep Generation Tests
# ================================================================

class TestSoundSweep:

    def test_generate_sweep(self):
        """EP: Sweep generates correct number of samples"""
        s = Sound("sweep")
        s.generate_sweep(800, 200, 0.5, 'sine', 0.5)
        assert s.duration == 0.5
        assert len(s.samples) == int(22050 * 0.5)

    def test_sweep_square(self):
        """EP: Square wave sweep generates samples"""
        s = Sound("sq_sweep")
        s.generate_sweep(500, 100, 0.2, 'square', 0.3)
        assert len(s.samples) > 0

    def test_sweep_unknown_type_defaults_to_sine(self):
        """EG: Unknown sweep wave type falls back to sine"""
        s = Sound("fallback")
        s.generate_sweep(500, 100, 0.1, 'unknown', 0.5)
        assert len(s.samples) > 0

    def test_sweep_envelope_applied(self):
        """EP: Sweep applies envelope (first samples near zero due to attack)"""
        s = Sound("env")
        s.generate_sweep(800, 200, 0.5, 'sine', 0.5)
        # First sample should be near zero due to attack envelope (t < 0.01)
        assert abs(s.samples[0]) < 0.01


# ================================================================
# Explosion Generation Tests
# ================================================================

class TestSoundExplosion:

    def test_generate_explosion(self):
        """EP: Explosion generates samples with decay"""
        s = Sound("boom")
        s.generate_explosion(0.5, 0.3)
        assert s.duration == 0.5
        assert len(s.samples) == int(22050 * 0.5)

    def test_explosion_decays(self):
        """EP: Explosion sound decays over time (last samples smaller than first)"""
        s = Sound("boom")
        s.generate_explosion(0.5, 0.3)
        # Compare RMS of first 10% vs last 10%
        n = len(s.samples)
        first_chunk = s.samples[:n // 10]
        last_chunk = s.samples[-(n // 10):]
        rms_first = math.sqrt(sum(x ** 2 for x in first_chunk) / len(first_chunk))
        rms_last = math.sqrt(sum(x ** 2 for x in last_chunk) / len(last_chunk))
        assert rms_last < rms_first

    def test_explosion_default_params(self):
        """EP: Explosion with default parameters works"""
        s = Sound("boom")
        s.generate_explosion()
        assert s.duration == 0.5
        assert len(s.samples) > 0


# ================================================================
# Engine Sound Generation Tests
# ================================================================

class TestSoundEngine:

    def test_generate_engine(self):
        """EP: Engine sound generates samples"""
        s = Sound("engine")
        s.generate_engine(80, 0.2, 0.2)
        assert s.duration == 0.2
        assert len(s.samples) == int(22050 * 0.2)

    def test_engine_default_params(self):
        """EP: Engine with default parameters works"""
        s = Sound("engine")
        s.generate_engine()
        assert s.duration == 0.2
        assert len(s.samples) > 0

    def test_engine_has_harmonics(self):
        """EG: Engine sound is not pure sine (has harmonics + noise)"""
        s = Sound("engine")
        s.generate_engine(80, 0.2, 0.5)
        # With harmonics and noise, samples should have varied magnitudes
        unique_magnitudes = len(set(round(abs(x), 4) for x in s.samples[:100]))
        assert unique_magnitudes > 5  # Not all the same value


# ================================================================
# SoundGenerator Tests
# ================================================================

class TestSoundGenerator:

    def test_constructor(self):
        """EP: SoundGenerator initializes correctly"""
        sg = SoundGenerator()
        assert sg.sounds == {}
        assert sg.playing is False
        assert sg.has_audio is True

    def test_create_bullet_sound(self):
        """EP: Create bullet sound returns valid Sound"""
        sg = SoundGenerator()
        sound = sg.create_bullet_sound()
        assert sound.name == "bullet"
        assert len(sound.samples) > 0

    def test_create_explosion_sound(self):
        """EP: Create explosion sound returns valid Sound"""
        sg = SoundGenerator()
        sound = sg.create_explosion_sound()
        assert sound.name == "explosion"
        assert len(sound.samples) > 0

    def test_create_engine_sound(self):
        """EP: Create engine sound returns valid Sound"""
        sg = SoundGenerator()
        sound = sg.create_engine_sound()
        assert sound.name == "engine"
        assert len(sound.samples) > 0

    def test_register_sound(self):
        """EP: Register a sound for later retrieval"""
        sg = SoundGenerator()
        sound = Sound("custom")
        sg.register_sound(sound)
        assert "custom" in sg.sounds
        assert sg.sounds["custom"] is sound

    def test_register_overwrites(self):
        """EG: Registering same name overwrites"""
        sg = SoundGenerator()
        s1 = Sound("test")
        s2 = Sound("test")
        sg.register_sound(s1)
        sg.register_sound(s2)
        assert sg.sounds["test"] is s2

    def test_generate_frequency_beep_high(self):
        """EP: High frequency produces 'beep!' pattern"""
        sg = SoundGenerator()
        # Just verify it doesn't crash - output goes to stdout
        sg.generate_frequency_beep(600, 0.1)

    def test_generate_frequency_beep_mid(self):
        """EP: Mid frequency produces 'boop!' pattern"""
        sg = SoundGenerator()
        sg.generate_frequency_beep(300, 0.1)

    def test_generate_frequency_beep_low(self):
        """EP: Low frequency produces 'boom!' pattern"""
        sg = SoundGenerator()
        sg.generate_frequency_beep(100, 0.1)

    def test_play_unregistered_sound(self):
        """EG: Playing unregistered sound doesn't crash"""
        sg = SoundGenerator()
        sg.play_sound("nonexistent")  # Should not raise


# ================================================================
# Error Guessing
# ================================================================

class TestSoundErrorGuessing:

    def test_regenerate_overwrites_samples(self):
        """EG: Generating again overwrites previous samples"""
        s = Sound("test")
        s.generate_tone(440, 0.5, 'sine', 0.5)
        first_count = len(s.samples)
        s.generate_tone(440, 1.0, 'sine', 0.5)
        assert len(s.samples) == int(22050 * 1.0)
        assert len(s.samples) != first_count

    def test_negative_frequency(self):
        """EG: Negative frequency doesn't crash (math handles it)"""
        s = Sound("neg")
        s.generate_tone(-440, 0.1, 'sine', 0.5)
        assert len(s.samples) > 0

    def test_very_high_amplitude(self):
        """EG: High amplitude doesn't crash"""
        s = Sound("loud")
        s.generate_tone(440, 0.1, 'sine', 100.0)
        assert len(s.samples) > 0
        # Samples should reach up to amplitude
        max_sample = max(abs(x) for x in s.samples)
        assert max_sample <= 100.0 + 1e-6
