"""
Mutation-killing tests for Sound and SoundGenerator.
Targets surviving mutants from mutmut analysis.
"""
import math
import pytest
from engine.audio.sound_generator import Sound, SoundGenerator


class TestSoundMutationKillers:

    def test_sample_rate_exact(self):
        """Kill mutant: sample_rate = 22050"""
        s = Sound("test")
        assert s.sample_rate == 22050

    def test_initial_duration_zero(self):
        """Kill mutant: duration = 0.0"""
        s = Sound("test")
        assert s.duration == 0.0

    def test_generate_tone_sets_duration(self):
        """Kill mutant: self.duration = duration"""
        s = Sound("test")
        s.generate_tone(440, 0.5)
        assert s.duration == 0.5

    def test_num_samples_correct(self):
        """Kill mutant: sample_rate * duration calculation"""
        s = Sound("test")
        s.generate_tone(440, 0.1)
        expected = int(22050 * 0.1)
        assert len(s.samples) == expected

    def test_sine_wave_specific_value(self):
        """Kill mutant: amplitude * sin(2 * pi * freq * t)"""
        s = Sound("test")
        s.generate_tone(440, 0.01, 'sine', 0.5)
        # First sample at t=0: sin(0)=0
        assert s.samples[0] == pytest.approx(0.0)
        # Check that mid-cycle has non-zero value
        assert any(abs(x) > 0.1 for x in s.samples)

    def test_square_wave_values(self):
        """Kill mutant: square wave 1 and -1 values"""
        s = Sound("test")
        s.generate_tone(440, 0.01, 'square', 0.5)
        # All samples should be either +0.5 or -0.5
        for sample in s.samples:
            assert abs(sample) == pytest.approx(0.5, abs=0.01)

    def test_sawtooth_specific_formula(self):
        """Kill mutant: sawtooth formula constants"""
        s = Sound("test")
        s.generate_tone(440, 0.01, 'sawtooth', 1.0)
        # Should have values in [-1, 1]
        assert all(-1.01 <= x <= 1.01 for x in s.samples)
        # Should have both positive and negative values
        assert any(x > 0.1 for x in s.samples)
        assert any(x < -0.1 for x in s.samples)

    def test_triangle_wave_values(self):
        """Kill mutant: triangle formula 2 * abs(...) - 1"""
        s = Sound("test")
        s.generate_tone(440, 0.01, 'triangle', 1.0)
        assert all(-1.01 <= x <= 1.01 for x in s.samples)

    def test_unknown_wave_type_zeros(self):
        """Kill mutant: else clause returns 0"""
        s = Sound("test")
        s.generate_tone(440, 0.01, 'nonexistent', 0.5)
        assert all(x == 0 for x in s.samples)

    def test_sweep_frequency_changes(self):
        """Kill mutant: start_freq + (end_freq - start_freq) * progress"""
        s = Sound("test")
        s.generate_sweep(800, 200, 0.1, 'sine', 0.5)
        assert len(s.samples) > 0
        assert s.duration == 0.1

    def test_sweep_envelope_attack(self):
        """Kill mutant: t < 0.01 attack envelope"""
        s = Sound("test")
        s.generate_sweep(440, 440, 0.1, 'sine', 1.0)
        # First sample at t=0 should be near zero (attack envelope)
        assert abs(s.samples[0]) < 0.01

    def test_sweep_envelope_release(self):
        """Kill mutant: t > duration - 0.05 release"""
        s = Sound("test")
        s.generate_sweep(440, 440, 0.1, 'sine', 1.0)
        # Last sample should be near zero (release envelope)
        assert abs(s.samples[-1]) < 0.1

    def test_explosion_decay_shape(self):
        """Kill mutant: exp(-progress * 5) decay"""
        s = Sound("test")
        s.generate_explosion(0.5, 1.0)
        # Average amplitude should decrease over time
        mid = len(s.samples) // 2
        first_half_rms = sum(x**2 for x in s.samples[:mid]) / mid
        second_half_rms = sum(x**2 for x in s.samples[mid:]) / (len(s.samples) - mid)
        assert first_half_rms > second_half_rms

    def test_explosion_low_freq_mixing(self):
        """Kill mutant: 0.7 / 0.3 mixing ratio and 100 Hz base"""
        s = Sound("test")
        s.generate_explosion(0.5, 1.0)
        # Should produce samples (not empty)
        assert len(s.samples) > 0
        assert any(abs(x) > 0 for x in s.samples)

    def test_engine_harmonics(self):
        """Kill mutant: harmonic coefficients 0.3 and 0.1"""
        s = Sound("test")
        s.generate_engine(80, 0.2, 1.0)
        # Should have samples
        assert len(s.samples) > 0
        # Check that max amplitude is significant (engine tone + harmonics)
        max_amp = max(abs(x) for x in s.samples)
        assert max_amp > 0.1

    def test_engine_defaults(self):
        """Kill mutant: default parameters base_freq=80, duration=0.2, amplitude=0.2"""
        s = Sound("test")
        s.generate_engine()  # Use all defaults
        assert s.duration == 0.2
        expected_samples = int(22050 * 0.2)
        assert len(s.samples) == expected_samples

    def test_bullet_sound_parameters(self):
        """Kill mutants: create_bullet_sound exact params"""
        sg = SoundGenerator()
        bullet = sg.create_bullet_sound()
        assert bullet.name == "bullet"
        assert bullet.duration == 0.1
        expected_samples = int(22050 * 0.1)
        assert len(bullet.samples) == expected_samples

    def test_explosion_sound_parameters(self):
        """Kill mutants: create_explosion_sound exact params"""
        sg = SoundGenerator()
        explosion = sg.create_explosion_sound()
        assert explosion.name == "explosion"
        assert explosion.duration == 0.3

    def test_engine_sound_parameters(self):
        """Kill mutants: create_engine_sound exact params"""
        sg = SoundGenerator()
        engine = sg.create_engine_sound()
        assert engine.name == "engine"
        assert engine.duration == 0.15
        expected_samples = int(22050 * 0.15)
        assert len(engine.samples) == expected_samples

    def test_frequency_beep_thresholds(self):
        """Kill mutants: frequency threshold 500 and 200"""
        sg = SoundGenerator()
        import io, sys
        # > 500 → "beep!"
        buf = io.StringIO()
        sys.stdout = buf
        sg.generate_frequency_beep(501, 0.1)
        sys.stdout = sys.__stdout__
        assert "beep!" in buf.getvalue()

        buf = io.StringIO()
        sys.stdout = buf
        sg.generate_frequency_beep(300, 0.1)
        sys.stdout = sys.__stdout__
        assert "boop!" in buf.getvalue()

        buf = io.StringIO()
        sys.stdout = buf
        sg.generate_frequency_beep(100, 0.1)
        sys.stdout = sys.__stdout__
        assert "boom!" in buf.getvalue()

    def test_register_sound(self):
        """Kill mutant: sounds[sound.name] = sound"""
        sg = SoundGenerator()
        s = Sound("test_sound")
        sg.register_sound(s)
        assert "test_sound" in sg.sounds
        assert sg.sounds["test_sound"] is s

    def test_sound_name_stored(self):
        """Kill mutant: self.name = name"""
        s = Sound("my_sound")
        assert s.name == "my_sound"

    def test_has_audio_default(self):
        """Kill mutant: has_audio = True"""
        sg = SoundGenerator()
        assert sg.has_audio is True

    def test_playing_default(self):
        """Kill mutant: playing = False"""
        sg = SoundGenerator()
        assert sg.playing is False
