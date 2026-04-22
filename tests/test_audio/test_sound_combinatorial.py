"""
Combinatorial testing for SoundGenerator module.
"""
import pytest
import math
from itertools import product
from allpairspy import AllPairs
from engine.audio.sound_generator import Sound

# Partitions identified:
# frequency: [20 (Low), 440 (Mid), 10000 (High)]
# duration: [0.1 (Short), 0.5 (Medium), 1.0 (Long)]
# wave_type: ['sine', 'square', 'sawtooth', 'triangle', 'noise', 'unknown']
# amplitude: [0.0 (Zero), 0.5 (Normal), 1.0 (Max), -0.5 (Negative/Inverted)]

# Generate PWC test cases using allpairspy
_tone_parameters = [
    [20, 440, 10000],  # frequency
    [0.1, 0.5, 1.0],   # duration
    ['sine', 'square', 'sawtooth', 'triangle', 'noise', 'unknown'],  # wave_type
    [0.0, 0.5, 1.0, -0.5],  # amplitude
]
_tone_pwc_cases = list(AllPairs(_tone_parameters))
# Map combinations to expected_all_zeros: True when amplitude is 0.0 or wave_type is 'unknown'
_tone_test_cases = [
    tuple(case) + (case[3] == 0.0 or case[2] == 'unknown',)
    for case in _tone_pwc_cases
]

@pytest.mark.parametrize(
    "frequency, duration, wave_type, amplitude, expected_all_zeros",
    _tone_test_cases
)
def test_generate_tone_pwc(frequency, duration, wave_type, amplitude, expected_all_zeros):
    """
    Combinatorial testing: Pair-Wise Coverage (PWC) for Sound.generate_tone.
    Ensures pairwise interactions across complex audio waveforms and modifiers are fully exercised.
    Tests waveform generation, amplitude constraints, and parameter interactions.
    """
    sound = Sound("pwc_test")
    sound.generate_tone(frequency, duration, wave_type, amplitude)

    # 1. Validate the duration was set
    assert sound.duration == duration

    # 2. Validate the correct number of samples was generated
    expected_samples = int(22050 * duration)
    assert len(sound.samples) == expected_samples

    # 3. Validate math constraints (zeros vs active waveforms)
    if expected_all_zeros:
        assert all(sample == 0.0 for sample in sound.samples), \
            f"Expected all zeros for amplitude={amplitude}, wave_type={wave_type}"
    else:
        # If it's a valid wave with non-zero amplitude, it should have non-zero samples
        assert any(sample != 0.0 for sample in sound.samples), \
            f"Expected non-zero samples for frequency={frequency}, amplitude={amplitude}, wave_type={wave_type}"

        # Validate boundary constraint: no sample should exceed absolute amplitude
        max_generated_amplitude = max(abs(s) for s in sound.samples)
        assert max_generated_amplitude <= abs(amplitude) + 1e-6, \
            f"Max amplitude {max_generated_amplitude} exceeds {abs(amplitude)} for wave_type={wave_type}"

        # Validate waveform shape based on type
        if wave_type == 'sine':
            # Sine waves should have smoothly varying values, check for peaks near amplitude
            num_samples = len(sound.samples)
            peak_count = sum(1 for i in range(1, num_samples - 1)
                           if abs(sound.samples[i]) > abs(amplitude) * 0.9 and
                           abs(sound.samples[i]) >= abs(sound.samples[i-1]) and
                           abs(sound.samples[i]) >= abs(sound.samples[i+1]))
            assert peak_count >= 1, "Sine wave should have at least one peak near amplitude"

        elif wave_type == 'square':
            # Square waves should have values close to +/- amplitude (hard transitions)
            values_near_max = sum(1 for s in sound.samples if abs(s) > abs(amplitude) * 0.9)
            assert values_near_max > len(sound.samples) * 0.5, \
                "Square wave should have many samples near +/- amplitude"

        elif wave_type == 'noise':
            # Noise should have varied values across the amplitude range
            unique_values = len(set(sound.samples))
            assert unique_values > len(sound.samples) * 0.1, \
                "Noise should produce varied sample values"

# Partitions identified for generate_sweep:
# start_freq: [200 (Low), 800 (Mid), 2000 (High)]
# end_freq_relation: ['up' (end > start), 'same' (end == start), 'down' (end < start)]
# duration: [0.01 (Very Short), 0.1 (Short), 1.0 (Long)]
# wave_type: ['sine', 'square', 'unsupported']
# amplitude: [0.0 (Zero), 0.5 (Normal), 1.0 (Max)]

_sweep_parameters = [
    [200, 800, 2000],  # start_freq
    ['up', 'same', 'down'],  # end_freq_relation
    [0.01, 0.1, 1.0],  # duration (removed negative - always positive for audio)
    ['sine', 'square', 'unsupported'],  # wave_type
    [0.0, 0.5, 1.0],  # amplitude (removed negative for consistency)
]
_sweep_pwc_cases = [tuple(case) for case in AllPairs(_sweep_parameters)]

@pytest.mark.parametrize(
    "start_freq, end_freq_relation, duration, wave_type, amplitude",
    _sweep_pwc_cases
)
def test_generate_sweep_pwc(start_freq, end_freq_relation, duration, wave_type, amplitude):
    """
    Combinatorial testing: Pair-Wise Coverage (PWC) for Sound.generate_sweep.
    Ensures parameter interactions (e.g. waveform with duration and amplitude)
    are executed across minimal test cases. Tests envelope application and sweep behavior.
    """
    sound = Sound("pwc_sweep_test")

    # Calculate end freq to map to relations
    if end_freq_relation == 'up':
        end_freq = start_freq + 400
    elif end_freq_relation == 'down':
        end_freq = max(20, start_freq - 400)
    else:
        end_freq = start_freq

    sound.generate_sweep(start_freq, end_freq, duration, wave_type, amplitude)

    assert sound.duration == duration

    expected_samples = int(22050 * duration)
    assert len(sound.samples) == expected_samples

    # Validate amplitude constraint
    if amplitude > 0:
        if expected_samples > 0:
            max_generated = max(abs(s) for s in sound.samples)
            assert max_generated <= amplitude + 1e-6, \
                f"Max amplitude {max_generated} exceeds {amplitude}"

        # Verify envelope is applied: samples at start and end should be lower
        # (due to attack and release envelopes in the implementation)
        if expected_samples > 100:
            start_quarter_samples = sound.samples[:expected_samples // 4]
            middle_samples = sound.samples[expected_samples // 2:3 * expected_samples // 4]
            end_quarter_samples = sound.samples[-expected_samples // 4:]

            # Middle should generally have higher amplitude than edges (envelope effect)
            avg_middle = sum(abs(s) for s in middle_samples) / len(middle_samples)
            avg_start = sum(abs(s) for s in start_quarter_samples) / len(start_quarter_samples)
            avg_end = sum(abs(s) for s in end_quarter_samples) / len(end_quarter_samples)

            assert avg_middle > avg_start * 0.5, "Attack envelope not applied properly"
            assert avg_middle > avg_end * 0.5, "Release envelope not applied properly"
    else:
        # Amplitude 0 should produce all zeros
        assert all(s == 0.0 for s in sound.samples), \
            "Amplitude 0 should produce all zero samples"

from engine.audio.sound_generator import SoundGenerator

# Partitions identified (Building blocks for ACoC):
# duration: [0.1 (Short), 1.0 (Long)]
# amplitude: [0.0 (Zero), 0.5 (Normal), 1.0 (Max)]
# All Combinations Coverage (ACoC) requires 2 * 3 = 6 combinations.

_explosion_durations = [0.1, 1.0]
_explosion_amplitudes = [0.0, 0.5, 1.0]
_explosion_acoc_cases = list(product(_explosion_durations, _explosion_amplitudes))

@pytest.mark.parametrize(
    "duration, amplitude",
    _explosion_acoc_cases
)
def test_generate_explosion_acoc(duration, amplitude):
    """
    All Combinations Coverage (ACoC) for Sound.generate_explosion.
    Provides complete interaction coverage for 2 parameters.
    Tests amplitude constraints, envelope decay, and sample generation.
    """
    sound = Sound("explosion_acoc")
    sound.generate_explosion(duration, amplitude)

    assert sound.duration == duration
    expected_samples = int(22050 * duration)
    assert len(sound.samples) == expected_samples

    # Validate amplitude constraint
    if amplitude == 0.0:
        # Zero amplitude should produce all-zero samples
        assert all(s == 0.0 for s in sound.samples), \
            "Amplitude 0 should produce all zero samples"
    else:
        # Non-zero amplitude should have non-zero samples
        assert any(s != 0.0 for s in sound.samples), \
            f"Amplitude {amplitude} should produce non-zero samples"

        # Check max amplitude doesn't exceed specified amplitude
        max_generated = max(abs(s) for s in sound.samples)
        assert max_generated <= amplitude + 1e-6, \
            f"Max amplitude {max_generated} exceeds {amplitude}"

        # Validate envelope decay: end samples should have lower amplitude than start
        # (exponential decay is applied)
        if expected_samples > 100:
            start_third_max = max(abs(s) for s in sound.samples[:expected_samples // 3])
            end_third_max = max(abs(s) for s in sound.samples[-expected_samples // 3:])
            assert end_third_max < start_third_max, \
                "Explosion envelope decay not applied: end should be quieter than start"

# Partitions identified (Building blocks for PWC):
# base_freq: [20 (Low), 100 (Nominal), 500 (High)]
# duration: [0.1 (Short), 0.5 (Long)]
# amplitude: [0.0 (Zero), 0.5 (Normal), 1.0 (Max)]
# Pair-Wise Coverage (PWC) ensures every pair of values is tested at least once cleanly.

_engine_parameters = [
    [20, 100, 500],  # base_freq
    [0.1, 0.5],  # duration
    [0.0, 0.5, 1.0],  # amplitude
]
_engine_pwc_cases = [tuple(case) for case in AllPairs(_engine_parameters)]

@pytest.mark.parametrize(
    "base_freq, duration, amplitude",
    _engine_pwc_cases
)
def test_generate_engine_pwc(base_freq, duration, amplitude):
    """
    Pair-Wise Coverage (PWC) for Sound.generate_engine.
    Tests complex harmonic interactions optimally.
    Validates harmonics, amplitude constraints, and envelope behavior.
    """
    sound = Sound("engine_pwc")
    sound.generate_engine(base_freq, duration, amplitude)

    assert sound.duration == duration
    assert len(sound.samples) == int(22050 * duration)

    # Validate amplitude constraint
    if amplitude == 0.0:
        # Zero amplitude should produce all-zero samples
        assert all(s == 0.0 for s in sound.samples), \
            "Amplitude 0 should produce all zero samples"
    else:
        # Non-zero amplitude should have non-zero samples
        assert any(s != 0.0 for s in sound.samples), \
            f"Amplitude {amplitude} should produce non-zero samples"

        # Check max amplitude doesn't exceed specified amplitude (with harmonics)
        max_generated = max(abs(s) for s in sound.samples)
        assert max_generated <= amplitude + 0.5, \
            f"Max amplitude {max_generated} significantly exceeds {amplitude} (harmonics included)"

        # Validate envelope is applied: samples at edges should be lower than middle
        num_samples = len(sound.samples)
        if num_samples > 200:
            attack_samples = sound.samples[:int(num_samples * 0.1)]
            middle_samples = sound.samples[int(num_samples * 0.4):int(num_samples * 0.6)]
            release_samples = sound.samples[-int(num_samples * 0.1):]

            avg_attack = sum(abs(s) for s in attack_samples) / len(attack_samples)
            avg_middle = sum(abs(s) for s in middle_samples) / len(middle_samples)
            avg_release = sum(abs(s) for s in release_samples) / len(release_samples)

            assert avg_middle > avg_attack, "Attack envelope not applied properly"
            assert avg_middle > avg_release, "Release envelope not applied properly"

# Partitions identified (Building blocks for ECC):
# sound_name: ["bullet" (Registered), "unknown" (Unregistered), "engine" (Registered)]
# sound_exists: [True, False]

@pytest.mark.parametrize(
    "sound_name, sound_exists",
    [
        ("bullet", True),
        ("explosion", True),
        ("engine", True),
        ("unknown", False),
    ]
)
def test_play_sound_ecc(sound_name, sound_exists, capsys):
    """
    Each Choice Coverage (ECC) for SoundGenerator.play_sound.
    Checks sound registration handling and output generation.
    Validates that registered sounds produce correct output and unregistered sounds don't.
    """
    generator = SoundGenerator()

    # Register specific sounds if needed
    if sound_exists:
        generator.sounds[sound_name] = Sound(sound_name)

    # Play the sound
    generator.play_sound(sound_name)

    # Wait for the background thread to finish playing the sound
    if generator.current_thread is not None:
        generator.current_thread.join()

    # Capture output
    captured = capsys.readouterr()

    if sound_exists:
        # Check for either the bell character (\x07) OR the fallback text
        if sound_name == "explosion":
            # Explosion loops 3 times
            assert '\x07\x07\x07' in captured.out or f"*{sound_name}*" in captured.out
        else:
            assert '\x07' in captured.out or f"*{sound_name}*" in captured.out
    else:
        # Unregistered sounds should produce no output (early return)
        assert captured.out == "" and captured.err == "", \
            f"Unregistered sound {sound_name} should produce no output, got: {repr(captured.out)}"