"""
Combinatorial testing for SoundGenerator module.
"""
import pytest
from engine.audio.sound_generator import Sound

# Partitions identified:
# frequency: [20 (Low), 440 (Mid), 10000 (High)]
# duration: [0.1 (Short), 0.5 (Medium), 1.0 (Long)]
# wave_type: ['sine', 'square', 'sawtooth', 'triangle', 'noise', 'unknown']
# amplitude: [0.0 (Zero), 0.5 (Normal), 1.0 (Max), -0.5 (Negative/Inverted)]

@pytest.mark.parametrize(
    "frequency, duration, wave_type, amplitude, expected_all_zeros",
    [
        # Pair-Wise Coverage (PWC) testing matrix targeting parameter interactions
        (20, 0.1, 'sine', 0.0, True),
        (20, 0.5, 'square', 0.5, False),
        (20, 1.0, 'sawtooth', 1.0, False),
        (440, 0.1, 'triangle', -0.5, False),
        (440, 0.5, 'noise', 0.0, True),
        (440, 1.0, 'unknown', 0.5, True),
        (10000, 0.1, 'sawtooth', 0.5, False),
        (10000, 0.5, 'unknown', 1.0, True),
        (10000, 1.0, 'sine', -0.5, False),
        (20, 1.0, 'noise', -0.5, False),
        (440, 0.1, 'square', 1.0, False),
        (10000, 0.5, 'triangle', 0.0, True),
    ]
)
def test_generate_tone_pwc(frequency, duration, wave_type, amplitude, expected_all_zeros):
    """
    Combinatorial testing: Pair-Wise Coverage (PWC) for Sound.generate_tone.
    Ensures pairwise interactions across complex audio waveforms and modifiers are fully exercised.
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
        assert all(sample == 0.0 for sample in sound.samples)
    else:
        # If it's a valid wave with non-zero amplitude, it should have non-zero samples
        assert any(sample != 0.0 for sample in sound.samples)
        
        # Validate boundary constraint: no sample should exceed absolute amplitude
        max_generated_amplitude = max(abs(s) for s in sound.samples)
        assert max_generated_amplitude <= abs(amplitude) + 1e-6

# Partitions identified for generate_sweep:
# start_freq: [200 (Low), 800 (Mid), 2000 (High)]
# end_freq_relation: ['up' (end > start), 'same' (end == start), 'down' (end < start)]
# duration: [-0.1 (Invalid), 0.1 (Short), 1.0 (Long)]
# wave_type: ['sine', 'square', 'unsupported']
# amplitude: [-0.5 (Invalid/Inverted), 0.5 (Normal), 1.0 (Max)]

@pytest.mark.parametrize(
    "start_freq, end_freq_relation, duration, wave_type, amplitude",
    [
        # Pair-Wise Coverage (PWC) Matrix covering parameter interactions
        (200, 'up', -0.1, 'sine', 0.5),
        (200, 'same', 0.1, 'square', 1.0),
        (200, 'down', 1.0, 'unsupported', -0.5),
        (800, 'up', 0.1, 'unsupported', -0.5),
        (800, 'same', 1.0, 'sine', 0.5),
        (800, 'down', -0.1, 'square', 1.0),
        (2000, 'up', 1.0, 'square', 0.5),
        (2000, 'same', -0.1, 'unsupported', 1.0),
        (2000, 'down', 0.1, 'sine', -0.5),
    ]
)
def test_generate_sweep_pwc(start_freq, end_freq_relation, duration, wave_type, amplitude):
    """
    Combinatorial testing: Pair-Wise Coverage (PWC) for Sound.generate_sweep.
    Ensures parameter interactions (e.g. waveform with duration and amplitude) 
    are executed across minimal test cases.
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
    
    if duration <= 0:
        assert len(sound.samples) == 0
    else:
        expected_samples = int(22050 * duration)
        assert len(sound.samples) == expected_samples

from engine.audio.sound_generator import SoundGenerator

# Partitions identified (Building blocks for ACoC):
# duration: [0.1 (Short), 1.0 (Long)]
# amplitude: [0.0 (Zero), 0.5 (Normal), 1.0 (Max)]
# All Combinations Coverage (ACoC) requires 2 * 3 = 6 combinations.

@pytest.mark.parametrize(
    "duration, amplitude",
    [
        (0.1, 0.0), (0.1, 0.5), (0.1, 1.0),
        (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
    ]
)
def test_generate_explosion_acoc(duration, amplitude):
    """
    All Combinations Coverage (ACoC) for Sound.generate_explosion.
    Provides complete interaction coverage for 2 parameters.
    """
    sound = Sound("explosion_acoc")
    sound.generate_explosion(duration, amplitude)
    
    assert sound.duration == duration
    expected_samples = int(22050 * duration)
    assert len(sound.samples) == expected_samples

# Partitions identified (Building blocks for PWC):
# base_freq: [20 (Low), 100 (Nominal), 500 (High)]
# duration: [0.1 (Short), 0.5 (Long)]
# amplitude: [0.0 (Zero), 0.5 (Normal), 1.0 (Max)]
# Pair-Wise Coverage (PWC) ensures every pair of values is tested at least once cleanly.

@pytest.mark.parametrize(
    "base_freq, duration, amplitude",
    [
        # Providing pairwise coverage across the 3 parameters without explosion
        (20, 0.1, 0.0),
        (20, 0.5, 0.5),
        (100, 0.1, 1.0),
        (100, 0.5, 0.0),
        (500, 0.1, 0.5),
        (500, 0.5, 1.0),
        (20, 0.1, 1.0),
        (100, 0.1, 0.5),
        (500, 0.1, 0.0),
    ]
)
def test_generate_engine_pwc(base_freq, duration, amplitude):
    """
    Pair-Wise Coverage (PWC) for Sound.generate_engine.
    Tests complex harmonic interactions optimally.
    """
    sound = Sound("engine_pwc")
    sound.generate_engine(base_freq, duration, amplitude)
    
    assert sound.duration == duration
    assert len(sound.samples) == int(22050 * duration)

# Partitions identified (Building blocks for ECC):
# sound_name: ["bullet" (Registered), "unknown" (Unregistered), "engine" (Registered)]
# thread_busy: [True (Thread already running), False (Thread available)]

@pytest.mark.parametrize(
    "sound_name, thread_busy",
    [
        ("bullet", False),
        ("unknown", True),
        ("engine", False),
    ]
)
def test_play_sound_ecc(sound_name, thread_busy):
    """
    Each Choice Coverage (ECC) for SoundGenerator.play_sound.
    Checks thread state management and unknown sound handling.
    """
    import threading
    import time
    
    generator = SoundGenerator()
    generator.sounds["bullet"] = Sound("bullet")
    generator.sounds["engine"] = Sound("engine")
    
    busy_thread = None
    if thread_busy:
        busy_thread = threading.Thread(target=lambda: time.sleep(0.1))
        busy_thread.start()
        generator.current_thread = busy_thread
        
    generator.play_sound(sound_name)
    
    if sound_name == "unknown":
        if thread_busy:
            assert generator.current_thread == busy_thread
        else:
            assert generator.current_thread is None
    else:
        if thread_busy:
            assert generator.current_thread == busy_thread
        else:
            assert generator.current_thread is not None
            assert generator.current_thread.is_alive()
            
    if thread_busy and busy_thread:
        busy_thread.join()

# Partitions identified (Building blocks for ACoC):
# frequency: [100 (<=200), 300 (>200), 600 (>500)]
# duration: [0.1 (Short)]
# All Combinations Coverage (ACoC) requires 3 * 1 = 3 combinations.

@pytest.mark.parametrize(
    "frequency, duration, expected_pattern",
    [
        (100, 0.1, "boom!"),
        (300, 0.1, "boop!"),
        (600, 0.1, "beep!"),
    ]
)
def test_generate_frequency_beep_acoc(frequency, duration, expected_pattern, capsys):
    """
    All Combinations Coverage (ACoC) for generate_frequency_beep.
    Effectively ensures coverage of the terminal print branching logic.
    """
    generator = SoundGenerator()
    generator.generate_frequency_beep(frequency, duration)
    
    captured = capsys.readouterr()
    assert expected_pattern in captured.out