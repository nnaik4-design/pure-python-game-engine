"""
Combinatorial testing for SoundGenerator module.
Applies Each Choice Coverage (ECC) to systematically test multiple parameters without combinatorial explosion.
"""
import pytest
from engine.audio.sound_generator import Sound

# Partitions identified (Building blocks for ECC):
# frequency: [20 (Low), 440 (Mid), 10000 (High)]
# duration: [0.1 (Short), 0.5 (Medium), 1.0 (Long)]
# wave_type: ['sine', 'square', 'sawtooth', 'triangle', 'noise', 'unknown']
# amplitude: [0.0 (Zero), 0.5 (Normal), 1.0 (Max), -0.5 (Negative/Inverted)]

@pytest.mark.parametrize(
    "frequency, duration, wave_type, amplitude, expected_all_zeros",
    [
        # Combination 1: Covers Freq-Low, Dur-Short, Sine, Amp-Zero
        (20, 0.1, 'sine', 0.0, True),
        
        # Combination 2: Covers Freq-Mid, Dur-Med, Square, Amp-Normal
        (440, 0.5, 'square', 0.5, False),
        
        # Combination 3: Covers Freq-High, Dur-Long, Sawtooth, Amp-Max
        (10000, 1.0, 'sawtooth', 1.0, False),
        
        # Combination 4: Re-uses Freq-Low, Dur-Med. Covers Triangle, Amp-Negative
        (20, 0.5, 'triangle', -0.5, False),
        
        # Combination 5: Re-uses Freq-Mid, Dur-Long. Covers Noise, re-uses Amp-Normal
        (440, 1.0, 'noise', 0.5, False),
        
        # Combination 6: Re-uses Freq-High, Dur-Short. Covers Unknown, re-uses Amp-Normal
        (10000, 0.1, 'unknown', 0.5, True),
    ]
)
def test_generate_tone_ecc(frequency, duration, wave_type, amplitude, expected_all_zeros):
    """
    Combinatorial testing: Each Choice Coverage (ECC) for Sound.generate_tone.
    Ensures every parameter partition is executed at least once across 6 minimal test cases.
    """
    sound = Sound("ecc_test")
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