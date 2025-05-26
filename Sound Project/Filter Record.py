import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.fft import fft
from scipy.signal import butter, lfilter

def record_audio(duration, sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait() 
    print("Recording complete.")
    return audio.ravel() 

def apply_filter(data, cutoff_freq, sample_rate, filter_type='low', order=5):
    nyquist_rate = sample_rate * 0.5
    normal_cutoff = cutoff_freq / nyquist_rate
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def perform_fourier_transform(data, sample_rate):
    N = len(data)
    T = 1.0 / sample_rate 
    fft_output = fft(data)
    frequencies = np.fft.fftfreq(N, T)[:N // 2]  
    magnitudes = 2.0 / N * np.abs(fft_output[:N // 2]) 
    return frequencies, magnitudes

duration = 5
sample_rate = 44100
cutoff_frequency = 1000  


audio = record_audio(duration, sample_rate)

filtered_audio = apply_filter(audio, cutoff_frequency, sample_rate, filter_type='low')

frequencies, magnitudes = perform_fourier_transform(filtered_audio, sample_rate)


sf.write('filtered_audio.wav', filtered_audio, sample_rate)

print("Noise reduction and Fourier transform completed.")