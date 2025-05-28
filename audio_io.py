import sounddevice as sd
from scipy.signal import resample_poly
import numpy as np

class AudioIO:
    def __init__(self, target_rate: int = None, block_size: int = 256): #Initially it was 1024
        if target_rate is None:
            dev = sd.query_devices(kind="output")
            target_rate = int(dev["default_samplerate"])
        self.target_rate = target_rate
        self.block_size = block_size

    def stream_play(self, samples: np.ndarray, samplerate: int):
        # stream and resample each block if needed
        up = down = 1
        if samplerate != self.target_rate:
            # Determine resample ratio
            from math import gcd
            factor = gcd(int(samplerate), int(self.target_rate))
            up = self.target_rate // factor
            down = samplerate // factor

        with sd.OutputStream(samplerate=self.target_rate, channels=1, dtype=samples.dtype) as out:
            for i in range(0, len(samples), self.block_size):
                chunk = samples[i:i+self.block_size]

                if samplerate != self.target_rate:
                    # Per-chunk polyphase resampling
                    chunk = resample_poly(chunk, up, down).astype(samples.dtype)

                out.write(chunk)

    def play(self, samples: np.ndarray, samplerate: int):
        self.stream_play(samples, samplerate)
