import numpy as np
import soundfile as sf
from elevenlabs.client import ElevenLabs
import io
import asyncio

class ElevenTTS:
    def __init__(self, api_key: str,
                 voice_id: str,
                 target_rate: int = 48000):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.target_rate = target_rate

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Fallback method: downloads full audio, then returns numpy data and sample rate.
        """
        backoff = 1
        while True:
            try:
                chunks = self.client.text_to_speech.convert(
                    text=text,
                    voice_id=self.voice_id,
                    model_id="eleven_multilingual_v2",
                )
                break
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5)

        # Join all byte chunks
        wav_bytes = b"".join(chunks) if hasattr(chunks, '__iter__') and not isinstance(chunks, (bytes, bytearray)) else chunks
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype='int16')
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        return data, sr

    async def stream_synthesize(self, text: str):
        """
        Streaming method: yields (numpy_array, sample_rate) for each chunk as soon as it's available.
        """
        # Initiate streaming request
        # latest ElevenLabs SDK supports async streaming via `stream`
        stream = self.client.text_to_speech.stream(
            text=text,
            voice_id=self.voice_id,
            model_id="eleven_multilingual_v2",
        )
        async for raw_chunk in stream:
            # raw_chunk is bytes (WAV or raw PCM)
            try:
                data, sr = sf.read(io.BytesIO(raw_chunk), dtype='int16')
            except Exception:
                # if chunk is raw PCM, interpret directly
                pcm = np.frombuffer(raw_chunk, dtype=np.int16)
                data, sr = pcm, self.target_rate

            if data.ndim > 1:
                data = data.mean(axis=1).astype(np.int16)

            yield data, sr
