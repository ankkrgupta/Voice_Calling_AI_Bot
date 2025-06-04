import asyncio
from typing import AsyncGenerator
from elevenlabs.client import ElevenLabs


class TTSClient:
    """
    Uses ElevenLabs API to stream PCM synthesized audio for given text.
    Implements exponential backoff for reliability.
    """

    def __init__(self, api_key: str, voice_id: str, target_rate: int = 16000):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.target_rate = target_rate  # Currently not resampling; Eleven Labs returns 22050 Hz

    async def stream_synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Streams back PCM audio chunks from ElevenLabs. Retries with backoff on failure.
        """
        backoff = 1.0
        max_backoff = 5.0
        while True:
            try:
                audio_iter = self.client.text_to_speech.stream(
                    text=text,
                    voice_id=self.voice_id,
                    model_id="eleven_multilingual_v2",
                    optimize_streaming_latency=2,
                    output_format="pcm_22050",
                )
                break
            except Exception as e:
                print(f"[TTS] Transient error: {e}, retrying in {backoff} seconds.")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        print("[TTS] Streaming started...")
        try:
            for chunk in audio_iter:
                if chunk:
                    yield chunk
        except Exception as e:
            raise RuntimeError(f"[TTS] Streaming failed mid-way: {e}")