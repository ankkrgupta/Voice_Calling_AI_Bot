import asyncio
from typing import Optional
from fastapi import WebSocket
from elevenlabs.client import ElevenLabs
import requests
import time

class TTSClient:
    """
    Useing ElevenLabs API to stream PCM synthesized audio for given text.
    Implementing exponential backoff for reliability and abstracts WebSocket forwarding.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        optimize_latency: int = 3
    ):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.optimize_latency = optimize_latency
        

    async def speak_sentence(
        self,
        sentence: str,
        ws: WebSocket,
        chosen_lang_el: str,
        last_user_end_ts: Optional[float]
    ) -> None:
        """
        Streams TTS audio for a sentence and sends bytes over WebSocket.
        Computes and logs latency metrics.
        """
        backoff = 1.0
        max_backoff = 5.0
        # Retry on transient errors
        while True:
            try:
                audio_iter = await asyncio.to_thread(
                    self.client.text_to_speech.stream,
                    text=sentence,
                    voice_id=self.voice_id,
                    model_id="eleven_flash_v2_5",
                    optimize_streaming_latency=self.optimize_latency,
                    language_code=chosen_lang_el,
                    output_format="pcm_44100",
                )
                break
            except Exception as e:
                print(f"[TTS] Transient error: {e}, retrying in {backoff} seconds.")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        print(f"[TTS] Streaming started for '{sentence[:30]}...'")
        first = True
        t0 = None

        for pcm_chunk in audio_iter:
            if not pcm_chunk:
                continue
            if first:
                t0 = time.perf_counter()
                if last_user_end_ts is not None:
                    latency = t0 - last_user_end_ts
                    print(f"[LATENCY] User→TTS startup: {latency:.3f} sec")
                print(f"[{t0:.3f}] ← first TTS chunk for '{sentence[:30]}...'" )
                first = False
            # send each chunk over WebSocket
            await ws.send_bytes(pcm_chunk)

        if t0:
            total = time.perf_counter() - t0
            print(f"[TIMING] TTS for sentence took {total:.3f} sec")