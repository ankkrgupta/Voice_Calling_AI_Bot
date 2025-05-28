import os
import asyncio
import json
import time
from collections import deque
from typing import Callable, Awaitable

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
import threading


def int16_to_wav(pcm_array, sample_rate, path):
    """
    Save int16 PCM numpy array to a WAV file on disk.
    """
    sf.write(path, pcm_array, sample_rate, format="WAV", subtype="PCM_16")

class DeepgramTranscriber:
    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        vad_mode: int = 1,
        silence_limit: float = 1.0,
        interim: bool = True,
        language: str = "en",
        raw_output_dir: str = "recordings",
    ):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)
        self.silence_limit = silence_limit
        self.interim = interim
        os.makedirs(raw_output_dir, exist_ok=True)
        self.raw_output_dir = raw_output_dir
        self._pcm_ring = deque(maxlen=int(sample_rate * (silence_limit + 30)))
        self.dg_client = DeepgramClient(api_key)
        self.dg_conn = None
        self._main_loop = None
        self._mic_enabled = True
        self._paused = asyncio.Event()
        self._paused.set()  # not paused initially

    def pause(self):
        print("[STT] Paused.")
        self._paused.clear()
        self._mic_enabled = False

    def resume(self):
        print("[STT] Resumed.")
        self._paused.set()
        self._mic_enabled = True

    def _handle_transcript(self, evt, result, on_transcript: Callable[[str, bool], Awaitable]):
        alt = result.channel.alternatives[0]
        text = alt.transcript.strip()
        is_final = getattr(result, "is_final", False)
        if not text:
            return
        # Send to app only if final
        asyncio.run_coroutine_threadsafe(on_transcript(text, is_final), self._main_loop)
        if is_final:
            pcm_array = np.array(self._pcm_ring, dtype=np.int16)
            self._pcm_ring.clear()
            ts = int(time.time())
            path = os.path.join(self.raw_output_dir, f"utt_{ts}.wav")
            int16_to_wav(pcm_array, self.sample_rate, path)
            print(f"[STT] Saved utterance: {path}")

    async def _send_audio(self):
        dev = sd.query_devices(kind="input")
        print(dev)
        mic_sr = int(dev.get("default_samplerate", self.sample_rate))
        print(f"[STT] Input mic sample rate: {mic_sr}")
        q = asyncio.Queue(maxsize=2000)
        send_buffer = bytearray()
        frame_bytes = int(self.sample_rate * 20 / 1000) * 2  # 30ms chunks

        def mic_callback(indata, frames, time_info, status):
            if not self._mic_enabled:
                # print("Mic is not enabled")
                return
            if status:
                print(f"[STT] Mic warning: {status}")
            mono = indata[:, 0]
            # Resample to match Deepgram sample rate
            resamp = np.interp(
                np.linspace(0, len(mono), int(len(mono) * self.sample_rate / mic_sr)),
                np.arange(len(mono)), mono
            )
            pcm = np.clip(np.round(resamp * 32767), -32768, 32767).astype(np.int16)
            raw_bytes = pcm.tobytes()
            self._pcm_ring.extend(pcm)
            send_buffer.extend(pcm.tobytes())
            while len(send_buffer) >= frame_bytes:
                chunk = send_buffer[:frame_bytes]
                del send_buffer[:frame_bytes]
                # if not self.vad.is_speech(raw_bytes, self.sample_rate):
                #     return
                try:
                    self._main_loop.call_soon_threadsafe(q.put_nowait, chunk)
                except asyncio.QueueFull:
                    pass

        with sd.InputStream(
            # device=0,
            samplerate=mic_sr,
            channels=1,
            dtype="float32",
            callback=mic_callback,
        ):
            print("[STT] 🎙️ Microphone stream started.")
            while True:
                await self._paused.wait()
                chunk = await q.get()
                self.dg_conn.send(chunk)

    async def start(self, on_transcript: Callable[[str, bool], Awaitable]):
        print("[STT] 🔌 Connecting to Deepgram WebSocket...")
        self.dg_conn = self.dg_client.listen.websocket.v("1")
        self.dg_conn.on(
            LiveTranscriptionEvents.Transcript,
            lambda evt, result: self._handle_transcript(evt, result, on_transcript)
        )
        self._main_loop = asyncio.get_running_loop()
        opts = LiveOptions(
            punctuate=True,
            smart_format=True,
            model="nova-3",
            encoding="linear16",
            sample_rate=self.sample_rate,
            channels=1,
            # interim_results=self.interim,
            interim_results=True,        # must be True
            vad_events=True,             # let DG endpoint as soon as pause
            endpointing=300              # 200 ms of trailing silence
        )
        success = await asyncio.to_thread(self.dg_conn.start, opts)
        if not success:
            raise RuntimeError("❌ Failed to start Deepgram connection.")
        print("[STT] ✅ Deepgram connection started.")

        # Keepalive pings
        def keepalive():
            while True:
                try:
                    self.dg_conn.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    pass
                time.sleep(5)
        threading.Thread(target=keepalive, daemon=True).start()

        # Start sending mic audio
        await self._send_audio()
        self.dg_conn.finish()
        print("[STT] 🛑 Deepgram connection closed.")
