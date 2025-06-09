# import os
# import asyncio
# import json
# import time
# import threading
# from collections import deque
# from typing import Callable, Optional
# from scipy.signal import resample_poly
# import numpy as np
# import soundfile as sf
# import sounddevice as sd
# import webrtcvad
# from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents


# class DeepgramTranscriber:
#     """
#     Streams raw PCM audio to Deepgram for real-time transcription.
#     Saves audio segments to disk when a final transcript is received.

#     Usage:
#         stt = DeepgramTranscriber(api_key, use_mic=False)
#         await stt.start(on_transcript_callback)
#         stt.feed_audio(pcm_bytes)  # if use_mic=False
#         stt.finish()  # to close connection
#     """

#     def __init__(
#         self,
#         api_key: str,
#         sample_rate: int = 16000,
#         vad_mode: int = 2,
#         silence_limit: float = 1.0,
#         interim: bool = True,
#         language: str = "en-IN",
#         # language: str = "multi",
#         raw_output_dir: str = "recordings",
#         use_mic: bool = False,
#     ):
#         self.sample_rate = sample_rate
#         self.vad = webrtcvad.Vad(vad_mode)
#         self.silence_limit = silence_limit
#         self.interim = interim
#         self.use_mic = use_mic
#         self.language = language

#         # Prepare output directory for saving raw WAVs
#         os.makedirs(raw_output_dir, exist_ok=True)
#         self.raw_output_dir = raw_output_dir

#         # Ring buffer holds up to (silence_limit + 30) seconds of PCM int16
#         max_len = int(sample_rate * (silence_limit + 30))
#         self._pcm_ring = deque(maxlen=max_len)

#         # Deepgram client and placeholder for WebSocket connection
#         self.dg_client = DeepgramClient(api_key)
#         self.dg_conn = None

#         # Will store asyncio loop reference once `start()` is called
#         self._main_loop: Optional[asyncio.AbstractEventLoop] = None

#         # Mic/state flags
#         self._mic_enabled = True
#         self._running = False

#         self._utterance_in_progress = False
#         self._utterance_start_ts = None


#     def _handle_transcript(self, evt, result, on_transcript: Callable[[str, bool], asyncio.Future]):
#         """
#         Internal callback invoked by Deepgram on each transcript event.

#         - Extracts the top alternative transcript.
#         - If non-empty, schedules `on_transcript(text, is_final)` on the asyncio loop.
#         - On final segments, saves buffered PCM to a WAV file.
#         """
#         try:
#             alt = result.channel.alternatives[0]
#             text = alt.transcript.strip()
#             is_final = getattr(result, "is_final", False)
#             words = getattr(alt, "words", [])
#         except Exception:
#             # Malformed payload; skip
#             return

#         if not text:
#             # No words recognized yet
#             return

#         # print(result.channel.alternatives[0])
#           # ➋ If this is an interim (is_final=False) and we haven’t stamped a start yet:
#         if not is_final and not self._utterance_in_progress:
#            # First time we see non‐empty interim: mark start
#             self._utterance_in_progress = True
#             self._utterance_start_ts = time.perf_counter()

#         # ➌ When this is final, record end time and compute latency
#         if is_final and self._utterance_in_progress:
#             utterance_end_ts = time.perf_counter()
#             ut_latency = utterance_end_ts - self._utterance_start_ts
#             print(f"[STT] Utterance transcription latency: {ut_latency:.3f} sec")
#             # Reset for next utterance
#             self._utterance_in_progress = False
#             self._utterance_start_ts = None

#         # Forward transcript to user callback in the asyncio loop
#         if self._main_loop and on_transcript:
#             try:
#                 asyncio.run_coroutine_threadsafe(on_transcript(text, is_final), self._main_loop)
#             except Exception:
#                 pass

#         if is_final:
#             # Save buffered PCM (int16) to disk as WAV
#             try:
#                 pcm_array = np.array(self._pcm_ring, dtype=np.int16)
#                 self._pcm_ring.clear()
#                 timestamp = int(time.time())
#                 filename = f"utt_{timestamp}.wav"
#                 path = os.path.join(self.raw_output_dir, filename)
#                 # sf.write(path, pcm_array, self.sample_rate, format="WAV", subtype="PCM_16")
#                 threading.Thread(target=lambda: sf.write(path, pcm_array, self.sample_rate, format="WAV", subtype="PCM_16"), daemon=True).start()
#                 print(f"Audio File saved as {path}")
#             except Exception as e:
#                 print(f"[STT] Error saving utterance: {e}")

#     async def start(self, on_transcript: Callable[[str, bool], asyncio.Future]):
#         """
#         Connects to Deepgram and begins streaming.

#         If use_mic=False, you must call `feed_audio()` to send PCM bytes.
#         Otherwise, internal mic capture (in `_send_audio()`) will send audio.

#         Args:
#             on_transcript: async callback receiving (text: str, is_final: bool).
#         """
#         if self._running:
#             raise RuntimeError("DeepgramTranscriber is already running.")
#         self._running = True

#         # Create a new Deepgram WebSocket connection
#         try:
#             self.dg_conn = self.dg_client.listen.websocket.v("1")
#         except Exception as e:
#             raise RuntimeError(f"[STT] Failed to initialize Deepgram client: {e}")

#         # Register transcript event handler
#         # Deepgram invokes handler(evt, result) on each transcript or VAD event
#         self.dg_conn.on(
#             LiveTranscriptionEvents.Transcript,
#             lambda evt, result: self._handle_transcript(evt, result, on_transcript),
#         )

#         # Store asyncio loop so we can schedule callbacks
#         self._main_loop = asyncio.get_running_loop()

#         # Build LiveOptions
#         opts = LiveOptions(
#             punctuate=True,
#             smart_format=True,
#             model="nova-3",
#             encoding="linear16",
#             # keyterm="Zomato",
#             sample_rate=self.sample_rate,
#             channels=1,
#             interim_results=self.interim,
#             vad_events=True,
#             endpointing=300,
#             language=self.language,
#         )

#         # from urllib.parse import urlencode

#         # query = {
#         #     "model":          opts.model,
#         #     "language":       opts.language,
#         #     "encoding":       opts.encoding,
#         #     "sample_rate":    opts.sample_rate,
#         #     "interim_results": str(opts.interim_results).lower(),
#         #     "smart_format":   str(opts.smart_format).lower(),
#         #     "vad_events":     str(opts.vad_events).lower(),
#         #     "endpointing":    opts.endpointing,
#         #     "keyterm":        opts.keyterm or "",
#         # }
#         # ws_url = f"wss://api.deepgram.com/v1/listen?{urlencode(query)}"
#         # print(f"[STT DEBUG] Deepgram WS URL → {ws_url}")

#         # Starting streaming in a thread, since .start() is blocking
#         try:
#             success = await asyncio.to_thread(self.dg_conn.start, opts)
#         except Exception as e:
#             raise RuntimeError(f"[STT] Deepgram start failed: {e}")

#         if not success:
#             raise RuntimeError("[STT] Deepgram connection failed to start.")

#         # Spawn keep-alive pings to prevent WebSocket timeout
#         def _keepalive():
#             while self._running:
#                 time.sleep(5)
#                 try:
#                     if self.dg_conn:
#                         self.dg_conn.send(json.dumps({"type": "KeepAlive"}))
#                 except Exception:
#                     break

#         threading.Thread(target=_keepalive, daemon=True).start()

#         # If using microphone internally, begin capturing and sending
#         if self.use_mic:
#             try:
#                 await self._send_audio()
#             except asyncio.CancelledError:
#                 # Graceful shutdown of mic capture
#                 pass
#             except Exception as e:
#                 print(f"[STT] Error in microphone streaming: {e}")
#         else:
#             # Otherwise, just stay alive until external calls to feed_audio() finish
#             await asyncio.Future()

#     def feed_audio(self, data: bytes):
#         """
#         Send raw PCM16 bytes (16 kHz mono) directly to Deepgram.
        
#         Args:
#             data: raw PCM chunks, in int16 little-endian bytes.
#         """

#         try:
#             pcm_samples = np.frombuffer(data, dtype=np.int16)
#             self._pcm_ring.extend(pcm_samples)
#         except Exception:
#             pass

#         if self.dg_conn:
#             try:
#                 self.dg_conn.send(data)
#             except Exception:
#                 pass

#     def finish(self):
#         """
#         Gracefully closes the Deepgram WebSocket connection.
#         """
#         self._running = False
#         if self.dg_conn:
#             try:
#                 self.dg_conn.finish()
#             except Exception:
#                 pass

#     async def _send_audio(self):
#         """
#         Internal coroutine to capture microphone audio and feed to Deepgram.

#         - Queries the default input device for its samplerate.
#         - Uses sounddevice to capture float32 audio, rescales to int16.
#         - Buffers PCM in `self._pcm_ring` for saving utterances.
#         - Sends 20 ms frames to Deepgram via `self.dg_conn.send()`.
#         """

#         # Query default input device
#         try:
#             dev_info = sd.query_devices(kind="input")
#             mic_sr = int(dev_info.get("default_samplerate", self.sample_rate))
#         except Exception as e:
#             raise RuntimeError(f"[STT] Failed to query microphone: {e}")

#         print(f"[STT] Input mic sample rate: {mic_sr}")

#         # Queue for fixed-size frames
#         frame_duration_ms = 20  # 20 ms per frame
#         frame_bytes = int(self.sample_rate * frame_duration_ms / 1000) * 2  # int16 → 2 bytes/sample

#         q: asyncio.Queue = asyncio.Queue(maxsize=5000)
#         send_buffer = bytearray()

#         # Event to pause/resume streaming if needed
#         self._paused = asyncio.Event()
#         self._paused.set()

#         def mic_callback(indata, frames, time_info, status):
#             """
#             sounddevice callback for each chunk of captured audio.

#             - `indata` is float32 in [-1, 1]; we convert to int16.
#             - Resample to match Deepgram sample_rate if needed.
#             - Append to ring buffer for WAV saving.
#             - Accumulate in send_buffer and enqueue fixed-size frames.
#             """
#             if not self._mic_enabled:
#                 return
#             if status:
#                 print(f"[STT] Mic warning: {status}")

#             # Use only first channel if multiple
#             mono = indata[:, 0]

#             # Resample if mic_sr != target sample_rate
#             if mic_sr != self.sample_rate:
#                 try:
#                     # resampled = np.interp(
#                     #     np.linspace(0, len(mono), int(len(mono) * self.sample_rate / mic_sr)),
#                     #     np.arange(len(mono)),
#                     #     mono,
#                     # )
#                     up = self.sample_rate
#                     down = mic_sr
#                     # resample_poly automatically handles anti‐aliasing
#                     resampled = resample_poly(mono, up, down)
#                 except Exception:
#                     return
#             else:
#                 resampled = mono

#             # Convert float32 [-1,1] → int16
#             pcm = np.clip((resampled * 32767).round(), -32768, 32767).astype(np.int16)
#             raw_bytes = pcm.tobytes()

#             # Buffer for saving utterance on final segments
#             self._pcm_ring.extend(pcm)

#             # Accumulate raw_bytes until we have a full 20 ms frame
#             send_buffer.extend(raw_bytes)
#             while len(send_buffer) >= frame_bytes:
#                 chunk = send_buffer[:frame_bytes]
#                 del send_buffer[:frame_bytes]
#                 # Optionally, do VAD check here:
#                 if not self.vad.is_speech(chunk, self.sample_rate):
#                     continue
#                 try:
#                     self._main_loop.call_soon_threadsafe(q.put_nowait, chunk)
#                 except asyncio.QueueFull:
#                     # Drop frame if queue is full
#                     pass

#         try:
#             # Open microphone input stream
#             with sd.InputStream(
#                 samplerate=mic_sr,
#                 blocksize=320,
#                 channels=1,
#                 dtype="float32",
#                 callback=mic_callback,
#                 latency='low',
#             ):
#                 print("[STT] 🎙️ Microphone stream started.")
#                 while self._running:
#                     # Wait until unpaused (always unpaused by default)
#                     await self._paused.wait()
#                     try:
#                         frame = await q.get()
#                         self.dg_conn.send(frame)
#                     except asyncio.CancelledError:
#                         break
#                     except Exception:
#                         continue
#         except Exception as e:
#             raise RuntimeError(f"[STT] Failed to open microphone stream: {e}")


'''Latency Update'''

import os
import asyncio
import json
import time
import threading
from collections import deque
from typing import Callable, Optional
from scipy.signal import resample_poly
import numpy as np
import soundfile as sf
import sounddevice as sd
import webrtcvad
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents


class DeepgramTranscriber:
    """
    Streams raw PCM audio to Deepgram for real-time transcription.
    Saves audio segments to disk when a final transcript is received.

    Usage:
        stt = DeepgramTranscriber(api_key, use_mic=False)
        await stt.start(on_transcript_callback)
        stt.feed_audio(pcm_bytes)  # if use_mic=False
        stt.finish()  # to close connection
    """

    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        vad_mode: int = 2,
        silence_limit: float = 1.0,
        interim: bool = True,
        # language: str = "en-IN",
        language: str = "multi",
        raw_output_dir: str = "recordings",
        use_mic: bool = False,
    ):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_mode)
        self.silence_limit = silence_limit
        self.interim = interim
        self.use_mic = use_mic
        self.language = language

        # Prepare output directory for saving raw WAVs
        os.makedirs(raw_output_dir, exist_ok=True)
        self.raw_output_dir = raw_output_dir

        # Ring buffer holds up to (silence_limit + 30) seconds of PCM int16
        max_len = int(sample_rate * (silence_limit + 30))
        self._pcm_ring = deque(maxlen=max_len)

        # Deepgram client and placeholder for WebSocket connection
        self.dg_client = DeepgramClient(api_key)
        self.dg_conn = None

        # Will store asyncio loop reference once `start()` is called
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Mic/state flags
        self._mic_enabled = True
        self._running = False

        self._utterance_in_progress = False
        self._utterance_start_ts = None


    def _handle_transcript(self, evt, result, on_transcript: Callable[[str, bool], asyncio.Future]):
        """
        Internal callback invoked by Deepgram on each transcript event.

        - Extracts the top alternative transcript.
        - If non-empty, schedules `on_transcript(text, is_final)` on the asyncio loop.
        - On final segments, saves buffered PCM to a WAV file.
        """
        try:
            alt = result.channel.alternatives[0]
            text = alt.transcript.strip()
            is_final = getattr(result, "is_final", False)
            words = getattr(alt, "words", [])
        except Exception:
            # Malformed payload; skip
            return

        if not text:
            # No words recognized yet
            return

        # print(result.channel.alternatives[0])
          # ➋ If this is an interim (is_final=False) and we haven’t stamped a start yet:
        now = time.perf_counter()
        if not is_final and not self._utterance_in_progress:
           # First time we see non‐empty interim: mark start
            print(f"[{time.perf_counter():.3f}] [STT] first interim: “{text}”")
            self._utterance_in_progress = True
            self._utterance_start_ts = time.perf_counter()

        # ➌ When this is final, record end time and compute latency
        if is_final and self._utterance_in_progress:
            print(f"[{time.perf_counter():.3f}] [STT] final transcript: “{text}”")
            utterance_end_ts = time.perf_counter()
            ut_latency = utterance_end_ts - self._utterance_start_ts
            print(f"[STT] Utterance transcription latency: {ut_latency:.3f} sec")
            # Reset for next utterance
            self._utterance_in_progress = False
            self._utterance_start_ts = None

        # Forward transcript to user callback in the asyncio loop
        if self._main_loop and on_transcript:
            try:
                asyncio.run_coroutine_threadsafe(on_transcript(text, is_final), self._main_loop)
            except Exception:
                pass

        if is_final:
            # Save buffered PCM (int16) to disk as WAV
            try:
                pcm_array = np.array(self._pcm_ring, dtype=np.int16)
                self._pcm_ring.clear()
                timestamp = int(time.time())
                filename = f"utt_{timestamp}.wav"
                path = os.path.join(self.raw_output_dir, filename)
                # sf.write(path, pcm_array, self.sample_rate, format="WAV", subtype="PCM_16")
                threading.Thread(target=lambda: sf.write(path, pcm_array, self.sample_rate, format="WAV", subtype="PCM_16"), daemon=True).start()
                print(f"Audio File saved as {path}")
            except Exception as e:
                print(f"[STT] Error saving utterance: {e}")

    async def start(self, on_transcript: Callable[[str, bool], asyncio.Future]):
        """
        Connects to Deepgram and begins streaming.

        If use_mic=False, you must call `feed_audio()` to send PCM bytes.
        Otherwise, internal mic capture (in `_send_audio()`) will send audio.

        Args:
            on_transcript: async callback receiving (text: str, is_final: bool).
        """
        if self._running:
            raise RuntimeError("DeepgramTranscriber is already running.")
        self._running = True

        # Create a new Deepgram WebSocket connection
        try:
            self.dg_conn = self.dg_client.listen.websocket.v("1")
        except Exception as e:
            raise RuntimeError(f"[STT] Failed to initialize Deepgram client: {e}")

        # Register transcript event handler
        # Deepgram invokes handler(evt, result) on each transcript or VAD event
        self.dg_conn.on(
            LiveTranscriptionEvents.Transcript,
            lambda evt, result: self._handle_transcript(evt, result, on_transcript),
        )

        # Store asyncio loop so we can schedule callbacks
        self._main_loop = asyncio.get_running_loop()

        # Build LiveOptions
        opts = LiveOptions(
            punctuate=True,
            smart_format=True,
            model="nova-3",
            encoding="linear16",
            # keyterm="Zomato",
            sample_rate=self.sample_rate,
            channels=1,
            interim_results=self.interim,
            # vad_events=True,
            endpointing=100,
            language=self.language,
        )

        try:
            success = await asyncio.to_thread(self.dg_conn.start, opts)
        except Exception as e:
            raise RuntimeError(f"[STT] Deepgram start failed: {e}")

        if not success:
            raise RuntimeError("[STT] Deepgram connection failed to start.")

        # Spawn keep-alive pings to prevent WebSocket timeout
        def _keepalive():
            while self._running:
                time.sleep(5)
                try:
                    if self.dg_conn:
                        self.dg_conn.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    break

        threading.Thread(target=_keepalive, daemon=True).start()

        # If using microphone internally, begin capturing and sending
        if self.use_mic:
            try:
                await self._send_audio()
            except asyncio.CancelledError:
                # Graceful shutdown of mic capture
                pass
            except Exception as e:
                print(f"[STT] Error in microphone streaming: {e}")
        else:
            # Otherwise, just stay alive until external calls to feed_audio() finish
            await asyncio.Future()

    def feed_audio(self, data: bytes):
        """
        Send raw PCM16 bytes (16 kHz mono) directly to Deepgram.
        
        Args:
            data: raw PCM chunks, in int16 little-endian bytes.
        """

        try:
            pcm_samples = np.frombuffer(data, dtype=np.int16)
            self._pcm_ring.extend(pcm_samples)
        except Exception:
            pass

        if self.dg_conn:
            try:
                self.dg_conn.send(data)
            except Exception:
                pass

    def finish(self):
        """
        Gracefully closes the Deepgram WebSocket connection.
        """
        self._running = False
        if self.dg_conn:
            try:
                self.dg_conn.finish()
            except Exception:
                pass

    async def _send_audio(self):
        """
        Internal coroutine to capture microphone audio and feed to Deepgram.

        - Queries the default input device for its samplerate.
        - Uses sounddevice to capture float32 audio, rescales to int16.
        - Buffers PCM in `self._pcm_ring` for saving utterances.
        - Sends 20 ms frames to Deepgram via `self.dg_conn.send()`.
        """

        # Query default input device
        try:
            dev_info = sd.query_devices(kind="input")
            mic_sr = int(dev_info.get("default_samplerate", self.sample_rate))
        except Exception as e:
            raise RuntimeError(f"[STT] Failed to query microphone: {e}")

        print(f"[STT] Input mic sample rate: {mic_sr}")

        # Queue for fixed-size frames
        frame_duration_ms = 20  # 20 ms per frame
        frame_bytes = int(self.sample_rate * frame_duration_ms / 1000) * 2  # int16 → 2 bytes/sample

        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        send_buffer = bytearray()

        # Event to pause/resume streaming if needed
        self._paused = asyncio.Event()
        self._paused.set()

        def mic_callback(indata, frames, time_info, status):
            """
            sounddevice callback for each chunk of captured audio.

            - `indata` is float32 in [-1, 1]; we convert to int16.
            - Resample to match Deepgram sample_rate if needed.
            - Append to ring buffer for WAV saving.
            - Accumulate in send_buffer and enqueue fixed-size frames.
            """
            if not self._mic_enabled:
                return
            if status:
                print(f"[STT] Mic warning: {status}")

            # Use only first channel if multiple
            mono = indata[:, 0]

            # Resample if mic_sr != target sample_rate
            if mic_sr != self.sample_rate:
                try:
                    # resampled = np.interp(
                    #     np.linspace(0, len(mono), int(len(mono) * self.sample_rate / mic_sr)),
                    #     np.arange(len(mono)),
                    #     mono,
                    # )
                    up = self.sample_rate
                    down = mic_sr
                    # resample_poly automatically handles anti‐aliasing
                    resampled = resample_poly(mono, up, down)
                except Exception:
                    return
            else:
                resampled = mono

            # Convert float32 [-1,1] → int16
            pcm = np.clip((resampled * 32767).round(), -32768, 32767).astype(np.int16)
            raw_bytes = pcm.tobytes()

            # Buffer for saving utterance on final segments
            self._pcm_ring.extend(pcm)

            # Accumulate raw_bytes until we have a full 20 ms frame
            send_buffer.extend(raw_bytes)
            while len(send_buffer) >= frame_bytes:
                chunk = send_buffer[:frame_bytes]
                del send_buffer[:frame_bytes]
                # Optionally, do VAD check here:
                if not self.vad.is_speech(chunk, self.sample_rate):
                    continue
                try:
                    self._main_loop.call_soon_threadsafe(q.put_nowait, chunk)
                except asyncio.QueueFull:
                    # Drop frame if queue is full
                    pass

        try:
            # Open microphone input stream
            with sd.InputStream(
                samplerate=mic_sr,
                blocksize=320,
                channels=1,
                dtype="float32",
                callback=mic_callback,
                latency='low',
            ):
                print("[STT] 🎙️ Microphone stream started.")
                while self._running:
                    # Wait until unpaused (always unpaused by default)
                    await self._paused.wait()
                    try:
                        frame = await q.get()
                        self.dg_conn.send(frame)
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        continue
        except Exception as e:
            raise RuntimeError(f"[STT] Failed to open microphone stream: {e}")

