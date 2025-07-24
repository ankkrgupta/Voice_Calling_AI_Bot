import asyncio
from typing import Optional
from fastapi import WebSocket
from elevenlabs.client import ElevenLabs
import requests
import time
from collections import deque
import json

class TTSClient:
    """
    Advanced ElevenLabs TTS client with optimized streaming and flow control.
    
    Features:
    - Persistent audio buffer queue with deque for smooth streaming
    - Batch sending with 200-300ms accumulation to reduce WebSocket overhead
    - Rolling latency window monitoring for adaptive throttling
    - Smart burst detection with conditional delays
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        optimize_latency: int = 3,
        *,
        sample_rate: int = 24000,
        batch_size_ms: int = 100,
        chunk_size_ms: int = 100,
    ):
        """Create a TTS client suitable for real-time mobile voice calls.

        Parameters
        ----------
        api_key, voice_id : str
            ElevenLabs credentials.
        optimize_latency : int, default 3
            ElevenLabs latency preset (0-4). Lower = lower latency but potentially more jitter.
        sample_rate : int, default 24000
            PCM sample-rate to request from ElevenLabs (reduces bandwidth vs 44.1 kHz).
        batch_size_ms : int, default 100
            Accumulate *this* many milliseconds of audio before flushing.
        chunk_size_ms : int, default 100
            Size of the slices actually sent over the socket (<= batch size).
        """

        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.optimize_latency = optimize_latency

        # Stream configuration
        self.sample_rate = sample_rate
        self.batch_size_ms = batch_size_ms
        self.chunk_size_ms = chunk_size_ms

        bytes_per_sample = 2  # 16-bit mono

        # Derived sizes
        self.batch_size_bytes = int(self.sample_rate * bytes_per_sample * (self.batch_size_ms / 1000))
        self.optimal_chunk_size = int(self.sample_rate * bytes_per_sample * (self.chunk_size_ms / 1000))

        # Internal buffers/metrics
        self.audio_buffer = deque()
        self.latency_window = deque(maxlen=10)
        self.last_send_time = 0.0
        self.total_bytes_sent = 0
        
    def _calculate_adaptive_delay(self) -> float:
        """Calculate adaptive delay based on rolling latency window"""
        if len(self.latency_window) < 3:
            return 0.0  # No delay until we have sufficient data
        
        avg_latency = sum(self.latency_window) / len(self.latency_window)
        
        # Adaptive throttling: higher latency = more aggressive throttling
        if avg_latency > 0.1:  # >100ms average latency
            return 0.02  # 20ms delay
        elif avg_latency > 0.05:  # >50ms average latency  
            return 0.01  # 10ms delay
        else:
            return 0.0  # No delay for good latency
    
    async def _send_batch_if_ready(self, ws: WebSocket, force: bool = False) -> None:
        """Send accumulated batch if ready or forced"""
        current_buffer_size = sum(len(chunk) for chunk in self.audio_buffer)
        
        # Send if buffer is large enough or forced (end of sentence)
        if current_buffer_size >= self.batch_size_bytes or (force and current_buffer_size > 0):
            # Combine all chunks in buffer
            combined_audio = b''.join(self.audio_buffer)
            self.audio_buffer.clear()
            
            # Split into optimal chunks for frontend
            while len(combined_audio) > 0:
                chunk = combined_audio[:self.optimal_chunk_size]
                combined_audio = combined_audio[self.optimal_chunk_size:]
                
                # Send with latency tracking
                send_start = time.perf_counter()
                await ws.send_bytes(chunk)
                send_duration = time.perf_counter() - send_start
                
                # Track latency for adaptive throttling
                self.latency_window.append(send_duration)
                self.total_bytes_sent += len(chunk)
                
                # Smart burst detection: only delay if sending lots of data quickly
                if len(combined_audio) > 2 * self.optimal_chunk_size:
                    adaptive_delay = self._calculate_adaptive_delay()
                    if adaptive_delay > 0:
                        await asyncio.sleep(adaptive_delay)
    
    def reset_buffer(self) -> None:
        """Reset audio buffer and stats for new session"""
        self.audio_buffer.clear()
        self.latency_window.clear()
        self.total_bytes_sent = 0
        print("[TTS] Buffer reset for new session")

    async def speak_sentence(
        self,
        sentence: str,
        ws: WebSocket,
        chosen_lang_el: str,
        last_user_end_ts: Optional[float]
    ) -> None:
        """
        Streams TTS audio for a sentence and sends bytes over WebSocket.
        
        OPTIMIZATION: Splits large ElevenLabs chunks into smaller ~150ms chunks
        for smooth frontend playback without merge delays or buffering hiccups.
        
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
                    output_format="pcm_24000" if self.sample_rate == 24000 else "pcm_44100",
                )
                break
            except Exception as e:
                print(f"[TTS] Transient error: {e}, retrying in {backoff} seconds.")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        print(f"[TTS] Streaming started for '{sentence[:30]}...' (batch mode)")
        first = True
        t0 = None
        chunk_count = 0
        
        # Reset streaming stats for this sentence
        sentence_start_time = time.perf_counter()
        
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
            
            # Add chunk to persistent buffer queue
            self.audio_buffer.append(pcm_chunk)
            chunk_count += 1
            
            # Send batch if buffer is ready (200-300ms accumulated)
            await self._send_batch_if_ready(ws)
        
        # Force send any remaining buffered audio at end of sentence
        await self._send_batch_if_ready(ws, force=True)

        if t0:
            total = time.perf_counter() - t0
            avg_latency = sum(self.latency_window) / len(self.latency_window) if self.latency_window else 0.0
            print(
                f"[TIMING] TTS sentence: {total:.3f}s, chunks: {chunk_count}, "
                f"avg_send_latency: {avg_latency*1000:.1f}ms"
            )