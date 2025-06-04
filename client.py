# import os
# import asyncio
# import json
# import numpy as np
# import sounddevice as sd
# import websockets
# from typing import Optional

# SERVICE_URI = os.getenv("SERVICE_URI", "ws://localhost:8000/ws")

# MIC_DEVICE_INDEX = 1     
# PLAYBACK_DEVICE_INDEX = 2 
# async def mic_sender(ws):
#     """
#     Captures microphone audio at 16kHz, 16-bit mono, and sends 20ms PCM frames to server.
#     """
#     SAMPLE_RATE = 16000
#     frame_duration_ms = 20
#     frame_bytes = int(SAMPLE_RATE * frame_duration_ms / 1000) * 2  # int16
#     buffer = bytearray()
#     q: asyncio.Queue[bytes] = asyncio.Queue()
#     loop = asyncio.get_running_loop()

#     def callback(indata, frames, time_info, status):
#         if status:
#             print(f"[Client] Mic status: {status}")
#         # Convert float32 [-1,1] to int16
#         pcm = (indata[:, 0] * 32767).astype('int16')
#         loop.call_soon_threadsafe(q.put_nowait, pcm.tobytes())

#     try:
#         with sd.InputStream(
#             device=MIC_DEVICE_INDEX,
#             samplerate=SAMPLE_RATE,
#             channels=1,
#             dtype='float32',
#             callback=callback,
#             blocksize=320,
#             latency='low',
#         ):
#             print("[Client] Microphone streaming...")
#             while True:
#                 try:
#                     chunk = await q.get()
#                 except asyncio.CancelledError:
#                     break

#                 buffer.extend(chunk)
#                 # Send full 20ms frames, keep leftover in buffer
#                 while len(buffer) >= frame_bytes:
#                     to_send = bytes(buffer[:frame_bytes])
#                     del buffer[:frame_bytes]
#                     try:
#                         await ws.send(to_send)
#                     except Exception:
#                         return
#     except Exception as e:
#         print(f"[Client] Error opening microphone: {e}")
#         return


# async def receiver(ws):
#     """
#     Receives PCM audio and JSON messages from server.
#     Plays back PCM via sounddevice and prints transcripts/tokens.
#     """
#     PCM_RATE = 48000  # matches ElevenLabs output
#     try:
#         pcm_stream = sd.OutputStream(
#             device=PLAYBACK_DEVICE_INDEX,
#             samplerate=PCM_RATE,
#             channels=1,
#             dtype='int16',
#             blocksize=320,
#             latency='low',
#         )
#         pcm_stream.start()
#     except Exception as e:
#         print(f"[Client] Failed to open output stream: {e}")
#         return

#     try:
#         async for message in ws:
#             if isinstance(message, bytes):
#                 # Write PCM to output device
#                 try:
#                     pcm_data = np.frombuffer(message, dtype=np.int16)
#                     pcm_stream.write(pcm_data)
#                 except Exception:
#                     continue
#             else:
#                 # JSON control message
#                 try:
#                     msg = json.loads(message)
#                 except json.JSONDecodeError:
#                     continue
#                 msg_type = msg.get('type')
#                 if msg_type == 'transcript':
#                     label = 'FINAL' if msg.get('final') else 'INTERIM'
#                     print(f"\nTRANSCRIPT [{label}]: {msg.get('text')}\n")
#                 elif msg_type == 'token':
#                     print(msg.get('token'), end='', flush=True)
#                     # print(f"BOT : {msg.get('token')}")
#     except websockets.ConnectionClosed:
#         pass
#     finally:
#         pcm_stream.stop()
#         pcm_stream.close()


# async def main():
#     """
#     Connects to WebSocket server, spawns mic_sender and receiver tasks.
#     """
#     try:
#         async with websockets.connect(SERVICE_URI) as ws:
#             send_task = asyncio.create_task(mic_sender(ws))
#             recv_task = asyncio.create_task(receiver(ws))
#             done, pending = await asyncio.wait(
#                 [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
#             )
#             for task in pending:
#                 task.cancel()
#     except Exception as e:
#         print(f"[Client] WebSocket connection failed: {e}")


# if __name__ == '__main__':
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("[Client] Exiting...")

'''Enabled Mic mute, now can be tested on default speaker also'''
import os
import asyncio
import json
import numpy as np
import sounddevice as sd
import websockets
from typing import Optional

SERVICE_URI = os.getenv("SERVICE_URI", "ws://localhost:8000/ws")

MIC_DEVICE_INDEX = 1     
PLAYBACK_DEVICE_INDEX = 2  

mute_mic = False

async def mic_sender(ws):
    """
    Captures microphone audio at 16kHz, 16-bit mono, and sends 20ms PCM frames to server.
    """
    SAMPLE_RATE = 16000
    frame_duration_ms = 20
    frame_bytes = int(SAMPLE_RATE * frame_duration_ms / 1000) * 2  # int16
    buffer = bytearray()

    global mute_mic

    q: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[Client] Mic status: {status}")
        # If bot is speaking, skip capturing
        if mute_mic:
            return
        # Convert float32 [-1,1] to int16
        pcm = (indata[:, 0] * 32767).astype('int16')
        loop.call_soon_threadsafe(q.put_nowait, pcm.tobytes())

    try:
        with sd.InputStream(
            device=MIC_DEVICE_INDEX,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=callback,
            blocksize=320,
            latency='low',
        ):
            print("[Client] Microphone streaming...")
            while True:
                try:
                    chunk = await q.get()
                except asyncio.CancelledError:
                    break

                buffer.extend(chunk)
                # Send full 20ms frames, keep leftover in buffer
                while len(buffer) >= frame_bytes:
                    to_send = bytes(buffer[:frame_bytes])
                    del buffer[:frame_bytes]
                    try:
                        await ws.send(to_send)
                    except Exception:
                        return
    except Exception as e:
        print(f"[Client] Error opening microphone: {e}")
        return


async def receiver(ws):
    """
    Receives PCM audio and JSON messages from server.
    Plays back PCM via sounddevice and prints transcripts/tokens.
    """
    PCM_RATE = 48000  # match the TTS sample rate
    try:
        pcm_stream = sd.OutputStream(
            device=PLAYBACK_DEVICE_INDEX,
            samplerate=PCM_RATE,
            channels=1,
            dtype='int16',
            blocksize=320,
            latency='low',
        )
        pcm_stream.start()
    except Exception as e:
        print(f"[Client] Failed to open output stream: {e}")
        return

    global mute_mic

    # Helper to clear mute_mic
    def unmute():
        global mute_mic
        mute_mic = False
        # print("[Client] Mic unmuted")

    try:
        async for message in ws:
            if isinstance(message, bytes):
                # Upon receiving any TTS chunk, immediately mute mic
                mute_mic = True

                # Play the PCM chunk once
                try:
                    pcm_data = np.frombuffer(message, dtype=np.int16)
                    pcm_stream.write(pcm_data)
                except Exception:
                    continue

                # Schedule unmute after chunk plays
                chunk_samples = len(message) // 2  # int16 samples
                chunk_duration_ms = int(chunk_samples / PCM_RATE * 1000)
                unmute_delay = (chunk_duration_ms + 50) / 1000  # in seconds

                asyncio.get_event_loop().call_later(unmute_delay, unmute)

            else:
                # JSON control message
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    continue
                msg_type = msg.get('type')
                if msg_type == 'transcript':
                    label = 'FINAL' if msg.get('final') else 'INTERIM'
                    print(f"\nTRANSCRIPT [{label}]: {msg.get('text')}\n")
                elif msg_type == 'token':
                    print(msg.get('token'), end='', flush=True)
    except websockets.ConnectionClosed:
        pass
    finally:
        pcm_stream.stop()
        pcm_stream.close()


async def main():
    """
    Connects to WebSocket server, spawns mic_sender and receiver tasks.
    """
    try:
        async with websockets.connect(SERVICE_URI) as ws:
            send_task = asyncio.create_task(mic_sender(ws))
            recv_task = asyncio.create_task(receiver(ws))
            done, pending = await asyncio.wait(
                [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
    except Exception as e:
        print(f"[Client] WebSocket connection failed: {e}")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Client] Exiting...")
