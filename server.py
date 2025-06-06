'''Working Code (Checked on Terminal)'''
# import os
# import re
# import asyncio
# import time
# import threading
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from os.path import exists
# from deepgram_stt import DeepgramTranscriber
# from llm_client import LLMClient
# from elevenlabs.client import ElevenLabs
# from dotenv import load_dotenv
# from typing import Optional
# from rag_module_integration import RAGEngine

# # Load environment variables from .env
# load_dotenv()
# DG_KEY = os.getenv("DEEPGRAM_API_KEY")
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
# VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
# PDF_PATH = os.getenv("ZOMATO_PDF_PATH", "Zomato_Annual_Report_2023-24.pdf")
# # INDEX_PATH = os.getenv("ZOMATO_INDEX_PATH", "zomato_index.faiss")
# INDEX_DIR = "zomato_hnsw_index"
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Instantiate ElevenLabs client once (shared)
# el_client = ElevenLabs(api_key=ELEVEN_KEY) if ELEVEN_KEY else None

# #Initialize RAG engine and build/load index at startup
# dag_rag_engine = RAGEngine(
#     openai_api_key=OPENAI_KEY,
#     pdf_path=PDF_PATH,
#     # index_path=INDEX_PATH
#     index_dir=INDEX_DIR
# )
# dag_rag_engine.build_index()
# # if not exists(INDEX_PATH):
# #     dag_rag_engine.build_index()
# # else:
# #     dag_rag_engine.load_index()

# @app.websocket("/ws")
# async def websocket_endpoint(ws: WebSocket):
#     """
#     WebSocket endpoint to handle bidirectional streaming:
#     - Receive PCM from client -> feed to DeepgramTranscriber
#     - On transcription final -> invoke LLM to generate response tokens
#       -> chunk tokens into sentences -> send each chunk to ElevenLabs TTS -> stream back PCM to client
#     """
#     await ws.accept()
#     stt = DeepgramTranscriber(DG_KEY, use_mic=False)
#     llm = LLMClient(OPENAI_KEY, dag_rag_engine)

#     last_sent_text = None
#     stable_count = 0
#     last_user_end_ts: Optional[float] = None
#     async def on_transcript(text: str, is_final: bool):
#         # nonlocal last_sent_text, stable_count
#         # utterance_in_progress = True
#         nonlocal last_user_end_ts
#         """
#         Called when Deepgram yields a transcription. Sends JSON transcript to client.
#         If final, generates LLM response, breaks into chunks, streams TTS.
#         """
#         # Forward transcription to client
#         try:
#             await ws.send_json({"type": "transcript", "text": text, "final": is_final})
#         except Exception:
#             return

#         # Only proceed if this is a final transcription
#         if not is_final:
#         #     if utterance_in_progress:
#         #         if text == last_sent_text:
#         #             stable_count += 1
#         #         else:
#         #             last_sent_text = text
#         #             stable_count = 0

#         #         # If we've seen the same interim 2 or 3 times in a row, assume it's “stable”
#         #         if stable_count >= 2:
#         #             await send_to_llm(text)
#         #             stable_count = 0  # reset so we don’t resend immediately
#             return

#         # if not utterance_in_progress:
#         #     utterance_in_progress = True
#         #     current_llm_task = asyncio.create_task(send_to_llm(text))

#         last_user_end_ts = time.perf_counter()
#         # Generate conversational response from LLM
#         t_llm_start = time.perf_counter()
#         response_buffer = ''
#         # A queue to send text chunks to speaker
#         chunk_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

#         # async def speaker_task():
#         #     """
#         #     Consumes text chunks from queue, streams TTS PCM back to client.
#         #     """
#         #     if not el_client:
#         #         return
#         #     while True:
#         #         chunk_text = await chunk_queue.get()
#         #         if chunk_text is None:
#         #             break
#         #         try:
#         #             audio_stream = el_client.text_to_speech.stream(
#         #                 text=chunk_text,
#         #                 voice_id=VOICE_ID,
#         #                 model_id="eleven_flash_v2_5",
#         #                 optimize_streaming_latency=3,
#         #                 output_format="pcm_22050",
#         #             )
#         #         except Exception as e:
#         #             print(f"[Server] TTS error: {e}")
#         #             continue

#         #         for pcm_chunk in audio_stream:
#         #             if pcm_chunk:
#         #                 try:
#         #                     await ws.send_bytes(pcm_chunk)
#         #                 except Exception:
#         #                     break

#         # # Launch speaker coroutine concurrently
#         # speaker = asyncio.create_task(speaker_task())

#         # # Stream LLM response tokens
#         # try:
#         #     async for token in llm.stream_response(text):
#         #         response_buffer += token
#         #         # Send each token as it arrives for real-time display
#         #         try:
#         #             await ws.send_json({"type": "token", "token": token})
#         #         except Exception:
#         #             pass

#         #         # Check for sentence boundary (punctuation + whitespace) or buffer length >200
#         #         if re.search(r"[\.!?]$", response_buffer) or len(response_buffer) > 200:
#         #             # Find end of last complete sentence or take up to 200 chars
#         #             m = list(re.finditer(r"[\.!?]", response_buffer))
#         #             if m and len(response_buffer) > 20:
#         #                 idx = m[-1].end()
#         #             else:
#         #                 idx = min(len(response_buffer), 200)
#         #             to_tts = response_buffer[:idx]
#         #             response_buffer = response_buffer[idx:]
#         #             await chunk_queue.put(to_tts)

#         #     # Any remaining text in buffer
#         #     if response_buffer:
#         #         await chunk_queue.put(response_buffer)
#         # except Exception as e:
#         #     print(f"[Server] LLM streaming error: {e}")
#         # finally:
#         #     # Signal speaker to finish and wait
#         #     await chunk_queue.put(None)
#         #     await speaker

#         async def speaker_task():
#             if not el_client:
#                 return
#             first_chunk_sent = False
#             t_tts_start = time.perf_counter()
#             while True:
#                 chunk_text = await chunk_queue.get()
#                 if chunk_text is None:
#                     break

#                 try:
#                     audio_stream = el_client.text_to_speech.stream(
#                         text=chunk_text,
#                         voice_id=VOICE_ID,
#                         model_id="eleven_flash_v2_5",
#                         optimize_streaming_latency=2,
#                         output_format="pcm_22050",
#                     )
#                 except Exception as e:
#                     print(f"[Server] TTS error: {e}")
#                     continue

#                 for pcm_chunk in audio_stream:
#                     if pcm_chunk:
#                         if not first_chunk_sent:
#                             bot_start_ts = time.perf_counter()
#                             if last_user_end_ts is not None:
#                                 user_to_bot_latency = bot_start_ts - last_user_end_ts
#                                 print(f"[TIMING] User→Bot start latency: {user_to_bot_latency:.3f} sec")
#                             else:
#                                 print("[TIMING] Warning: last_user_end_ts was None, cannot compute user→bot latency")
#                             first_chunk_sent = True
#                         try:
#                             await ws.send_bytes(pcm_chunk)
#                         except Exception:
#                             break
            
#             t_tts_end = time.perf_counter()
#             tts_time = t_tts_end - t_tts_start
#             print(f"[TIMING] Total TTS/speaking took {tts_time:.3f} sec")

#         speaker = asyncio.create_task(speaker_task())

#         # Now stream tokens from the LLM, sentence-by-sentence:
#         try:
#             async for token in llm.stream_response(text):
#                 response_buffer += token

#                 # Immediately send each token to client as “token” for transcription display:
#                 try:
#                     await ws.send_json({"type": "token", "token": token})
#                 except Exception:
#                     pass

#                 # Check if the buffer now ends in a sentence terminator:
#                 # (You can adjust the regex to catch “. ”, “! ”, “? ”, or end-of-text.)
#                 # Here we look for any of .?! followed by whitespace or end of buffer.
#                 sentence_end_match = re.search(r"([\.!?])(\s|$)", response_buffer)
#                 if sentence_end_match:
#                     # Find the position of the FIRST sentence boundary
#                     # (so we break off one sentence at a time)
#                     idx = sentence_end_match.end()

#                     # Extract the first full sentence
#                     sentence = response_buffer[:idx].strip()
#                     response_buffer = response_buffer[idx:]  # keep remainder

#                     # Queue that sentence for immediate TTS
#                     await chunk_queue.put(sentence)

#             # After the LLM stream is finished, if anything remains (a partial sentence),
#             # send it too:
#             if response_buffer.strip():
#                 await chunk_queue.put(response_buffer.strip())

#         except Exception as e:
#             print(f"[Server] LLM streaming error: {e}")
#         finally:
#             t_llm_end = time.perf_counter()
#             llm_time = t_llm_end - t_llm_start
#             print(f"[TIMING] LLM inference took {llm_time:.3f} sec")
#             # Signal the speaker to finish, then wait for it to drain
#             await chunk_queue.put(None)
#             await speaker
        


#         ''''''

#     # Kick off initial greeting by sending an "empty" final to produce system prompt-based greeting
#     llm.reset()
#     asyncio.create_task(on_transcript("", True))

#     # Run STT in a background thread
#     def stt_runner():
#         asyncio.run(stt.start(on_transcript))

#     stt_thread = threading.Thread(target=stt_runner, daemon=True)
#     stt_thread.start()

#     try:
#         # Main loop: receive binary PCM frames from client and feed to Deepgram
#         while True:
#             try:
#                 data = await ws.receive_bytes()
#                 stt.feed_audio(data)
#             except WebSocketDisconnect:
#                 break
#             except Exception:
#                 continue
#     finally:
#         # Cleanup on disconnect
#         stt.finish()
#         stt_thread.join(timeout=2)
#         await ws.close()



'''Code changes for UI Integration'''

# import os
# import re
# import asyncio
# import time
# import threading
# import numpy as np
# from scipy.signal import resample_poly
# from os.path import exists
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# from deepgram_stt import DeepgramTranscriber
# from llm_client import LLMClient
# from elevenlabs.client import ElevenLabs
# from dotenv import load_dotenv
# from typing import Optional
# from rag_module_integration import RAGEngine

# # Load environment variables from .env
# load_dotenv()
# DG_KEY = os.getenv("DEEPGRAM_API_KEY")
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
# VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
# PDF_PATH = os.getenv("ZOMATO_PDF_PATH", "Zomato_Annual_Report_2023-24.pdf")
# INDEX_PATH = os.getenv("ZOMATO_INDEX_PATH", "zomato_index.faiss")

# app = FastAPI()

# # Allow CORS (in case you test from a different origin)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve index.html at GET /
# @app.get("/")
# async def read_index():
#     return FileResponse("static/index.html")

# # Mount everything under /static for CSS/JS/images
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Instantiate ElevenLabs client once (shared)
# el_client = ElevenLabs(api_key=ELEVEN_KEY) if ELEVEN_KEY else None

# # Initialize RAG engine and build or load index at startup
# dag_rag_engine = RAGEngine(
#     openai_api_key=OPENAI_KEY,
#     pdf_path=PDF_PATH,
#     index_path=INDEX_PATH
# )

# if not exists(INDEX_PATH):
#     dag_rag_engine.build_index()
# else:
#     dag_rag_engine.load_index()

# @app.websocket("/ws")
# async def websocket_endpoint(ws: WebSocket):
#     """
#     WebSocket endpoint to handle bidirectional streaming:
#     - Receive PCM from client → feed to DeepgramTranscriber
#     - On final transcripts → invoke LLM → break it into sentences → send each sentence to ElevenLabs TTS → stream PCM back
#     """
#     await ws.accept()
#     stt = DeepgramTranscriber(DG_KEY, use_mic=False)
#     llm = LLMClient(OPENAI_KEY, dag_rag_engine)

#     last_user_end_ts: Optional[float] = None

#     async def on_transcript(text: str, is_final: bool):
#         nonlocal last_user_end_ts

#         # 1) Forward transcript (interim/final) to browser
#         try:
#             await ws.send_json({"type": "transcript", "text": text, "final": is_final})
#         except Exception:
#             return

#         # Only act when it's a final transcript
#         if not is_final:
#             return

#         # 2) Note the time user’s utterance ended
#         last_user_end_ts = time.perf_counter()

#         # 3) Generate LLM response → chunk → TTS → send PCM
#         t_llm_start = time.perf_counter()
#         response_buffer = ""
#         chunk_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

#         async def speaker_task():
#             if not el_client:
#                 return
#             first_chunk_sent = False
#             t_tts_start = time.perf_counter()
#             while True:
#                 chunk_text = await chunk_queue.get()
#                 if chunk_text is None:
#                     break

#                 try:
#                     audio_stream = el_client.text_to_speech.stream(
#                         text=chunk_text,
#                         voice_id=VOICE_ID,
#                         model_id="eleven_flash_v2_5",
#                         # optimize_streaming_latency=2,
#                         output_format="pcm_44100",
#                     )
#                 except Exception as e:
#                     print(f"[Server] TTS error: {e}")
#                     continue

#                 for pcm_chunk in audio_stream:
#                     if pcm_chunk:
#                         if not first_chunk_sent:
#                             bot_start_ts = time.perf_counter()
#                             if last_user_end_ts is not None:
#                                 user_to_bot_latency = bot_start_ts - last_user_end_ts
#                                 print(f"[TIMING] User→Bot start latency: {user_to_bot_latency:.3f} sec")
#                             else:
#                                 print("[TIMING] Warning: last_user_end_ts was None")
#                             first_chunk_sent = True
#                         # Convert raw bytes → numpy array @ 22 050 Hz
#                         try:
#                             arr_22050 = np.frombuffer(pcm_chunk, dtype=np.int16)
#                         except Exception as e:
#                             print(f"[Server] Failed to parse TTS chunk into int16: {e}")
#                             continue

#                         # Resample 22 050 → 48 000 via resample_poly
#                         try:
#                             arr_48000 = resample_poly(arr_22050, up=48000, down=44100)
#                         except Exception as e:
#                             print(f"[Server] Resampling error: {e}")
#                             arr_48000 = arr_22050

#                         # Clip to int16 range just in case
#                         arr_48000 = np.clip(arr_48000, -32768, 32767).astype(np.int16)

#                         # Send the upsampled bytes
#                         try:
#                             # await ws.send_bytes(pcm_chunk)
#                             await ws.send_bytes(arr_48000.tobytes())
#                         except Exception:
#                             break

#             t_tts_end = time.perf_counter()
#             tts_time = t_tts_end - t_tts_start
#             print(f"[TIMING] Total TTS/speaking took {tts_time:.3f} sec")

#         speaker = asyncio.create_task(speaker_task())

#         # 4) Stream tokens from the LLM, break into sentences
#         try:
#             async for token in llm.stream_response(text):
#                 response_buffer += token

#                 # Send each token to browser
#                 try:
#                     await ws.send_json({"type": "token", "token": token})
#                 except Exception:
#                     pass

#                 # If buffer ends in . ! or ? (plus whitespace/end), send that sentence to TTS
#                 sentence_end_match = re.search(r"([\.!?])(\s|$)", response_buffer)
#                 if sentence_end_match:
#                     idx = sentence_end_match.end()
#                     sentence = response_buffer[:idx].strip()
#                     response_buffer = response_buffer[idx:]
#                     await chunk_queue.put(sentence)

#             # 5) If any leftover text remains after streaming, send it too
#             if response_buffer.strip():
#                 await chunk_queue.put(response_buffer.strip())

#             # 6) Send response_end message after all tokens are streamed
#             await ws.send_json({"type": "response_end"})

#         except Exception as e:
#             print(f"[Server] LLM streaming error: {e}")
#         finally:
#             t_llm_end = time.perf_counter()
#             llm_time = t_llm_end - t_llm_start
#             print(f"[TIMING] LLM inference took {llm_time:.3f} sec")
#             await chunk_queue.put(None)
#             await speaker

#     # Kick off an initial greeting by sending an “empty” final transcript
#     llm.reset()
#     asyncio.create_task(on_transcript("", True))

#     # Run STT in a separate thread (DeepgramTranscriber.start blocks)
#     def stt_runner():
#         asyncio.run(stt.start(on_transcript))

#     stt_thread = threading.Thread(target=stt_runner, daemon=True)
#     stt_thread.start()

#     try:
#         #Main loop: receive raw PCM frames from browser → feed to Deepgram
#         while True:
#             try:
#                 data = await ws.receive_bytes()
#                 stt.feed_audio(data)
#             except WebSocketDisconnect:
#                 break
#             except Exception:
#                 continue
#     finally:
#         # Cleanup on disconnect
#         stt.finish()
#         stt_thread.join(timeout=2)
#         await ws.close()



'''Code change for UI with Language Change Option'''
# import os
# import re
# import asyncio
# import time
# import threading
# import numpy as np
# from scipy.signal import resample_poly
# from os.path import exists
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# from deepgram_stt import DeepgramTranscriber
# from llm_client import LLMClient
# from elevenlabs.client import ElevenLabs
# from dotenv import load_dotenv
# from typing import Optional
# from rag_module_integration import RAGEngine
# import json  # ← need this to parse/set language messages

# # Load environment variables from .env
# load_dotenv()
# DG_KEY = os.getenv("DEEPGRAM_API_KEY")
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
# VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
# PDF_PATH = os.getenv("ZOMATO_PDF_PATH", "Zomato_Annual_Report_2023-24.pdf")
# INDEX_PATH = os.getenv("ZOMATO_INDEX_PATH", "zomato_index.faiss")

# app = FastAPI()

# # Allow CORS (in case you test from a different origin)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve index.html at GET /
# @app.get("/")
# async def read_index():
#     return FileResponse("static/index.html")

# # Mount everything under /static for CSS/JS/images
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Instantiate ElevenLabs client once (shared)
# el_client = ElevenLabs(api_key=ELEVEN_KEY) if ELEVEN_KEY else None

# # Initialize RAG engine and build or load index at startup
# dag_rag_engine = RAGEngine(
#     openai_api_key=OPENAI_KEY,
#     pdf_path=PDF_PATH,
#     index_path=INDEX_PATH
# )

# if not exists(INDEX_PATH):
#     dag_rag_engine.build_index()
# else:
#     dag_rag_engine.load_index()

# @app.websocket("/ws")
# async def websocket_endpoint(ws: WebSocket):
#     """
#     WebSocket endpoint to handle bidirectional streaming:
#     - Receive PCM from client → feed to DeepgramTranscriber
#     - On final transcripts → invoke LLM → break it into sentences → send each sentence to ElevenLabs TTS → stream PCM back
#     - Allows client to send {"type":"set_language","language":"en-IN"|"hi"|"multi"} at any time
#       to restart DeepgramTranscriber with a new language.
#     """
#     await ws.accept()

#     # 1) Create STT client with a default language ("en-IN")
#     stt = DeepgramTranscriber(DG_KEY, use_mic=False, language="en-IN")
#     llm = LLMClient(OPENAI_KEY, dag_rag_engine)

#     last_user_end_ts: Optional[float] = None

#     async def on_transcript(text: str, is_final: bool):
#         nonlocal last_user_end_ts

#         # 1) Forward transcript (interim/final) to browser
#         try:
#             await ws.send_json({"type": "transcript", "text": text, "final": is_final})
#         except Exception:
#             return

#         # Only act when it's a final transcript
#         if not is_final:
#             return

#         # 2) Note the time user’s utterance ended
#         last_user_end_ts = time.perf_counter()

#         # 3) Generate LLM response → chunk → TTS → send PCM
#         t_llm_start = time.perf_counter()
#         response_buffer = ""
#         chunk_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

#         async def speaker_task():
#             if not el_client:
#                 return
#             first_chunk_sent = False
#             t_tts_start = time.perf_counter()
#             while True:
#                 chunk_text = await chunk_queue.get()
#                 if chunk_text is None:
#                     break

#                 try:
#                     audio_stream = el_client.text_to_speech.stream(
#                         text=chunk_text,
#                         voice_id=VOICE_ID,
#                         model_id="eleven_flash_v2_5",
#                         # optimize_streaming_latency=2,
#                         output_format="pcm_44100",
#                     )
#                 except Exception as e:
#                     print(f"[Server] TTS error: {e}")
#                     continue

#                 for pcm_chunk in audio_stream:
#                     if pcm_chunk:
#                         if not first_chunk_sent:
#                             bot_start_ts = time.perf_counter()
#                             if last_user_end_ts is not None:
#                                 user_to_bot_latency = bot_start_ts - last_user_end_ts
#                                 print(f"[TIMING] User→Bot start latency: {user_to_bot_latency:.3f} sec")
#                             else:
#                                 print("[TIMING] Warning: last_user_end_ts was None")
#                             first_chunk_sent = True
#                         # Convert raw bytes → numpy array @ 22 050 Hz
#                         try:
#                             arr_22050 = np.frombuffer(pcm_chunk, dtype=np.int16)
#                         except Exception as e:
#                             print(f"[Server] Failed to parse TTS chunk into int16: {e}")
#                             continue

#                         # Resample 22 050 → 48 000 via resample_poly
#                         try:
#                             arr_48000 = resample_poly(arr_22050, up=48000, down=44100)
#                         except Exception as e:
#                             print(f"[Server] Resampling error: {e}")
#                             arr_48000 = arr_22050

#                         # Clip to int16 range just in case
#                         arr_48000 = np.clip(arr_48000, -32768, 32767).astype(np.int16)

#                         # Send the upsampled bytes
#                         try:
#                             await ws.send_bytes(arr_48000.tobytes())
#                         except Exception:
#                             break

#             t_tts_end = time.perf_counter()
#             tts_time = t_tts_end - t_tts_start
#             print(f"[TIMING] Total TTS/speaking took {tts_time:.3f} sec")

#         speaker = asyncio.create_task(speaker_task())

#         # 4) Stream tokens from the LLM, break into sentences
#         try:
#             async for token in llm.stream_response(text):
#                 response_buffer += token

#                 # Send each token to browser
#                 try:
#                     await ws.send_json({"type": "token", "token": token})
#                 except Exception:
#                     pass

#                 # If buffer ends in . ! or ? (plus whitespace/end), send that sentence to TTS
#                 sentence_end_match = re.search(r"([\.!?])(\s|$)", response_buffer)
#                 if sentence_end_match:
#                     idx = sentence_end_match.end()
#                     sentence = response_buffer[:idx].strip()
#                     response_buffer = response_buffer[idx:]
#                     await chunk_queue.put(sentence)

#             # 5) If any leftover text remains after streaming, send it too
#             if response_buffer.strip():
#                 await chunk_queue.put(response_buffer.strip())

#             # 6) Send response_end message after all tokens are streamed
#             await ws.send_json({"type": "response_end"})

#         except Exception as e:
#             print(f"[Server] LLM streaming error: {e}")
#         finally:
#             t_llm_end = time.perf_counter()
#             llm_time = t_llm_end - t_llm_start
#             print(f"[TIMING] LLM inference took {llm_time:.3f} sec")
#             await chunk_queue.put(None)
#             await speaker

#     # Kick off an initial greeting by sending an “empty” final transcript
#     llm.reset()
#     asyncio.create_task(on_transcript("", True))

#     # ─── Start STT in a separate thread ──────────────────────────────────────
#     def stt_runner():
#         asyncio.run(stt.start(on_transcript))

#     stt_thread = threading.Thread(target=stt_runner, daemon=True)
#     stt_thread.start()

#     try:
#         # ─── Main loop: now use ws.receive() so we can detect text vs. bytes ───────
#         while True:
#             msg = await ws.receive()

#             if msg["type"] == "websocket.receive":
#                 # (a) If this is a text frame, check for set_language
#                 if "text" in msg:
#                     try:
#                         m = json.loads(msg["text"])
#                         if m.get("type") == "set_language" and "language" in m:
#                             new_lang = m["language"]  # "en-IN", "hi", or "multi"
#                             print(f"[Server] Received set_language → {new_lang}")

#                             # 1) Stop the old STT thread:
#                             stt.finish()
#                             stt_thread.join(timeout=1)

#                             # 2) Create a brand‐new DeepgramTranscriber with the new language
#                             stt = DeepgramTranscriber(DG_KEY, use_mic=False, language=new_lang)

#                             # 3) Restart STT in a fresh thread
#                             stt_thread = threading.Thread(target=lambda: asyncio.run(stt.start(on_transcript)), daemon=True)
#                             stt_thread.start()

#                             # 4) Acknowledge back to client that language changed
#                             await ws.send_json({"type": "language_ack", "language": new_lang})
#                             continue  # skip feeding audio this iteration

#                     except (json.JSONDecodeError, KeyError):
#                         # Not a valid JSON or missing fields → ignore
#                         pass

#                 # (b) If this is a binary frame, it's PCM audio
#                 if "bytes" in msg:
#                     stt.feed_audio(msg["bytes"])

#             elif msg["type"] == "websocket.disconnect":
#                 break

#     finally:
#         # Cleanup on disconnect
#         stt.finish()
#         stt_thread.join(timeout=2)
#         await ws.close()

'''Try 2'''

import os
import re
import asyncio
import time
import threading
import numpy as np
from scipy.signal import resample_poly
from os.path import exists
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from deepgram_stt import DeepgramTranscriber
from llm_client import LLMClient
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from typing import Optional
from rag_module_integration import RAGEngine
import json  # for completeness, though not used here

# Load environment variables from .env
load_dotenv()
DG_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
PDF_PATH = os.getenv("ZOMATO_PDF_PATH", "Zomato_Annual_Report_2023-24.pdf")
INDEX_PATH = os.getenv("ZOMATO_INDEX_PATH", "zomato_index.faiss")

app = FastAPI()

# Allow CORS (in case you test from a different origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at GET /
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Mount everything under /static for CSS/JS/images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instantiate ElevenLabs client once (shared)
el_client = ElevenLabs(api_key=ELEVEN_KEY) if ELEVEN_KEY else None

# Initialize RAG engine and build or load index at startup
dag_rag_engine = RAGEngine(
    openai_api_key=OPENAI_KEY,
    pdf_path=PDF_PATH,
    index_path=INDEX_PATH
)

if not exists(INDEX_PATH):
    dag_rag_engine.build_index()
else:
    dag_rag_engine.load_index()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint to handle bidirectional streaming:
    1) Read the `lang` query parameter (en-IN, hi, or multi).
    2) Instantiate DeepgramTranscriber(...) with that language and stt.start().
    3) In the main loop, feed each binary frame to stt.feed_audio().
    4) on_transcript → LLM → chunk → TTS → send TTS PCM back to client.
    """
    await ws.accept()

    # ─── Read the “lang” query parameter on WS URL ─────────────────────────
    # FastAPI’s WebSocket has `ws.query_params` as a MultiDict
    params = ws.query_params
    chosen_language = params.get("lang", "en-IN")  # default to en-IN if missing
    print(f"[Server] Starting STT with language = {chosen_language}")

    # Set up LLM client
    llm = LLMClient(OPENAI_KEY, dag_rag_engine)
    last_user_end_ts: Optional[float] = None

    async def on_transcript(text: str, is_final: bool):
        nonlocal last_user_end_ts

        # 1) Forward transcript (interim/final) to browser
        try:
            await ws.send_json({"type": "transcript", "text": text, "final": is_final})
        except Exception:
            return

        # If not final, do nothing else
        if not is_final:
            return

        # 2) Note the time user’s utterance ended
        last_user_end_ts = time.perf_counter()

        # 3) Generate LLM response → chunk → TTS → send PCM
        t_llm_start = time.perf_counter()
        response_buffer = ""
        chunk_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def speaker_task():
            if not el_client:
                return
            first_chunk_sent = False
            t_tts_start = time.perf_counter()
            while True:
                chunk_text = await chunk_queue.get()
                if chunk_text is None:
                    break

                try:
                    audio_stream = el_client.text_to_speech.stream(
                        text=chunk_text,
                        voice_id=VOICE_ID,
                        model_id="eleven_flash_v2_5",
                        # optimize_streaming_latency=2,
                        output_format="pcm_44100",
                    )
                except Exception as e:
                    print(f"[Server] TTS error: {e}")
                    continue

                for pcm_chunk in audio_stream:
                    if pcm_chunk:
                        if not first_chunk_sent:
                            bot_start_ts = time.perf_counter()
                            if last_user_end_ts is not None:
                                user_to_bot_latency = bot_start_ts - last_user_end_ts
                                print(f"[TIMING] User→Bot start latency: {user_to_bot_latency:.3f} sec")
                            else:
                                print("[TIMING] Warning: last_user_end_ts was None")
                            first_chunk_sent = True
                        # Convert raw bytes → numpy array @ 22 050 Hz
                        try:
                            arr_22050 = np.frombuffer(pcm_chunk, dtype=np.int16)
                        except Exception as e:
                            print(f"[Server] Failed to parse TTS chunk into int16: {e}")
                            continue

                        # Resample 22 050 → 48 000 via resample_poly
                        try:
                            arr_48000 = resample_poly(arr_22050, up=48000, down=44100)
                        except Exception as e:
                            print(f"[Server] Resampling error: {e}")
                            arr_48000 = arr_22050

                        # Clip to int16 range just in case
                        arr_48000 = np.clip(arr_48000, -32768, 32767).astype(np.int16)

                        # Send the upsampled bytes
                        try:
                            await ws.send_bytes(arr_48000.tobytes())
                        except Exception:
                            break

            t_tts_end = time.perf_counter()
            tts_time = t_tts_end - t_tts_start
            print(f"[TIMING] Total TTS/speaking took {tts_time:.3f} sec")

        speaker = asyncio.create_task(speaker_task())

        # 4) Stream tokens from the LLM, break into sentences
        try:
            async for token in llm.stream_response(text):
                response_buffer += token

                # Send each token to browser
                try:
                    await ws.send_json({"type": "token", "token": token})
                except Exception:
                    pass

                # If buffer ends in . ! or ? (plus whitespace/end), send that sentence to TTS
                sentence_end_match = re.search(r"([\.!?])(\s|$)", response_buffer)
                if sentence_end_match:
                    idx = sentence_end_match.end()
                    sentence = response_buffer[:idx].strip()
                    response_buffer = response_buffer[idx:]
                    await chunk_queue.put(sentence)

            # 5) If any leftover text remains after streaming, send it too
            if response_buffer.strip():
                await chunk_queue.put(response_buffer.strip())

            # 6) Send response_end message
            await ws.send_json({"type": "response_end"})

        except Exception as e:
            print(f"[Server] LLM streaming error: {e}")
        finally:
            t_llm_end = time.perf_counter()
            llm_time = t_llm_end - t_llm_start
            print(f"[TIMING] LLM inference took {llm_time:.3f} sec")
            await chunk_queue.put(None)
            await speaker

    # ─── Create and start STT immediately with the chosen_language ─────────────────
    stt = DeepgramTranscriber(DG_KEY, use_mic=False, language=chosen_language)
    llm.reset()
    asyncio.create_task(on_transcript("", True))
    def stt_runner():
        asyncio.run(stt.start(on_transcript))

    stt_thread = threading.Thread(target=stt_runner, daemon=True)
    stt_thread.start()

    try:
        # ─── Main loop: feed incoming audio frames to STT ────────────────────────────
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.receive":
                if "bytes" in msg:
                    stt.feed_audio(msg["bytes"])
                # We no longer expect any text, since language is fixed via URL.
            elif msg["type"] == "websocket.disconnect":
                break

    finally:
        # Cleanup on disconnect
        stt.finish()
        stt_thread.join(timeout=2)
        await ws.close()
