import os
import re
import asyncio
import time
import threading
import numpy as np
from scipy.signal import resample_poly
from os.path import exists
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from deepgram_stt import DeepgramTranscriber
from llm_client import LLMClient
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from typing import Optional
from rag_module_integration import RAGEngine
import json

# Environment setup
load_dotenv()
DG_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
PDF_PATH = os.getenv("ZOMATO_PDF_PATH", "Zomato_Annual_Report_2023-24.pdf")
INDEX_PATH = os.getenv("ZOMATO_INDEX_PATH", "zomato_index.faiss")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ElevenLabs client
el_client = ElevenLabs(api_key=ELEVEN_KEY) if ELEVEN_KEY else None

# RAG engine
rag_engine = RAGEngine(
    openai_api_key=OPENAI_KEY,
    pdf_path=PDF_PATH,
    index_path=INDEX_PATH
)
if not exists(INDEX_PATH):
    rag_engine.build_index()
else:
    rag_engine.load_index()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    chosen_language = ws.query_params.get("lang", "en-IN")
    print(f"[Server] Starting STT with language = {chosen_language}")
    chosen_lang_el = "en" if chosen_language=="en-IN" else "hi"

    llm = LLMClient(OPENAI_KEY, rag_engine)
    llm.reset()

    # Stability heuristic state
    last_interim: str = ""
    last_interim_ts: float = 0.0
    spec_launched: bool = False

    # Helper: speak a sentence chunk concurrently
    async def speak_sentence(sentence: str):
        if not el_client:
            return
        first = True
        t0 = None
        try:
            audio_iter = await asyncio.to_thread(
                el_client.text_to_speech.stream,
                text=sentence,
                voice_id=VOICE_ID,
                model_id="eleven_flash_v2_5",
                optimize_streaming_latency=3,
                language_code=chosen_lang_el,
                output_format="pcm_44100",
            )
        except Exception as e:
            print(f"[Server] TTS error: {e}")
            return

        for pcm_chunk in audio_iter:
            if not pcm_chunk:
                continue
            if first:
                t0 = time.perf_counter()
                print(f"[{t0:.3f}] ← first TTS chunk for '{sentence[:30]}...'" )
                first = False
            # arr_22050 = np.frombuffer(pcm_chunk, dtype=np.int16)
            # arr_48000 = resample_poly(arr_22050, up=48000, down=44100)
            # arr_48000 = np.clip(arr_48000, -32768, 32767).astype(np.int16)
            # await ws.send_bytes(arr_48000.tobytes())
            await ws.send_bytes(pcm_chunk)
        if t0:
            print(f"[TIMING] TTS for sentence took {time.perf_counter()-t0:.3f} sec")

    async def keep_silence(ws, frame_size=1600, interval=0.2):
        silence = b"\x00" * frame_size
        while True:
            await ws.send_bytes(silence)
            await asyncio.sleep(interval)

    async def launch_llm(text: str):
        nonlocal spec_launched, last_interim
        t_llm_start = time.perf_counter()
        print(f"[{t_llm_start:.3f}] → sending to LLM (text='{text[:30]}...')")

        response_buffer = ""
        first_token_logged = False
        last_token_ts = None
        try:
            async for token in llm.stream_response(text):
                now = time.perf_counter()
                if not first_token_logged and token:
                    print(f"[{now:.3f}] ← first LLM token")
                    first_token_logged = True
                last_token_ts = now
                await ws.send_json({"type": "token", "token": token})

                response_buffer += token
                # flush each complete sentence immediately
                while True:
                    m = re.search(r"([\.\!,?])(\s|$)", response_buffer)
                    if not m:
                        break
                    end = m.end()
                    sentence = response_buffer[:end].strip()
                    print(f"[{time.perf_counter():.3f}] [Buffer] flush: '{sentence}'")
                    asyncio.create_task(speak_sentence(sentence))
                    await asyncio.sleep(1)
                    response_buffer = response_buffer[end:]

            if last_token_ts:
                print(f"[{last_token_ts:.3f}] ← last LLM token")
            if response_buffer.strip():
                final_sent = response_buffer.strip()
                print(f"[{time.perf_counter():.3f}] [Buffer] final flush: '{final_sent}'")
                asyncio.create_task(speak_sentence(final_sent))

            await ws.send_json({"type": "response_end"})
        finally:
            t_llm_end = time.perf_counter()
            print(f"[TIMING] LLM inference took {t_llm_end - t_llm_start:.3f} sec")
            spec_launched = False
            last_interim = ""

    async def on_transcript(text: str, is_final: bool):
        nonlocal last_interim, last_interim_ts, spec_launched
        await ws.send_json({"type": "transcript", "text": text, "final": is_final})
        now = time.perf_counter()

        if not is_final:
            if len(text.split()) >= 3:
                if text == last_interim and (now - last_interim_ts) >= 0.15 and not spec_launched:
                    spec_launched = True
                    print(f"[{now:.3f}] [STT] stable interim: '{text}'")
                    asyncio.create_task(launch_llm(text))
                elif text != last_interim:
                    last_interim = text
                    last_interim_ts = now
            return

        if not spec_launched:
            spec_launched = True
            print(f"[{now:.3f}] [STT] final transcript kickoff: '{text}'")
            asyncio.create_task(launch_llm(text))

    # Start STT
    stt = DeepgramTranscriber(DG_KEY, use_mic=False, language=chosen_language)
    llm.reset()
    # initial greeting
    lang = "Hindi" if chosen_language == "multi" else "English"
    asyncio.create_task(on_transcript(f"greet in {lang}", True))
    threading.Thread(target=lambda: asyncio.run(stt.start(on_transcript)), daemon=True).start()

    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                stt.feed_audio(msg["bytes"])
            elif msg.get("type") == "websocket.disconnect":
                break
    finally:
        stt.finish()
        await ws.close()
