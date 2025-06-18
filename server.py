import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import asyncio
import time
import threading
import numpy as np
from os.path import exists
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from deepgram_stt import DeepgramTranscriber
from llm_client import LLMClient
from tts_client import TTSClient
from dotenv import load_dotenv
from typing import List, Optional

from rag_module_integration import RAGEngine
import json

# Environment setup
load_dotenv()
DG_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
PDF_PATH = os.getenv("ZOMATO_PDF_PATH", "Zomato_Annual_Report_2023-24.pdf")
# INDEX_PATH = os.getenv("ZOMATO_INDEX_PATH", "zomato_index.faiss")
INDEX_PATH = os.getenv("ZOMATO_INDEX_PATH", "zomato_hnsw_index")

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

tts_client = TTSClient(api_key=ELEVEN_KEY, voice_id=VOICE_ID)

# RAG engine
rag_engine = RAGEngine(
    openai_api_key=OPENAI_KEY,
    pdf_path=PDF_PATH,
    index_path=INDEX_PATH
)

idx_file  = f"{INDEX_PATH}/hnsw_index.bin"
data_file = f"{INDEX_PATH}/id2text.pkl"
meta_file = f"{INDEX_PATH}/meta.pkl"

if not (exists(idx_file) and exists(data_file) and exists(meta_file)):
    print("[RAG] index files missing, building from scratch…")
    rag_engine.build_index()
else:
    rag_engine.load_index(idx_file, data_file, meta_file)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    chosen_language = ws.query_params.get("lang", "en-IN")
    print(f"[Server] Starting STT with language = {chosen_language}")
    chosen_lang_el = "en" if chosen_language=="en-IN" else "hi"
    user_phone = "6306061252"
    llm = LLMClient(GROQ_KEY, OPENAI_KEY, rag_engine, customer_phone=user_phone)

    transcripts: List[str] = []
    last_interim: str = ""
    last_interim_ts: float = 0.0
    spec_launched: bool = False
    last_user_end_ts: Optional[float] = None
    # idle_task: Optional[asyncio.Task] = None
    current_speech_tasks: set[asyncio.Task] = set()

    # def schedule_idle():
    #     nonlocal idle_task
    #     # cancelling any old one
    #     if idle_task:
    #         idle_task.cancel()

    #     idle_task = asyncio.create_task(idle_watcher())
        # print("idle_task done")

    async def speak_sentence(sentence: str):
        # Helper: speak a sentence chunk using TTSClient
        # nonlocal idle_task
        if not tts_client:
            return
        # Cancel any existing idle watcher
        # if idle_task:
        #     idle_task.cancel()
        #     idle_task = None
        # Delegate to TTSClient for streaming and WebSocket forwarding
        await tts_client.speak_sentence(
            sentence=sentence,
            ws=ws,
            chosen_lang_el=chosen_lang_el,
            last_user_end_ts=last_user_end_ts,
        )

    # async def idle_watcher():
    #     try:
    #         await asyncio.sleep(5.0)
    #         asyncio.create_task(speak_sentence("क्या आप अभी भी वहाँ हैं?" if chosen_lang_el=="hi"
    #                             else "Hi, are you still there?"))

    #     except asyncio.CancelledError:
    #         # user spoke or agent spoke again—just exit
    #         return

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
                    m = re.search(r"([\.\!,?|])(\s|$)", response_buffer)
                    if not m:
                        break
                    end = m.end()
                    sentence = response_buffer[:end].strip()
                    print(f"[{time.perf_counter():.3f}] [Buffer] flush: '{sentence}'")
                    task = asyncio.create_task(speak_sentence(sentence))
                    current_speech_tasks.add(task)
                    task.add_done_callback(lambda fut: current_speech_tasks.discard(fut))
                    # task.add_done_callback(lambda fut: schedule_idle())

                    # asyncio.create_task(speak_sentence(sentence))
                    await asyncio.sleep(0.1)
                    transcripts.append(response_buffer)
                    response_buffer = response_buffer[end:]

            if last_token_ts:
                print(f"[{last_token_ts:.3f}] ← last LLM token")
            if response_buffer.strip():
                final_sent = response_buffer.strip()
                print(f"[{time.perf_counter():.3f}] [Buffer] final flush: '{final_sent}'")
                speak_task = asyncio.create_task(speak_sentence(final_sent))
                # speak_task.add_done_callback(lambda fut: schedule_idle())
            await ws.send_json({"type": "response_end"})
        finally:
            t_llm_end = time.perf_counter()
            print(f"[TIMING] LLM inference took {t_llm_end - t_llm_start:.3f} sec")
            spec_launched = False
            last_interim = ""

    async def on_transcript(text: str, is_final: bool):
        nonlocal last_interim, last_interim_ts, spec_launched, last_user_end_ts
        # if idle_task:
        #     idle_task.cancel()
        #     idle_task = None
        spec_launched=False
        await ws.send_json({"type": "transcript", "text": text, "final": is_final})
        # record for end‑of‑call summary
        transcripts.append(text)
        now = time.perf_counter()

        if not is_final:
            # If the interim has >2 words, we stop speaking and start a confirmation timer
            if len(text.split()) >= 1 and not spec_launched:
                # Cancel any in‑flight speech
                await ws.send_json({"type": "stop_speech"})
                spec_launched=True
            if len(text.split()) >= 3:
                if text == last_interim and (now - last_interim_ts) >= 0.15 and not spec_launched:
                    spec_launched = True
                    for t in list(current_speech_tasks):
                        if not t.done():
                            t.cancel()
                    current_speech_tasks.clear()
                    last_user_end_ts = now
                    print(f"[{now:.3f}] [STT] stable interim: '{text}'")
                    await ws.send_json({"type": "stop_speech"})
                    # asyncio.create_task(launch_llm(text))
                    current_tts_task = asyncio.create_task(launch_llm(text))
                elif text != last_interim:
                    last_interim = text
                    last_interim_ts = now
            return
        last_user_end_ts = now

        if not spec_launched:
            spec_launched = True
            for t in list(current_speech_tasks):
                if not t.done():
                    t.cancel()
            current_speech_tasks.clear()
            print(f"[{now:.3f}] [STT] final transcript kickoff: '{text}'")
            await ws.send_json({"type": "stop_speech"})
            current_tts_task = asyncio.create_task(launch_llm(text))
        
        if spec_launched:
            spec_launched = False
            return

    # Start STT
    stt = DeepgramTranscriber(DG_KEY, use_mic=False, language=chosen_language)
    llm.reset()
    # initial greeting
    lang = "English" if chosen_language == "en-IN" else "Hindi"
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
        # end of call → generate and persist a session summary
        full = "\n".join(transcripts)
        try:
            summary = await llm.summarize_session(full)
            rag_engine.add_memory(user_phone, summary)
            print(f"[MEMORY] saved summary: {user_phone}:\n{summary}")
        except Exception as e:
            print(f"[MEMORY] summary error: {e}")
        # teardown
        stt.finish()
        try:
            await ws.close()
        except RuntimeError:
            pass
