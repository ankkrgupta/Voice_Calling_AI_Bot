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
from mongodb_utils import mongodb_client
import json
import uuid
import httpx

# Environment setup
load_dotenv()
DG_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
DEFAULT_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
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

@app.get("/api/characters")
async def get_characters():
    """Fetch available characters from MongoDB for frontend dropdown"""
    try:
        await mongodb_client.connect_async()
        characters_collection = mongodb_client.async_database.characters
        
        # Fetch all characters with just the fields we need
        cursor = characters_collection.find(
            {}, 
            {"_id": 1, "name": 1, "voiceId": 1, "prompt": 1}
        ).limit(50)  # Limit to 50 characters max
        
        characters = []
        async for char in cursor:
            characters.append({
                "id": str(char["_id"]),
                "name": char.get("name", "Unknown Character"),
                "hasVoiceId": bool(char.get("voiceId")),
                "hasPrompt": bool(char.get("prompt"))
            })
        
        print(f"[API] Found {len(characters)} characters in database")
        return {"characters": characters}
        
    except Exception as e:
        print(f"[API] Error fetching characters: {e}")
        import traceback
        traceback.print_exc()
        return {"characters": [], "error": str(e)}

@app.post("/api/voice-calls/terminate")
async def terminate_voice_call(request_data: dict):
    """Terminate a voice call session due to insufficient credits"""
    try:
        session_id = request_data.get("sessionId")
        reason = request_data.get("reason", "Insufficient credits")
        
        if not session_id:
            return {"error": "Session ID required"}, 400
        
        # In a real implementation, you'd look up the active WebSocket connection
        # For now, we'll just log the termination request
        print(f"[TERMINATION] Request to terminate session {session_id}: {reason}")
        
        # The actual termination happens in the credit_deduction_task when 
        # the webhook returns TERMINATE_CALL action
        
        return {"success": True, "message": f"Termination request processed for session {session_id}"}
        
    except Exception as e:
        print(f"[TERMINATION] Error: {e}")
        return {"error": str(e)}, 500

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

# After environment setup, add webhook URL envs
API_BASE_URL = os.getenv("API_BASE_URL")
SESSION_ENDPOINT = f"{API_BASE_URL}/v1/voice-calls/session"
WEBHOOK_ENDPOINT = f"{API_BASE_URL}/v1/voice-calls/webhook"

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    chosen_language = ws.query_params.get("lang", "en-IN")
    auth_token = ws.query_params.get("token")
    character_id = ws.query_params.get("character")

    if not auth_token:
        # Close with policy violation (1008)
        await ws.close(code=1008, reason="Auth token required")
        return

    # ─── Create unique session id and notify credits API ────────────────
    session_id = str(uuid.uuid4())
    # Track how many credits have been deducted during the lifetime of this session
    total_credits_deducted: int = 0
    user_id: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Pre-payment model: deduct 10 credits upfront for first minute
            resp = await client.post(
                SESSION_ENDPOINT,
                json={
                    "sessionId": session_id,
                    "prePayment": True,
                    "creditsToDeduct": 10
                },
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            print(f"[CREDITS] Pre-payment attempt for session {session_id}")
            
            if resp.status_code == 400:
                # Insufficient credits - cannot start call
                error_data = resp.json()
                print(f"[CREDITS] Insufficient credits: {error_data}")
                await ws.close(code=1008, reason="Insufficient credits to start call")
                return
            
            resp.raise_for_status()
            data = resp.json()
            # Attempt to capture user identifier from response if provided
            session_data = data.get("session") or data.get("data") or {}
            user_id = session_data.get("user") or session_data.get("userId")
            print(f"[CREDITS] Pre-payment successful, 10 credits deducted for first minute")
            # First minute has been prepaid – update running total
            total_credits_deducted = 10
            print(f"[CREDITS] Running total after first minute: {total_credits_deducted}")
            # Inform client immediately about the initial deduction
            try:
                await ws.send_json({
                    "type": "credits_info",
                    "totalCreditDeducted": total_credits_deducted,
                })
                print("[CREDITS] Sent initial credits_info to client")
            except Exception as e:
                print(f"[CREDITS] Failed to send initial credits_info: {e}")
    except Exception as e:
        print(f"[CREDITS] Error during pre-payment: {e}")
        await ws.close(code=1011, reason="Session init failed")
        return

    # ─── Fetch character data from MongoDB or fallback ──────────────────
    voice_id = None
    character_prompt = None
    character_name = "Assistant"
    
    print(f"[MongoDB] Character ID received: '{character_id}'")

    if character_id:
        try:
            print(f"[MongoDB] Attempting to fetch character data for ID: {character_id}")
            voice_id_db, character_prompt_db = await mongodb_client.get_character_voice_and_prompt_async(character_id)
            character_data = await mongodb_client.get_character_by_id_async(character_id)

            print(f"[MongoDB] Raw fetch results - voice_id: {voice_id_db}, prompt: {character_prompt_db is not None}, character_data: {character_data is not None}")

            if character_data:
                print(f"[MongoDB] Character data keys: {list(character_data.keys()) if character_data else 'None'}")

            # Use DB values if present
            if voice_id_db:
                voice_id = voice_id_db
                print(f"[MongoDB] Using voice ID from DB: {voice_id}")
            else:
                print(f"[MongoDB] No voice ID found in DB for character '{character_id}'")
            
            if character_prompt_db:
                character_prompt = character_prompt_db
                print(f"[MongoDB] Using custom prompt from DB (length: {len(character_prompt_db)} chars)")
                print(f"[MongoDB] Prompt preview: {character_prompt_db[:100]}...")
            else:
                print(f"[MongoDB] No custom prompt found in DB for character '{character_id}'")
            
            if character_data is not None and character_data.get("name"):
                character_name = character_data["name"]
                print(f"[MongoDB] Using character name from DB: {character_name}")
            else:
                print(f"[MongoDB] No character name found in DB for character '{character_id}'")

            if not voice_id_db and not character_prompt_db:
                print(f"[MongoDB] WARNING: No voice ID or prompt found for character '{character_id}'")

        except Exception as e:
            print(f"[MongoDB] ERROR fetching character data for '{character_id}': {e}")
            import traceback
            traceback.print_exc()

    # Fallback to defaults if no character ID or DB fetch failed
    if not voice_id:
        voice_id = DEFAULT_VOICE_ID
        print(f"[MongoDB] Falling back to default voice ID: {voice_id}")
    
    if not character_prompt:
        print(f"[MongoDB] No custom prompt found, will use default Zomato prompt")

    # Final sanity check
    if not voice_id:
        print("[Server] CRITICAL: No voice ID available after all fallbacks. Closing connection.")
        await ws.close(code=1011, reason="No voice ID available")
        return

    print(f"[Server] Final configuration - Character: {character_name}, Voice ID: {voice_id}, Custom Prompt: {character_prompt is not None}")

    print(f"[Server] Starting STT with language = {chosen_language}")
    chosen_lang_el = "en" if chosen_language=="en-IN" else "hi"
    user_phone = "6306061252"
    llm = LLMClient(GROQ_KEY, OPENAI_KEY, rag_engine, customer_phone=user_phone, 
                    assistant_name=character_name, character_prompt=character_prompt)
    
    # Initialize TTS client with improved mobile streaming settings
    tts_client = TTSClient(
        api_key=ELEVEN_KEY,
        voice_id=voice_id,
        sample_rate=24000,   # lighter than 44.1 kHz → lower bandwidth
        batch_size_ms=100,   # flush every 100 ms
        chunk_size_ms=100,   # slices sent to client ≈ 100 ms each
    )
    
    # Reset buffer for clean session state
    tts_client.reset_buffer()

    # ─── Background task: ping webhook every minute ─────────────────────
    async def credit_deduction_task(stop_event: asyncio.Event):
        # Access the outer-scope counter so we can update it from here
        nonlocal total_credits_deducted
        async with httpx.AsyncClient(timeout=10.0) as client:
            minute_count = 1  # First minute already paid for during session init
            while not stop_event.is_set():
                # Sleep for 60 seconds (or until stop event)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=60.0)
                except asyncio.TimeoutError:
                    pass  # Expected timeout after 60 seconds
                if stop_event.is_set():
                    break
                
                minute_count += 1
                # Pre-payment for upcoming minute (deduct at START of interval)
                payload = {
                    "callId": session_id,
                    "minute": minute_count,
                    "prePayment": True,
                    "creditsToDeduct": 10
                }
                if user_id:
                    payload["userId"] = user_id
                
                try:
                    resp = await client.post(WEBHOOK_ENDPOINT, json=payload)
                    
                    if resp.status_code == 200:
                        response_data = resp.json()
                        if response_data.get("action") == "TERMINATE_CALL":
                            print(f"[Webhook] Insufficient credits at minute {minute_count} - terminating call")
                            # Signal main websocket loop to terminate
                            stop_event.set()
                            # Inform client about the credits spent so far BEFORE closing
                            try:
                                await ws.send_json({
                                    "type": "credits_info",
                                    "totalCreditDeducted": total_credits_deducted,
                                })
                            except Exception:
                                pass  # Socket might already be closing
                            await ws.close(code=1008, reason="Insufficient credits")
                            break
                        else:
                            print(f"[Webhook] Pre-payment successful for minute {minute_count}")
                            # Add credits for this successfully-paid minute
                            total_credits_deducted += 10
                            print(f"[CREDITS] Running total after minute {minute_count}: {total_credits_deducted}")
                            # Notify client of updated total
                            try:
                                await ws.send_json({
                                    "type": "credits_info",
                                    "totalCreditDeducted": total_credits_deducted,
                                })
                            except Exception:
                                pass
                    else:
                        resp.raise_for_status()
                        print(f"[Webhook] Pre-payment OK for minute {minute_count}")
                        total_credits_deducted += 10
                        print(f"[CREDITS] Running total after minute {minute_count}: {total_credits_deducted}")
                        try:
                            await ws.send_json({
                                "type": "credits_info",
                                "totalCreditDeducted": total_credits_deducted,
                            })
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[Webhook] Error during minute {minute_count} pre-payment: {e}")
                    # Continue call on webhook errors (don't terminate for network issues)

    stop_event = asyncio.Event()
    webhook_task = asyncio.create_task(credit_deduction_task(stop_event))

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
        import re
        # Remove bracketed or asterisk-wrapped stage directions to prevent TTS from reading them aloud
        clean_sentence = re.sub(r"(\[[^\]]*\]|\([^\)]*\)|\*[^\*]*\*)", " ", sentence)
        clean_sentence = re.sub(r"\s+", " ", clean_sentence).strip()

        if clean_sentence:
            await tts_client.speak_sentence(
                sentence=clean_sentence,
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
            # Check if we need to terminate due to insufficient credits
            if stop_event.is_set():
                print(f"[Session] Terminating session {session_id} due to insufficient credits")
                break
                
            try:
                # Use timeout to periodically check stop_event
                msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                if msg.get("bytes") is not None:
                    stt.feed_audio(msg["bytes"])
                elif msg.get("type") == "websocket.disconnect":
                    break
            except asyncio.TimeoutError:
                # Timeout is expected - continue loop to check stop_event
                continue
    finally:
        # end of call → generate and persist a session summary
        # stop webhook task
        stop_event.set()
        webhook_task.cancel()
        try:
            await webhook_task
        except asyncio.CancelledError:
            pass

        full = "\n".join(transcripts)
        try:
            summary = await llm.summarize_session(full)
            rag_engine.add_memory(user_phone, summary)
            print(f"[MEMORY] saved summary: {user_phone}:\n{summary}")
        except Exception as e:
            print(f"[MEMORY] summary error: {e}")
        # teardown
        stt.finish()
        # Always try to send a final credit summary before closing the socket
        try:
            await ws.send_json({
                "type": "credits_info",
                "totalCreditDeducted": total_credits_deducted,
            })
        except Exception:
            pass

        print(f"[CREDITS] Final total credits deducted for session {session_id}: {total_credits_deducted}")
        try:
            await ws.close()
        except RuntimeError:
            pass
