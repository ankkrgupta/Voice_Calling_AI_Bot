import os
import asyncio
import signal
import time
from dotenv import load_dotenv
from stt import DeepgramTranscriber
from nlu import DialogueManager
from tts import ElevenTTS
from audio_io import AudioIO

load_dotenv()
DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "ABC")

for name, val in [
    ("Deepgram key", DEEPGRAM_KEY),
    ("OpenAI key", OPENAI_KEY),
    ("ElevenLabs key", ELEVEN_KEY),
    ("ElevenLabs voice ID", ELEVEN_VOICE_ID),
]:
    if not val:
        raise RuntimeError(f"Missing {name}")

async def main():
    dm = DialogueManager(api_key=OPENAI_KEY, assistant_name=ASSISTANT_NAME)
    tts = ElevenTTS(api_key=ELEVEN_KEY, voice_id=ELEVEN_VOICE_ID)
    audio = AudioIO()
    stt = DeepgramTranscriber(api_key=DEEPGRAM_KEY)

    utterance_queue = asyncio.Queue()

    # Define transcript handler
    # async def on_transcript(text, wav_bytes):
    #     await utterance_queue.put(text)
    async def on_transcript(text, is_final):
        if is_final:  # Only process final transcriptions
            print(f"[STT] Pushing to queue: {text}")
            await utterance_queue.put(text)

    async def worker():
        # Initial greeting
        greet = f"Hello, this is {ASSISTANT_NAME}. How can I help you?"
        # Pause STT so bot doesn't hear its own greeting
        stt.pause()
        data, sr = await tts.synthesize(greet)
        print(f"Bot: {greet}")
        await asyncio.to_thread(audio.stream_play, data, sr)
        # await asyncio.sleep(0.2)
        stt.resume()

        while True:
            user_text = await utterance_queue.get()
            print(f"User: {user_text}")

            assistant_text = ''
            print("Bot: ", end='', flush=True)
            try:
                async for token in dm.ask_stream(user_text):
                    assistant_text += token
                    print(token, end='', flush=True)
                print()

                stt.pause()
                data, sr = await tts.synthesize(assistant_text)
                await asyncio.to_thread(audio.stream_play, data, sr)
                # await asyncio.sleep(0.2)
                stt.resume()
            except Exception as e:
                print(f"Error in worker: {e}")
            finally:
                utterance_queue.task_done()

    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    print("🎤 Starting voice agent...")
    # Launch STT send/receive loop with callback
    stt_task = asyncio.create_task(stt.start(on_transcript))
    worker_task = asyncio.create_task(worker())

    await stop

    # Cancel both tasks on shutdown
    stt_task.cancel()
    worker_task.cancel()
    await asyncio.gather(stt_task, worker_task, return_exceptions=True)
    print("🛑 Shutdown complete.")

if __name__ == '__main__':
    asyncio.run(main())
