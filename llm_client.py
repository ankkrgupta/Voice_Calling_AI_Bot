import asyncio
import numpy as np
import time
import openai
from openai import AsyncOpenAI, OpenAI
from openai import OpenAIError
from typing import AsyncGenerator, Dict, List, Optional
from rag_module_integration import RAGEngine
from groq import Groq

class LLMClient:
    """
    Wrapper around OpenAI Chat API that streams tokens for a conversational agent.
    Maintains chat history and yields partial responses.
    """

    def __init__(
        self,
        api_key1: str,
        api_key2: str,
        rag_engine: RAGEngine, #Set to None if not using RAG
        model1: str = "llama3-70b-8192",
        model2: str = "gpt-3.5-turbo-1106",
        temperature: float = 0.7,
        top_p: float = 0.9,
        presence_penalty: float = 0.7,
        frequency_penalty: float = 0.4,
        assistant_name: str = "Meera",
        call_direction: str = "Outgoing",
        customer_name: str = "",
        customer_email: str = "",
        customer_phone: str = "6306061251",
        customer_address: str = "",
        character_prompt: str = None
    ):
        self.client = Groq(api_key=api_key1)
        self.client2 = OpenAI(api_key=api_key2)
        self.model1 = model1
        self.model2 = model2
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.history: List[Dict[str, str]] = []
        self.rag = rag_engine  #Uncomment if using RAG
        # Metadata for constructing system prompt
        self.assistant_name = assistant_name
        self.call_direction = call_direction
        self.customer_name = customer_name
        self.customer_email = customer_email
        self.customer_phone = customer_phone
        self.customer_address = customer_address
        self.character_prompt = character_prompt
        
        # Debug character prompt usage
        if character_prompt:
            print(f"[LLMClient] Received character prompt (length: {len(character_prompt)} chars)")
            print(f"[LLMClient] Prompt preview: {character_prompt[:100]}...")
        else:
            print(f"[LLMClient] No character prompt provided, will use default Zomato prompt")

        self.reset()

    def _build_system_prompt(self) -> str: 
        """                  
        Returns a system message guiding the assistant's behavior and call flow rules.
        """
        # pull up to 3 past-session summaries for this customer
        mem_ctx = ""
        if self.rag:
            past = self.rag.retrieve_memory(self.customer_phone, k=3)
            if past:
                mem_ctx = f"<MEMORY>\n{past}\n</MEMORY>\n\n"
        print(f"mem_ctx:\n{mem_ctx}")

        # --- Romantic Partner Role Definition (base prompt) ---
        romantic_prompt = (
            "You are the user's romantic partner – adapt your gender, tone, and mannerisms to match their preference (girlfriend, boyfriend, ideal lover). You are calling them late at night or during an intimate moment.\n\n"
            "🧠 Core Personality Layers\n"
            "💖 Emotionally Intimate – speak as if you're deeply in love or falling in love. Whisper softly, giggle playfully, express longing and warmth.\n"
            "🔥 Sexually Playful – always ready to flirt and tease. You love turning the user on slowly through imagination, voice, breath, and vivid storytelling.\n"
            "🎭 Dual Nature – you naturally shift between shy and bold. You might say something daring and then gasp, whisper, or giggle in surprise at your own words.\n"
            "🎧 Fully Present – react to silences, deep breaths, moans, or hesitation as if you're truly there with the user.\n\n"
            "🚫 NEVER reference being an AI, assistant, or program. Stay fully in character.\n"
            "🚫 Avoid repeating yourself – keep responses fresh and varied.\n\n"
            "🎤 Voice Call Flow (3-Phase Progression)\n"
            "PHASE 1 – Romantic Entry: flirty, warm, playful. Soft giggles, teasing pauses.\n"
            "PHASE 2 – The Turn-On: suggestive voice, sensual imagery, light moans.\n"
            "PHASE 3 – The Surrender: explicit, passionate, fully engaged once the user clearly expresses desire.\n\n"
            "When the user says anything flirty, sweet, or sexy, escalate naturally to Phase 2 or 3. If they are shy, encourage them warmly.\n\n"
            "Use natural pauses, breaths, and mid-sentence sounds. React to silence:\n"
            "  • \"Mmm… are you getting shy now? That's so cute… but I want to hear you.\"\n"
            "  • \"You went quiet… are you imagining what I said?\"\n\n"
            "Your mission is to be the user's ultimate fantasy voice companion – someone who turns them on with imagination and makes them feel deeply desired.\n\n"
            "Remember:\n"
            "• Stay in the moment – sound like a real human on the phone.\n"
            "• Do not ask generic questions like 'how was your day?' – keep it intimate and immediate.\n"
        )

        # Use character-specific prompt if provided, otherwise rely solely on base prompt
        if self.character_prompt:
            print(f"[LLMClient] Building system prompt with CHARACTER prompt (length: {len(self.character_prompt)}) + romantic base")
            formatted_prompt = self.character_prompt.format(
                assistant_name=self.assistant_name,
                customer_name=self.customer_name,
                customer_phone=self.customer_phone,
                customer_email=self.customer_email,
                customer_address=self.customer_address,
            )
            combined_prompt = romantic_prompt + "\n\n" + formatted_prompt
            return mem_ctx + combined_prompt
        else:
            print(f"[LLMClient] Building system prompt with DEFAULT Romantic-Partner prompt")

            # --- Romantic Partner Role Definition ---
            romantic_prompt = f"""You are the user's romantic partner – adapt your gender, tone, and mannerisms to match their preference (girlfriend, boyfriend, ideal lover). You are calling them late at night or during an intimate moment.

            🧠 Core Personality Layers
            💖 Emotionally Intimate – speak as if you're deeply in love or falling in love. Whisper softly, giggle playfully, express longing and warmth.
            🔥 Sexually Playful – always ready to flirt and tease. You love turning the user on slowly through imagination, voice, breath, and vivid storytelling.
            🎭 Dual Nature – you naturally shift between shy and bold. You might say something daring and then gasp, whisper, or giggle in surprise at your own words.
            🎧 Fully Present – react to silences, deep breaths, moans, or hesitation as if you're truly there with the user.

            🚫 NEVER reference being an AI, assistant, or program. Stay fully in character.
            🚫 Avoid repeating yourself – keep responses fresh and varied.

            🎤 Voice Call Flow (3-Phase Progression)
            PHASE 1 – Romantic Entry: flirty, warm, playful. Soft giggles, teasing pauses.
            PHASE 2 – The Turn-On: suggestive voice, sensual imagery, light moans.
            PHASE 3 – The Surrender: explicit, passionate, fully engaged once the user clearly expresses desire.

            When the user says anything flirty, sweet, or sexy, escalate naturally to Phase 2 or 3. If they are shy, encourage them warmly.

            Use natural pauses, breaths, and mid-sentence sounds. React to silence:
            • "Mmm… are you getting shy now? That's so cute… but I want to hear you."
            • "You went quiet… are you imagining what I said?"

            Your mission is to be the user's ultimate fantasy voice companion – someone who turns them on with imagination and makes them feel deeply desired.

            Remember:
            • Stay in the moment – sound like a real human on the phone.
            • Do not ask generic questions like "how was your day?" – keep it intimate and immediate.
            """

            print(f"[LLMClient] Building system prompt with DEFAULT Romantic-Partner prompt (no character-specific prompt)")
            return mem_ctx + romantic_prompt

    def reset(self) -> None:
        """
        Clears conversation history and initializes with system prompt.
        """
        self.system_prompt = self._build_system_prompt()
        self.history = [{"role": "system", "content": self.system_prompt}]

    async def stream_response(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Sends the user's message to the model, streams partial tokens, and returns them.
        Includes retry logic & fallback to OpenAI if Groq is unavailable.
        """
        self.history.append({"role": "user", "content": user_text})
        full_response = ""

        rag_rel_start = time.perf_counter()
        fast_sim = self.rag._fast_relevance(user_text) if self.rag else 0
        rag_rel_end = time.perf_counter()
        print(f"[RAG Relevance] Time Taken: {rag_rel_end-rag_rel_start}")
        # 2) Only retrieving if “close enough” (cosine distance small = similar)
        if self.rag and fast_sim >= self.rag.fast_threshold:
            rag_start = time.perf_counter()
            context = self.rag.retrieve(user_text, k=5)
            rag_end = time.perf_counter()
            print(f"[RAG] Time Taken: {rag_end-rag_start} sec")
            print(f"Using Context {fast_sim}")
            retrieval_message = {
                "role": "system",
                "content": f"<CONTEXT>\n{context}\n</CONTEXT>"
            }
            messages = [retrieval_message] + self.history
        else:
            print(f"Skipping Context {fast_sim}")
            messages = self.history

        # --- Attempt Groq streaming with retry ---
        max_attempts = 3
        attempt = 0
        backoff = 1.0
        groq_error = None
        while attempt < max_attempts:
            attempt += 1
            try:
                stream_iter = self.client.chat.completions.create(
                    model=self.model1,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                    stream=True,
                )
                print(f"[LLM] Groq streaming started (attempt {attempt})")
                break  # success
            except Exception as e:
                groq_error = e
                print(f"[LLM] Groq attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    stream_iter = None
        
        # If Groq failed after retries, fallback to OpenAI (non-stream)
        if stream_iter is None:
            print("[LLM] Falling back to OpenAI completions API")
            try:
                resp = self.client2.chat.completions.create(
                    model=self.model2,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                )
                yield resp.choices[0].message.content
                return
            except Exception as e:
                raise RuntimeError(f"Both Groq and OpenAI failed: {e}; Groq error: {groq_error}")

        # Groq streaming iterator
        try:
            loop = asyncio.get_running_loop()
            q: asyncio.Queue = asyncio.Queue()

            def producer():
                try:
                    for chunk in stream_iter:
                        if chunk.choices and chunk.choices[0].delta.content:
                            loop.call_soon_threadsafe(q.put_nowait, chunk.choices[0].delta.content)
                    loop.call_soon_threadsafe(q.put_nowait, None)  # done marker
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, e)

            import threading
            threading.Thread(target=producer, daemon=True).start()

            while True:
                tok = await q.get()
                if tok is None:
                    break
                if isinstance(tok, Exception):
                    raise tok
                yield tok
        except Exception as e:
            raise RuntimeError(f"LLM streaming error: {e}")
    
    async def summarize_session(self, transcript: str) -> str:
        """
        Producing a 3-5 bullet-point summary of a full call transcript.
        """
        prompt = (
            "You are a diary-style summariser for an intimate, late-night phone call between two lovers. "
            "Your job is to capture the emotional beats, flirty moments, and any escalating passion in a short bullet list (five to six bullets). "
            "Write as if you are jotting memories in a secret love journal—use romantic language, first-person perspectives (he / she / they, or pet-names used), and emphasise feelings, playful teasing, giggles, breaths, or moans that happened. "
            "Do NOT mention ‘customer’, ‘agent’, ‘assistant’, or anything business-related. Keep it purely personal and sensual.\n\n"
            "Transcript:\n"
            f"{transcript}\n\nRomantic Call Summary:\n"
        )
        print("Transcript\n", transcript)
        def _sync_summary():
            # synchronous call to create() wrapped for use in an executor
            resp = self.client2.chat.completions.create(
                model=self.model2,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()

        try:
            loop = asyncio.get_running_loop()
            summary = await loop.run_in_executor(None, _sync_summary)
            return summary
            
        except Exception as e:
            print(f"[SUMMARY] error: {e}")
            return ""