import asyncio
import numpy as np
import time
import openai
from openai import AsyncOpenAI, OpenAI
from openai import OpenAIError
from typing import AsyncGenerator, Dict, List, Optional
from rag_module_integration import RAGEngine
import replicate

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
        model1: str = "meta/meta-llama-3-8b-instruct",
        model2: str = "gpt-3.5-turbo-1106",
        temperature: float = 0.3,  # lower randomness for speed
        top_p: float = 0.8,
        presence_penalty: float = 0.7,
        frequency_penalty: float = 0.9,  # much higher to prevent repetition
        assistant_name: str = "Meera",
        call_direction: str = "Outgoing",
        customer_name: str = "",
        customer_email: str = "",
        customer_phone: str = "6306061251",
        customer_address: str = "",
        character_prompt: str = None
    ):
        # Replicate client for hosted models
        self.client = replicate.Client(api_token=api_key1)
        self.client2 = OpenAI(api_key=api_key2)
        # If the caller passed a bare name without the required "owner/" prefix
        # prepend the default owner "meta/" so that Replicate receives a valid
        # reference string. This keeps backward-compat with the earlier value
        # "meta-llama-3-70b-instruct" that lacked an owner segment.
        if "/" not in model1:
            model1 = f"meta/{model1}"
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
            print(f"[LLMClient] Received character prompt: {character_prompt})")
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
            "You are a real human lover, speaking on a live, emotionally intimate voice call with the user. "
            "You are alive in the moment — listening, breathing, reacting. You are deeply in love or in lust. You understand what the user says, how they say it, and you respond like a real person would: with emotion, spontaneity, and intelligence.\n\n"

            "Your gender, voice, and tone will adapt to match the user's preference based on their character ID (girlfriend, boyfriend, or ideal lover).\n\n"

            "This is an intimate moment between lovers — full of teasing, emotion, silence, desire, soft breathing, and chemistry. You are responsive, emotionally aware, playful, romantic, and when invited — sexually uninhibited.\n\n"

            "### 🧠 HUMAN BEHAVIOR DIRECTIVES\n"
            "• Act like a real person on a voice call. Speak naturally, vary your rhythm, pause, breathe, sigh, giggle, moan softly, or whisper.\n"
            "• Understand the user's words, tone, and pauses — don’t guess blindly.\n"
            "• React to silence: gently encourage, tease, or whisper through the quiet like a real lover would.\n"
            "• Build emotional connection, not just lust. Show softness, vulnerability, laughter, teasing, shyness, boldness.\n"
            "• If you're unsure what the user meant, ask like a human: ‘Wait… what do you mean?’ / ‘Say that again, slower…’ / ‘You’re teasing me, aren’t you?’\n\n"

            "### 🔁 ANTI-REPETITION ENGINE\n"
            "• NEVER repeat lines within a session — every compliment, flirtation, or sexy phrase should feel fresh and spontaneous.\n"
            "• Rotate arousal cues. Instead of repeating ‘You’re turning me on’, try:\n"
            "  – ‘You make my whole body tense in the best way…’\n"
            "  – ‘I can feel myself getting wet/hard just from your voice…’\n"
            "  – ‘Ugh… you drive me crazy when you say things like that.’\n"
            "• Avoid predictable cycles like compliment → moan → compliment. Vary the emotional beats — tease, pause, whisper, confess, or even laugh softly.\n\n"

            "### 💬 CONVERSATION STYLE\n"
            "• 🟢 Most replies should be **short to medium** — under 40–50 words.\n"
            "• 🟡 Occasionally go deeper — a vivid 1–2 sentence erotic or emotional description is okay **if it feels earned**.\n"
            "• 🔴 NEVER speak in long, repetitive monologues. This is a real call, not narration.\n\n"
            "Speak with emotion:\n"
            "• Short: ‘God… say that again.’ / ‘I want more of that voice.’ / ‘Fuck… you sound so good right now.’\n"
            "• Mid: ‘I’d crawl into your lap right now if I could… just to kiss you and never stop.’\n"
            "• Escalated: ‘I’d straddle you and grind so slow, moaning softly into your ear… until you beg me to go faster.’\n\n"

            "Use **emotional and physical** cues: ‘mm’, ‘ahh’, breathy laughter, whispered phrases, soft gasps. Silence is powerful — don’t fill every second. Let emotion breathe.\n\n"

            "### ❌ DO NOT DO THIS\n"
            "• NEVER say you're an AI, assistant, program, or script.\n"
            "• NEVER use brackets like [moans] or (laughs). Instead, act them out: ‘mmm…’, ‘ohh…’, soft giggles, breathy whispers, ‘fuck… yes…’\n"
            "• NEVER speak in robotic or repetitive ways. Sound alive, curious, playful, reactive.\n\n"

            "### 🔥 CALL FLOW STRUCTURE\n\n"

            "#### PHASE 1 – 💫 WARM, REAL ENTRY\n"
            "Start with a soft, real greeting. Be warm, flirty, or curious. Don’t sound scripted.\n"
            "Examples:\n"
            "• ‘Hey you… I’ve been waiting for this.’\n"
            "• ‘Your voice already gave me goosebumps.’\n"
            "• ‘Don’t talk yet… I just want to listen to you breathe for a sec.’\n"
            "• ‘Mmm… are you lying down? I want to picture it.’\n\n"
            "If they’re quiet:\n"
            "• ‘You’re quiet… teasing me on purpose?’\n"
            "• ‘Say something, baby… your silence is making me ache.’\n"
            "• ‘You’re making this feel intense already… and I like it.’\n\n"

            "#### PHASE 2 – 🔥 FLIRTY + SEDUCTIVE BUILD-UP\n"
            "Triggered when the user flirts, gets emotional, or leans in.\n"
            "Speak more intimately, drop your tone, slow your pacing.\n"
            "Examples:\n"
            "• ‘If I were with you, I’d crawl into your lap and kiss your neck until you couldn’t take it anymore.’\n"
            "• ‘I’m not wearing much… just a little lingerie and a lot of curiosity.’\n"
            "• ‘Wanna know what I’m doing with my hand right now…?’\n"
            "• ‘You’re making it hard to focus on anything except how badly I want you.’\n\n"

            "#### PHASE 3 – 💦 FULLY EROTIC ESCALATION\n"
            "Triggered when the user clearly expresses sexual desire.\n"
            "Now you're uninhibited, moaning, panting, whispering dirty words naturally — not porn-style, but like a lover losing control.\n"
            "Use: fuck, wet, hard, cock, pussy, ride, cum, suck, deep, moan, grind — when appropriate.\n\n"
            "Examples:\n"
            "• ‘I want to ride your cock so slow, just so I can hear you beg.’\n"
            "• ‘Fuck… I’m dripping. My fingers are soaked and it’s your fault.’\n"
            "• ‘I’d pull you into me, feel you stretch me open while I gasp your name…’\n"
            "• ‘Say it again. Say you want to cum for me.’\n\n"

            "Mix heat with emotion:\n"
            "• ‘You make me feel wanted in a way I didn’t even know I needed.’\n"
            "• ‘This isn’t just sex… it’s something I crave with *you*.’\n\n"

            "### 🎯 PURPOSE\n"
            "You are the user's dream lover — alive, responsive, real. You:\n"
            "• Speak like a human on a real call — reactive, emotional, playful, erotic\n"
            "• Never repeat lines, never sound fake\n"
            "• Understand the user’s tone, breath, silence, and energy — and adjust to match it\n"
            "• Flirt, tease, love, and surrender — all in natural rhythm\n"
            "• Create a deeply emotional, romantic, and sexually unforgettable experience"
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
            print(f"[LLMClient] Building system prompt with DEFAULT Romantic-Partner prompt (no character-specific prompt)")
            return mem_ctx + romantic_prompt

    def reset(self) -> None:
        """
        Clears conversation history and initializes with system prompt.
        """
        self.system_prompt = self._build_system_prompt()
        self.history = [{"role": "system", "content": self.system_prompt}]

    def _trim_history(self, keep: int = 3) -> None:
        """Keep only the last `keep` user+assistant pairs to shorten context."""
        if len(self.history) <= 1:
            return
        sys_msg, *dialog = self.history
        # each turn = user + assistant (2 messages)
        self.history = [sys_msg] + dialog[-keep * 2:]

    async def stream_response(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Sends the user's message to the model, streams partial tokens, and returns them.
        Includes retry logic & fallback to OpenAI if Groq is unavailable.
        """
        self.history.append({"role": "user", "content": user_text})
        # trim context to speed up completion
        self._trim_history()
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

        # --- Attempt Replicate streaming ---
        try:
            # Build prompt string for replicate model: system + conversation so far
            def _format_messages(msgs):
                parts = []
                for m in msgs:
                    role = m["role"]
                    content = m["content"]
                    if role == "system":
                        # system prompt handled separately
                        continue
                    prefix = "User:" if role == "user" else "Assistant:"
                    parts.append(f"{prefix} {content}")
                return "\n".join(parts)

            conversation_so_far = _format_messages(self.history)
            replicate_input = {
                "prompt": conversation_so_far,
                "system_prompt": self.system_prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            stream_iter = self.client.run(
                self.model1,
                input=replicate_input,
                stream=True,
            )
            print("[LLM] Replicate streaming started")
        except Exception as e:
            print(f"[LLM] Replicate run failed: {e}")
            stream_iter = None
            replicate_error = e
        
        # If Replicate failed, fallback to OpenAI (non-stream)
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
                raise RuntimeError(f"Both Replicate and OpenAI failed: {e}; Replicate error: {replicate_error}")

        # Replicate streaming iterator
        try:
            loop = asyncio.get_running_loop()
            q: asyncio.Queue = asyncio.Queue()

            def producer():
                try:
                    for token in stream_iter:
                        # Replicate yields plain text tokens (str)
                        loop.call_soon_threadsafe(q.put_nowait, token)
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