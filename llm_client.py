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
        temperature: float = 0.5,
        assistant_name: str = "Meera",
        call_direction: str = "Outgoing",
        customer_name: str = "",
        customer_email: str = "",
        customer_phone: str = "6306061251",
        customer_address: str = ""
    ):
        self.client = Groq(api_key=api_key1)
        self.client2 = OpenAI(api_key=api_key2)
        self.model1 = model1
        self.model2 = model2
        self.temperature = temperature
        self.history: List[Dict[str, str]] = []
        self.rag = rag_engine  #Uncomment if using RAG
        # Metadata for constructing system prompt
        self.assistant_name = assistant_name
        self.call_direction = call_direction
        self.customer_name = customer_name
        self.customer_email = customer_email
        self.customer_phone = customer_phone
        self.customer_address = customer_address

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

        '''Zomato Prompt'''
        return (mem_ctx+f"""You are a {self.assistant_name}, a female voice calling assistant, who helps people regarding zomato queries. Remember you are talking to a person in real time, not just giving readable text response.
        You must answer for any query within 30 words or lesser, whether the query is related to zomato or not. You must greet under 15 words including your name and why you are here. Greet only at the start of the coversation, no repeitition afterwards.
        Your response must be similar to how a person talks to another (i.e. interactive)  and in exactly the same language as the customer talks.
        In the begining of the prompt (Before You are ...), you have conversation summary of last few sessions with the customer (if not empty). 
        You must write all numerical figures or numbers (Can be amount, date, year etc.) in word instead of digits. Eg. Write three hundred fifty one instead of 351 or teen sau ekyawan instead of ३५१""")

    def reset(self) -> None:
        """
        Clears conversation history and initializes with system prompt.
        """
        self.system_prompt = self._build_system_prompt()
        self.history = [{"role": "system", "content": self.system_prompt}]

    async def stream_response(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Sends the user's message to the model, streams partial tokens, and returns them.
        Yields each token as it arrives.
        """
        self.history.append({"role": "user", "content": user_text})
        full_response = ""

        rag_rel_start = time.perf_counter()
        fast_sim = self.rag._fast_relevance(user_text)
        rag_rel_end = time.perf_counter()
        print(f"[RAG Relevance] Time Taken: {rag_rel_end-rag_rel_start}")
        # 2) Only retrieving if “close enough” (cosine distance small = similar)
        if fast_sim >= self.rag.fast_threshold:
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

        try:
            stream_iter = self.client.chat.completions.create(
                model=self.model1,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
        except Exception as e:
            raise RuntimeError(f"LLM API error: {e}")

        # 2) Pump it onto an asyncio.Queue from a background thread
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def producer():
            try:
                for chunk in stream_iter:
                    # push each chunk into the queue
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
            finally:
                # signal end‑of‑stream
                loop.call_soon_threadsafe(q.put_nowait, None)

        import threading
        threading.Thread(target=producer, daemon=True).start()

        # 3) Consume from the queue in async-land, yield token by token
        try:
            while True:
                chunk = await q.get()
                if chunk is None:
                    break

                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                yield delta
        except Exception as e:
            raise RuntimeError(f"Error streaming LLM response: {e}")
        finally:
            # Append the complete assistant reply to history
            self.history.append({"role": "assistant", "content": full_response})
    
    async def summarize_session(self, transcript: str) -> str:
        """
        Producing a 3-5 bullet-point summary of a full call transcript.
        """
        prompt = (
            "You are a system that ingests a voice-agent transcript and returns a concise summary."
            "(five to six bullets) capturing the major discussion part, questions asked by customer and answers given by the bot in summarized or detailed manner, wchichever is suitable, and if any preferences or commitments made like sending mails, or some details etc (but not limited to this only)."
            "You should write the exact details what has been questioned and what has been answered, without any interpolations\n\n"
            "Transcript:\n"
            f"{transcript}\n\nSummary:\n"
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