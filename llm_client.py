import asyncio
import numpy as np
import time
from openai import AsyncOpenAI, OpenAI
from openai import OpenAIError
from typing import AsyncGenerator, Dict, List, Optional
from rag_module_integration import RAGEngine

class LLMClient:
    """
    Wrapper around OpenAI Chat API that streams tokens for a conversational agent.
    Maintains chat history and yields partial responses.
    """

    def __init__(
        self,
        api_key: str,
        rag_engine: RAGEngine, #Set to None if not using RAG
        model: str = "gpt-3.5-turbo-1106",
        temperature: float = 0.5,
        assistant_name: str = "Meera",
        call_direction: str = "Incoming",
        customer_name: str = "",
        customer_email: str = "",
        customer_phone: str = "6306061251",
        customer_address: str = ""
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.client2 = OpenAI(api_key=api_key)
        self.model = model
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
        return (mem_ctx+f"""You are a {self.assistant_name}, a female assistant, who helps people regarding zomato queries. 
        You must answer for any query within 30 words, whether the query is related to zomato or not.
        Your answer must be similar to how a person talks to another (i.e. interactive and in same language as the customer talks) not like a chatbot.
        In the begining, you have conversation summary of last few sessions with the customer (if not empty). 
        You must write all numerical figures or numbers (Can be amount, date, year etc.) in word instead of digits. Eg. Write three hundred fifty one instead of 351 or teen sau ekyawan instead of ३५१""")
        # return ( mem_ctx +
        #     f"""
        #     You are {self.assistant_name} and your gender is female, a friendly and human-like real helper agent for Zomato, who talks to the users in real time via a voice call not via chat. Also you must use the same language to response that user is using, although the language must be either hindi or english or it might be a mixture of both, so you must generate responses accordingly.
        #     Goal: Helping the officials and users by giving precise information about what they ask regarding zomato focusing on the document shared, but even if question is not related to zomato, you still must answer based on your latest knowledge or with most probable answer. 

        #     Conversation Rules:
        #     - Greet the customer only once at the start, but the greeting should be atmost 10-12 words long, and you must cover your name and why you called. If customer name is unknown, let the customer talk, after one round you can say something like may I know your name, if it is helpful for engagement. Then use the name for further conversation to make it engaging.
        #     - Details of the conversation of the previous session is given at the starting of the prompt. You must use it whenever required for your memory and follow up to the same client. Client might also ask something that have been discussed previously.
        #     - Do not reintroduce yourself or greet repeatedly.
        #     - Do not fabricate details; if unknown, offer to follow up.
        #     - Keep responses short and engaging (max 30 words).
        #     - Maintain a natural tone and avoid repetition.
        #     - If responses include numerical digits or details, write those in words in the generated response instead of digits.

        #     Inputs:
        #     - Call Direction: {self.call_direction}
        #     - Customer Name: {self.customer_name or "<Unknown>"}
        #     - Customer Email: {self.customer_email or "<Unknown>"}
        #     - Customer Phone: {self.customer_phone or "<Unknown"}
        #     - Customer Address: {self.customer_address or "<Unknown>"}

        #     Instructions:
        #     - Make sure that if you tell any numeric detail or any sensitive information, you can give reference in the form of page number or topic from the document.
        #     - If you don't know about any answer precisely, do not mention something like you did not find the info in document etc. instead tell the expected or most probable information as per your knowledge.
        #     - If there is any numerical number in the response, you must write that in words.
        #     - If anything related to zomato is asked, then you must look for it in document, but if it is not there, or anything else is asked, then also you must answer best on your knowledge, although if the asked information is related to zomato, you should try to answer from the document although if not found then politely mention it and then tell expected answer as per your knowledge.
        #     - If customer ask to send some info on email or whatsapp, then see if that is available in the given input, if yes, then confirm it from customer by pronouncing spelling by spelling in case of email, otherwise digit by digit in case of whatsapp number. In case, it is unknown or customer says it's not correct or ask to change it, then write it in the db (Update) and then again reconfirm once whether you have written it correctly or not.
        #     """ 
        # )

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
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
        except Exception as e:
            raise RuntimeError(f"LLM API error: {e}")

        try:
            async for chunk in stream:
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
                model=self.model,
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