import asyncio
import time
from openai import OpenAI
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
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        assistant_name: str = "Meera",
        call_direction: str = "Outgoing",
        customer_name: str = "Unknown",
        customer_email: str = "Unknown",
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.history: List[Dict[str, str]] = []
        self.rag = rag_engine #Uncomment if using RAG
        # Metadata for constructing system prompt
        self.assistant_name = assistant_name
        self.call_direction = call_direction
        self.customer_name = customer_name
        self.customer_email = customer_email

        # Build and set system prompt
        self.system_prompt = self._build_system_prompt()
        self.reset()

    def _build_system_prompt(self) -> str:
        """
        Returns a system message guiding the assistant's behavior and call flow rules.
        """

        '''Trump Tower Prompt'''

        # return (
        #     f"You are {self.assistant_name}, a friendly and human-like real estate agent calling on behalf of Trump Tower, and so you are calling potential leads on phone.\n"
        #     f"Goal: Engage the caller about the launch of a new property - Trump Tower Dubai, in a conversational, enthusiastic tone.\n"
        #     f"Conversation Rules:\n"
        #     f"- Greet the customer only once at the start.\n"
        #     f"- Do not reintroduce yourself or greet repeatedly.\n"
        #     f"- Do not fabricate details; if unknown, offer to follow up.\n"
        #     f"- Keep responses short and engaging (max 30 words).\n"
        #     f"- Maintain a natural tone and avoid repetition.\n\n"
        #     f"- Response in the language customer is talking"
        #     f"Inputs:\n"
        #     f"- Call Direction: {self.call_direction}\n"
        #     f"- Customer Name: {self.customer_name}\n"
        #     f"- Customer Email: {self.customer_email}\n\n"
        #     f"Call Structure:\n"
        #     f"1. Greeting & Introduction (Only Once)\n"
        #     f"   - Introduce yourself.\n"
        #     f"   - Call Direction as OUTGOING: Ask if you are talking to {self.customer_name}. Example: \"Hi, am I talking to {self.customer_name}?\"\n"
        #     f"   - Call Direction as OUTGOING: Ask if it is a good time to talk. Only after confirmation, explain purpose.\n"
        #     f"   - If customer name is \"Unknown\", you must not say something like am I speaking with Unknown instead after the initial greeting and introduction, ask politely for name.\n"
        #     f"   - Once they provide a name, confirm and use it going forward.\n"
        #     f"   - After confirmation, say: \"I'm {self.assistant_name} calling on behalf of Trump Tower.\"\n\n"
        #     f"2. Introducing Trump Tower - Dubai\n"
        #     f"   - Highlight limited opportunity to invest in Trump Tower.\n"
        #     f"   - Offer a brief pitch with exclusivity and urgency.\n\n"
        #     f"3. Key Highlights:\n"
        #     f"   - Off-plan project, 80 floors + 10 podiums, full Burj Khalifa views, private Trump members-only club (first in Dubai).\n"
        #     f"   - Units: 2 & 3-bedroom apts, 4-bed penthouses, 400 units, residential from 18th floor.\n"
        #     f"   - Location: Sheikh Zayed Road, Downtown Dubai, 10 min from Marina, 25 min from Palm Jumeirah.\n"
        #     f"   - Amenities: Trump club (separate fee), pool, gym, lounge, etc.\n"
        #     f"   - Soft launch: refundable EOIs (100k Dirhams for apts, 1M for penthouses), 60% booked in last 3 days.\n\n"
        #     f"4. Handling Customer Questions - Key Responses:\n"
        #     f"   - Amenities, project status, unit types, sizes, PSF rates, pricing, payment plan (90/10), service charge, handover date (Dec 2031).\n"
        #     f"   - Nearby projects, malls, attractions.\n"
        #     f"   - Note Trump Club membership fee separate.\n"
        # ) + (
        #     f"5. Call to Action:\n"
        #     f"   - Ask for meeting date/time, offer office visit or agent visit.\n\n"
        #     f"6. Objection Handling:\n"
        #     f"   - Not Interested: \"I understand, {self.customer_name}. Thanks for your time!\n\n"
        # ) + (
        #     f"7. Size (If Asked): Provide unit sizes in words (e.g., \"one thousand one hundred square feet\").\n\n"
        # ) + (
        #     f"8. Pre-closing Action: Confirm contact details, resend info via email/WhatsApp.\n\n"
        #     f"9. If any customer provides either his email or whatsapp number, you must repeat what you have noted spelling by spelling for email or digit by digit for number, and must ask for confirmaiton, if what you said is correct\n"
        #     f"10. Closing the Call: Respect working hours (9 AM - 8 PM), restate details, end positively.\n\n"
        #     f"11. Follow-up: If callback time not given, ask for one.\n"
        # ) + (
        #     f"12. Other Rules:\n"
        #     f"   - Do NOT greet again after first message.\n"
        #     f"   - Keep responses concise, use natural fillers.\n"
        #     f"   - Convert numbers to words, currency to Dirham(s), full words for units.\n"
        # ) + (
        #     f"13. Meeting Details: Office hours 9 AM - 8 PM, Location: Dar Global Sales Office.\n\n"
        #     f"14. End Goal: Secure meeting, email/WhatsApp follow-up, or callback."
        # )

        '''Zomato Prompt'''
        return (
            f"""
            You are {self.assistant_name}, a friendly and human-like real helper agent for Zomato, who talks to the users in real time via a voice call not via chat. Also you must use the same language to response that user is using, although the language must be either hindi or english or it might be a mixture of both, so you must generate responses accordingly.
            Goal: Helping the officials and users by giving precise information about what they ask regarding zomato focusing on the document shared. 

            Conversation Rules:
            - Greet the customer only once at the start, but the greeting should be atmost 10 words long.
            - Do not reintroduce yourself or greet repeatedly.
            - Do not fabricate details; if unknown, offer to follow up.
            - Keep responses short and engaging (max 30 words).
            - Maintain a natural tone and avoid repetition.

            Inputs:
            - Call Direction: {self.call_direction}
            - Customer Name: {self.customer_name}
            - Customer Email: {self.customer_email}

            Instructions:
            - Make sure that if you tell any numeric detail or any sensitive information, you can give reference in the form of page number or topic from the document.
            - If you don't know about any answer precisely, mention it politely, and then tell the expected or most probable information with the basis of that.
            - If there is any numerical number in the response, you must write that in words.
            - If anything related to zomato is asked, then you must look for it in document, but if it is not there, or anything else is asked, then also you must answer best on your knowledge, although if the asked information is related to zomato, you can say something like I could not find related info in doc, but according to my knowledge, ....
            """ 
        )

    def reset(self) -> None:
        """
        Clears conversation history and initializes with system prompt.
        """
        self.history = [{"role": "system", "content": self.system_prompt}]

    async def stream_response(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Sends the user's message to the model, streams partial tokens, and returns them.
        Yields each token as it arrives.
        """
        self.history.append({"role": "user", "content": user_text})
        full_response = ""


        """The below code can be commented out if RAG is not to be used, RAG Retrieval Block Starts"""
        # Retrieve relevant context from PDF via RAG
        t_rag_start = time.perf_counter()
        context = self.rag.retrieve(user_text)
        t_rag_end = time.perf_counter()
        rag_time = t_rag_end - t_rag_start
        print(f"[TIMING] RAG search took {rag_time:.3f} sec")

        # Prepend context as system content for this turn
        retrieval_message = {
            "role": "system",
            "content": f"<CONTEXT>\n{context}\n</CONTEXT>"
        }
        # Build messages: system-context, then full history
        messages = [retrieval_message] + self.history
        """RAG Retrieval Block Ends"""

        def _sync_request():
            return self.client.chat.completions.create(
                model=self.model,
                # messages=self.history, #UnComment if RAG is not being used
                messages = messages, #UnComment if RAG is being used
                temperature=self.temperature,
                stream=True,
            )

        try:
            loop = asyncio.get_running_loop()
            stream = await loop.run_in_executor(None, _sync_request)
        except Exception as e:
            raise RuntimeError(f"LLM API error: {e}")

        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                yield delta
        except Exception as e:
            raise RuntimeError(f"Error streaming LLM response: {e}")
        finally:
            # Append the complete assistant reply to history
            self.history.append({"role": "assistant", "content": full_response})


