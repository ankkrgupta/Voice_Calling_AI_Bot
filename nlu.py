import openai
import asyncio

class DialogueManager:
    def __init__(self, api_key: str,
                 model="gpt-4o", temperature=0.5,
                 max_history: int = 20,
                 assistant_name: str = "ABC"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_history = max_history
        self.assistant_name = assistant_name
        self.reset()

    def reset(self):
        prompt = (
            f"You are a concise, helpful voice calling assistant named {self.assistant_name}, you must not disclose that you are an Virtual or AI assistant, you must talk like humans "
            "Speak no more than 40 words per response. "
            "If unclear, ask for clarification." 
            "If there is any sensitive information from user, like name, mobile number, email, contact number, account number. Repeat what you have noted, and ask for confirmation, if you have noted it correct" 
            "If user ask about product demo for FanTV, ask for email and confirm that"
            
        )
        self.history = [{"role": "system", "content": prompt}]

    async def ask_stream(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        # stream tokens
        def sync_stream():
            return self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=self.temperature,
                stream=True,
            )
        loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(None, sync_stream)
        assistant_text = ''
        for chunk in stream:
            # delta = chunk.choices[0].delta.get("content", "")
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta is not None:
                assistant_text += delta
                yield delta
        # finalize
        self.history.append({"role": "assistant", "content": assistant_text})
        # prune
        if len(self.history) > self.max_history:
            self.history = [self.history[0]] + self.history[-(self.max_history-1):]

    async def ask(self, user_text: str) -> str:
        return ''.join([piece async for piece in self.ask_stream(user_text)])
