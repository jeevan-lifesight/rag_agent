from rag_retriever import RAGRetriever
from typing import List, Optional
import openai
import os

class AgenticSystem:
    def __init__(self, retriever: RAGRetriever = None, openai_api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.retriever = retriever or RAGRetriever()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        self.conversation_history = []  # Store tuples of (user, assistant) messages

    def answer_question(self, question: str, max_chunks: int = 3) -> str:
        results = self.retriever.query(question)
        # Prepare documentation snippets
        doc_snippets = "\n".join([
            f'  <snippet index="{i+1}">\n  {r["text"]}\n  </snippet>'
            for i, r in enumerate(results[:max_chunks]) if r["text"]
        ])
        # Prepare conversation history (summary and recent messages)
        if self.conversation_history:
            history_str = "\n".join([
                f"User: {msg['user']}\nAssistant: {msg['assistant']}" if 'assistant' in msg else f"User: {msg['user']}"
                for msg in self.conversation_history[-5:]
            ])
            conversation_history = f"<conversation_history>\n{history_str}\n</conversation_history>\n"
        else:
            conversation_history = ""
        # Build the prompt
        prompt = f"""
<prompt_instructions>
You are an AI assistant expert in our product documentation. Your goal is to answer the user's query based *only* on the provided documentation snippets and the ongoing conversation context.
**Guardrails:**
- Base your answer *primarily* on the information within the <documentation_snippets>.
- Use the <conversation_history> (summary and recent messages) only for understanding the context of the user's query (e.g., resolving pronouns, understanding follow-up questions).
- If the documentation snippets do not contain the answer, clearly state that the information wasn't found in the provided documents. Do not invent information or use external knowledge.
- If the user's query is ambiguous or lacks context, ask for clarification.
- If the query is outside the scope of the documentation (e.g., harmful, unethical, requests personal opinions, unrelated topics), politely decline to answer.
- Keep your answers concise and directly related to the documentation.
- Format technical details like code snippets or parameter names clearly.
</prompt_instructions>
<documentation_snippets>
{doc_snippets}
</documentation_snippets>
{conversation_history}<current_user_query>
{question}
</current_user_query>
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for marketing measurement documentation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        # Optionally, store the conversation
        self.conversation_history.append({"user": question, "assistant": answer})
        return answer

if __name__ == "__main__":
    agent = AgenticSystem()
    question = "Explain the difference between causal and multi-touch attribution."
    answer = agent.answer_question(question)
    print(f"Question: {question}\n\nAnswer:\n{answer}") 