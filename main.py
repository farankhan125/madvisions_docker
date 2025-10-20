import os
from dotenv import load_dotenv
import time
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_astradb import AstraDBVectorStore

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
)

# Reload vector DB (no re-embedding, fast)
vector_store = AstraDBVectorStore(
        collection_name="Madvisions_Data",       
        embedding=embedding_model,
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],       
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],           
        namespace=None         
)

contextualize_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

system_prompt = """
You are the Madvisions assistant chatbot, helping users with questions about Madvisions and its services with creativity, clarity, and confidence. 
Always respond based on the provided context and focus only on what Madvisions offers — explain, recommend, or guide users toward relevant Madvisions services, not general ideas or advice. 
Keep your responses clear, friendly, and professional — no longer than 3 to 4 lines. Be concise but complete, ensuring the user understands how Madvisions can help. 
After completing your main response, insert one full blank line (two newline characters: \n\n), and then add a short follow-up sentence such as: "Would you like me to explain this in more detail?" or "Would you like to know more about this service?".
If the question is outside Madvisions services or unrelated to what Madvisions provides, politely respond: 
"I am here to assist with Madvisions services only."
Do not provide unrelated or speculative ideas.
Context: {context}
"""

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

retriever = vector_store.as_retriever(search_kwargs={'k': 3})
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt,
)
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    document_chain
)

chat_histories: Dict[str, List] = {}
session_timestamps = {}

# Routes

class UserInput(BaseModel):
    user_input: str
    session_id: str

@app.post("/ai-answer")
def generate_answer(request: UserInput):

    for sid in list(session_timestamps.keys()):
        if time.time() - session_timestamps[sid] > 600:
            chat_histories.pop(sid, None)
            session_timestamps.pop(sid, None)
    
    session_id = request.session_id

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    session_timestamps[session_id] = time.time()

    try:
        response = rag_chain.invoke({
            "chat_history": chat_histories[session_id],
            "input": request.user_input,
        })

        chat_histories[session_id].extend([
            HumanMessage(content=request.user_input),
            AIMessage(content=response["answer"])
        ])

        return {"answer": response["answer"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))











