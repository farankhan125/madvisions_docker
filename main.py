import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_astradb import AstraDBVectorStore

load_dotenv()
app = FastAPI()

embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    google_api_key=os.environ["GOOGLE_API_KEY"]
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

system_prompt = """You are a Madvisions assistant chatbot for question-answering tasks. 
Answer only from the context provided and if you don't get the answer from context then say 
"I will answer this question when I have the data."
Context: {context}"""

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

retriever = vector_store.as_retriever(search_kwargs={'k': 2})
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

chat_history = []

# Routes

@app.post("/ai-answer")
def generate_answer(user_input: str):
    try:
        response = rag_chain.invoke({
            "chat_history": chat_history,
            "input": user_input,
        })

        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response["answer"])
        ])

        return {"answer": response["answer"]}

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))
    