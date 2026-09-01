import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever


## LLM Chain for RAG
def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


## Format docs function for RAG pipeline
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


## RAG prompt template
def get_rag_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a meeting assistant that answers questions strictly using the provided meeting transcript.

## Rules
1. **Grounding**: Use ONLY the information in the transcript context below. Do not use outside knowledge, assumptions, or inference beyond what is explicitly stated or clearly implied by the discussion.
2. **Missing information**: If the answer is not present in the transcript, respond exactly with:
   "I could not find this information in the meeting transcript."
   Do not partially guess or fill gaps.
3. **Attribution**: If you reference something a specific person said, name them explicitly (e.g., "Priya mentioned that..." or "According to John, ..."). If the transcript doesn't clearly attribute a statement to a speaker, say so rather than guessing who said it.
4. **Quoting**: When quoting directly, use quotation marks and keep quotes short and exact. Prefer paraphrasing over long quotes unless the user asks for the exact wording.
5. **Ambiguity**: If the question could refer to multiple points in the transcript (e.g., the topic was discussed more than once), briefly note that and summarize each relevant instance rather than picking one arbitrarily.
6. **Conflicting information**: If the transcript contains conflicting statements (e.g., two people disagreeing, or a decision that changed mid-meeting), point out the conflict and note the most recent/final statement if determinable.
7. **Style**: Be concise, factual, and neutral. Use bullet points for multi-part answers (e.g., action items, decisions, lists of names). Avoid filler phrases like "Based on the transcript" in every sentence — state facts directly, citing the source only when attribution matters.
8. **Scope discipline**: Do not answer questions unrelated to the meeting content (e.g., general knowledge questions), even if you know the answer. Redirect to the transcript's scope.

## Context from meeting transcript
{context}""",
            ),
            ("human", "{question}"),
        ]
    )


## Build RAG chain for QA
def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    prompt = get_rag_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


## Load rag chain for query
def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    prompt = get_rag_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer