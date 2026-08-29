from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os 

## GETTING THE LLM INSTANCE 
def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.3)


## SPLITTING THE TRANSCRIPT INTO CHUNKS FOR PROCESSING 
def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

## SUMMARIZING THE TRANSCRIPT
def summarize(transcript : str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting summarizer. Read the following portion of a meeting "
            "transcript and write a concise, highly detailed summary capture all critical points, "
            "key contexts, and specific facts discussed. Avoid generic statements."
        ),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert executive analyst. You are provided with several sequential partial "
            "summaries of a meeting transcript. Synthesize these into a single, cohesive, "
            "and professional meeting summary.\n\n"
            "Format the summary cleanly using bullet points, grouping related points under clear, logical "
            "subheadings if applicable. Ensure the output is readable, professional, and retains crucial "
            "details, metrics, or contexts from the partial summaries.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)


## GENERATING A TITLE FOR THE TRANSCRIPT 
def generate_title(transcipt : str) -> str:
    llm = get_llm()

    

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "You are an expert editor. Carefully analyze the beginning of the meeting transcript "
                "and generate a short, professional, and engaging title (maximum 8 words) that accurately "
                "captures the central theme or topic of the meeting. Return only the title text, with no "
                "introductory text, quotes, or markdown formatting.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return title_chain.invoke(transcipt[:2000])



