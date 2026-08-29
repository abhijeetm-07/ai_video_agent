from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 

## GETTING THE LLM INSTANCE
def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)


## BUILDING THE CHAIN FOR EXTRACTING KEY INFORMATION FROM THE TRANSCRIPT
def build_chain(system_prompt : str):
  llm = get_llm()
  return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )


## EXTRACTING ACTION ITEMS FROM THE TRANSCRIPT 
def extract_action_items(transcript:str)->str:
    chain = build_chain(
        "You are an expert meeting analyst. Carefully read the meeting transcript "
        "and extract all action items, tasks, and follow-ups. For each action item, "
        "provide the following details:\n"
        "- Task Description: What needs to be done (be specific and actionable).\n"
        "- Owner: Who is responsible for executing this task (specify name or role, or write 'Unassigned' if not mentioned).\n"
        "- Deadline: When the task needs to be completed (specify the exact date/time or relative timeline, or write 'Not specified' if not mentioned).\n\n"
        "Format the output as a clean, professionally styled numbered list. "
        "If no action items are discussed in the transcript, output exactly: 'No action items found.'"
    )

    return chain.invoke(transcript)


## EXTRACTING KEY DECISIONS FROM THE TRANSCRIPT 
def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. Carefully read the meeting transcript "
        "and extract all key decisions, agreements, policies, or conclusions that "
        "were finalized during the meeting. For each decision, provide context or "
        "the rationale behind it if discussed.\n\n"
        "Format the output as a clean, professionally styled numbered list. "
        "If no key decisions are made in the transcript, output exactly: 'No key decisions found.'"
    )
    return chain.invoke(transcript)


## EXTRACTING UNRESOLVED QUESTIONS FROM THE TRANSCRIPT 
def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. Carefully read the meeting transcript "
        "and extract all unresolved questions, open issues, or topics requiring "
        "further discussion or follow-up. For each item, indicate who raised the question "
        "or who is expected to follow up, if mentioned.\n\n"
        "Format the output as a clean, professionally styled numbered list. "
        "If no open questions or unresolved topics are found, output exactly: 'No open questions found.'"
    )
    return chain.invoke(transcript)