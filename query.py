from embeddings import get_embedding_function
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential
import streamlit as st

endpoint = "https://models.inference.ai.azure.com"
model_name = "Llama-3.3-70B-Instruct"

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
    Based solely on the provided context, generate meaningful interview-style questions and corresponding answers to help improve understanding and mastery of the subject.

    Context:
    {context}

    ---
    Question topic specified by the user: {question}

    Guidelines for creating questions:
    - Focus on conceptual clarity, problem-solving, and key differentiators in the topic.
    - Aim for questions that would be relevant in a technical interview.
    - Keep the format as follows:
        Question: [Clear, focused interview-style question]
        Answer: [Concise and accurate answer based only on the given context]
    - Generate exactly {num_questions} questions for each of the given topics.

    Example:
    1-Question: Explain the key difference between dynamic and static testing.
    \nAnswer: Dynamic testing involves executing the software to observe its behavior, while static testing analyzes the code and documentation without running the software.

    Provide the response in a structured format suitable for direct inclusion in a Microsoft Word document.
    """


def query_rag(query_text: str, selected_files, num_questions, ai_selector, EmbModel):
    embedding_function = get_embedding_function(EmbModel)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    search_files_paths = [f"Data\\Documents\\{name}" for name in selected_files]
    # Search the DB
    results = db.similarity_search_with_score(
        query_text, k=20, filter={"source": {"$in": search_files_paths}}
    )

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text, question=query_text, num_questions=num_questions
    )

    if ai_selector == "LLAMA3.2":
        model = OllamaLLM(model="llama3.2", num_thread=3, temperature=0)
        response_text = model.invoke(prompt)
    elif ai_selector == "LLAMA 3.3 API":
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(st.secrets["APIs"]["GITHUB_TOKEN"]),
        )
        response = client.complete(
            messages=[UserMessage(content=prompt)], temperature=0, model=model_name
        )
        # print("response",response.choices[0].message.content)
        response_text = response.choices[0].message.content.replace(
            "Answer", "\nAnswer"
        )

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    return response_text, sources
