
import os
import json

from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA


app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


local_llm = "neural-chat-7b-v3-1.Q4_K_M.gguf"

config = {
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": max(1, int(os.cpu_count() / 2))
}


llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="avx2",
    **config
)

print("LLM Initialized....")


prompt_template = """
Use the following pieces of information to answer the user's question.

If you don't know the answer, just say that you don't know,
don't try to make up an answer.

Context:

{context}

Question:

{question}

Only return the helpful answer below and nothing else.

Helpful answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

general_prompt = """
Answer the user's question directly and helpfully.

If the user asks for code, provide the complete code with a brief explanation.

Question:
{question}

Helpful answer:
"""

general_prompt = PromptTemplate(
    template=general_prompt,
    input_variables=["question"]
)


model_name = "BAAI/bge-large-en"

model_kwargs = {
    "device": "cpu"
}

encode_kwargs = {
    "normalize_embeddings": False
}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding Model Initialized....")


load_vector_store = Chroma(
    persist_directory="stores/pet_cosine",
    embedding_function=embeddings
)

print("Vector Store Loaded....")


retriever = load_vector_store.as_retriever(
    search_kwargs={
        "k": 1
    }
)

print("Retriever Initialized....")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html"
)


@app.post("/get_response")
async def get_response(query: str = Form(...)):

    results = load_vector_store.similarity_search_with_score(
        query,
        k=1
    )

    print("QUERY:", query)
    print("SCORE:", results[0][1] if results else "NO RESULT")

    threshold = 0.15

    if results and results[0][1] <= threshold:

        chain_type_kwargs = {
            "prompt": prompt
        }

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
            verbose=True
        )

        response = qa.invoke(
            {
                "query": query
            }
        )

        answer = response["result"]

        source_documents = response.get(
            "source_documents",
            []
        )

        if source_documents:
            source_document = source_documents[0].page_content
            doc = source_documents[0].metadata.get(
                "source",
                "Unknown"
            )
        else:
            source_document = ""
            doc = "Unknown"

    else:

        response = llm.invoke(
            general_prompt.format(
                question=query
            )
        )

        answer = response
        source_document = ""
        doc = "General Knowledge"

    response_data = jsonable_encoder(
        json.dumps(
            {
                "answer": answer,
                "source_document": source_document,
                "doc": doc
            }
        )
    )

    return Response(
        content=response_data,
        media_type="application/json"
    )

