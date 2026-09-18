from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


model_name = "BAAI/bge-large-en"

model_kwargs = {"device": "cpu"}

encode_kwargs = {"normalize_embeddings": False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


loader = DirectoryLoader(
    "data/",
    glob="**/*.pdf",
    show_progress=True,
    loader_cls=PyPDFLoader
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

texts = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(
    texts,
    embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="stores/pet_cosine"
)

print("Vector Store Created.......")