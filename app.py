import streamlit as st
import os
import qdrant_client
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
)
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.core.schema import ImageDocument
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from PIL import Image
import tempfile

st.set_page_config(page_title="Multimodal RAG with Gemini & Qdrant", layout="centered")
st.title("Multimodal RAG Query Interface 🖼️📄")
st.caption("Query your data using text or images with Gemini Pro Vision and Qdrant.")

QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
QDRANT_COLLECTION_NAME = "sujet-finance-multi-vdr"
EMBED_MODEL_ID = "llamaindex/vdr-2b-multi-v1"

if not QDRANT_URL or not QDRANT_API_KEY:
    st.error(
        "🚨 QDRANT_URL and QDRANT_API_KEY must be configured via Streamlit secrets or environment variables."
    )
    st.stop()
if not GOOGLE_API_KEY:
    st.error(
        "🚨 GOOGLE_API_KEY must be configured via Streamlit secrets or environment variables."
    )
    st.stop()


@st.cache_resource(show_spinner="Connecting to Qdrant and loading models...")
def load_llama_index_components():
    try:
        client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        vector_store = QdrantVectorStore(
            client=client, collection_name=QDRANT_COLLECTION_NAME
        )

        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID, trust_remote_code=True)
        Settings.embed_model = embed_model

        gemini_llm = GeminiMultiModal(
            api_key=GOOGLE_API_KEY,
            model_name="models/gemini-2.0-flash",
        )
        Settings.llm = gemini_llm

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        query_engine = SimpleMultiModalQueryEngine(
            retriever=index.as_retriever(similarity_top_k=3),
            multi_modal_llm=gemini_llm,
        )
        return (
            query_engine,
            client,
        )
    except Exception as e:
        st.error(f"🚨 Failed to initialize LlamaIndex components: {e}")
        st.stop()


query_engine, qdrant_client_instance = load_llama_index_components()
st.success("✅ Connected to Qdrant and models loaded.")
st.subheader("Query Your Data")

query_text = st.text_input(
    "Enter your text query:", placeholder="e.g., Describe the images related to..."
)
uploaded_image = st.file_uploader(
    "Upload an image (optional):", type=["png", "jpg", "jpeg", "bmp", "gif"]
)

query_image_document = None
temp_image_path = None

if uploaded_image is not None:
    try:
        pil_image = Image.open(uploaded_image)
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_image.name)[1]
        ) as tmpfile:
            tmpfile.write(uploaded_image.getvalue())
            temp_image_path = tmpfile.name  # Get the path to the temporary file

        reader = SimpleDirectoryReader(input_files=[temp_image_path])
        docs = reader.load_data()
        if docs and isinstance(docs[0], ImageDocument):
            query_image_document = docs[0]
        else:
            st.warning("Could not process the uploaded image correctly.")
            query_image_document = None

    except Exception as e:
        st.error(f"🚨 Error processing uploaded image: {e}")
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        query_image_document = None

if st.button("Submit Query"):
    if not query_text and not query_image_document:
        st.warning("Please enter a text query or upload an image.")
    else:
        query_str = (
            query_text
            if query_text
            else "Describe the provided image and related documents."
        )
        image_docs_for_query = [query_image_document] if query_image_document else None

        with st.spinner("🧠 Querying Gemini with retrieved context..."):
            try:
                response = query_engine.query(
                    query_str, image_documents=image_docs_for_query
                )
                st.subheader("Response:")
                st.markdown(str(response))

                st.subheader("Retrieved Context Nodes:")
                for node in response.source_nodes:
                    st.text(f"Node ID: {node.node_id}, Score: {node.score:.4f}")
                    st.text(node.get_content()[:200] + "...")  # Display snippet
                    st.divider()

            except Exception as e:
                st.error(f"🚨 An error occurred during querying: {e}")

    if temp_image_path and os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
        except Exception as e:
            st.warning(f"Could not remove temporary image file: {e}")
