import streamlit as st
import os
import qdrant_client
import logging
from llama_index.core import Settings
from llama_index.core.schema import ImageDocument
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import TextNode
from PIL import Image
import tempfile
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Multimodal RAG with Gemini & Qdrant", layout="centered")
st.title("Multimodal RAG Query Interface 🖼️📄")
st.caption("Query your data using text or images with Gemini Pro Vision and Qdrant.")

QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
QDRANT_COLLECTION_NAME = "sujet-finance-multi-vdr"
EMBED_MODEL_ID = "llamaindex/vdr-2b-multi-v1"
TOP_K = 3  # Number of results to retrieve

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
def load_components():
    try:
        logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
        client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Inspect the collection to see available vector names
        collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
        logger.info(f"Collection info: {collection_info}")

        # Initialize embedding model for query encoding
        logger.info(f"Loading embedding model: {EMBED_MODEL_ID}")
        embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_ID,
            trust_remote_code=True,
            embed_batch_size=1,  # Process one at a time to avoid memory issues
        )

        # Set as default embedding model for LlamaIndex
        Settings.embed_model = embed_model

        # Initialize Gemini LLM
        logger.info("Initializing Gemini LLM")
        gemini_llm = GoogleGenAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
        Settings.llm = gemini_llm

        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(streaming=True, llm=gemini_llm)

        return (
            client,
            embed_model,
            gemini_llm,
            response_synthesizer,
        )
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}", exc_info=True)
        st.error(f"🚨 Failed to initialize components: {e}")
        st.stop()


def search_qdrant(client, embed_model, query_text=None, query_image_path=None, top_k=3):
    """
    Search Qdrant directly using text and/or image queries.
    Returns matched documents with their scores.
    """
    results = []

    # Choose the query vector and field based on input
    if query_text and query_image_path:
        # If both text and image are provided, search with both
        text_query_vector = embed_model.get_text_embedding(query_text)
        image_query_vector = embed_model.get_image_embedding(query_image_path)

        # Search text vectors using the text query
        text_results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=("text", text_query_vector),
            limit=top_k,
            with_payload=True,
        )

        # Search with image query
        image_results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=("image", image_query_vector),
            limit=top_k,
            with_payload=True,
        )

        # Combine and deduplicate results
        all_ids = set()
        combined_results = []

        for result in text_results + image_results:
            if result.id not in all_ids:
                all_ids.add(result.id)
                combined_results.append(result)

        # Sort by score and take top_k
        results = sorted(combined_results, key=lambda x: x.score, reverse=True)[:top_k]

    elif query_text:
        # Text-only query
        text_query_vector = embed_model.get_text_embedding(query_text)
        results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=("text", text_query_vector),
            limit=top_k,
            with_payload=True,
        )

    elif query_image_path:
        # Image-only query
        image_query_vector = embed_model.get_image_embedding(query_image_path)
        results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=("image", image_query_vector),
            limit=top_k,
            with_payload=True,
        )

    return results


def convert_to_nodes(search_results):
    """
    Convert Qdrant search results to LlamaIndex nodes
    """
    nodes = []

    for result in search_results:
        payload = result.payload
        score = result.score

        # Create a TextNode from the content
        node = TextNode(
            text=payload.get("content", ""),
            id_=str(result.id),
            score=score,
            metadata={
                "doc_id": payload.get("doc_id", ""),
                "score": score,
                # Add other metadata as needed
            },
        )

        # If there's an image path in the payload, we could also process it
        image_path = payload.get("image_path")
        if image_path:
            node.metadata["image_path"] = image_path

        nodes.append(node)

    return nodes


def query_with_gemini(
    query_str, source_nodes, query_image_document=None, response_synthesizer=None
):
    """
    Query Gemini with the provided query, source nodes, and optional image document.
    Returns the synthesized response.
    """
    try:
        # Create the Gemini prompt with retrieved context
        context_str = "\n\n".join(node.get_content() for node in source_nodes)

        prompt_template = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and no prior knowledge, "
            "answer the query: {query_str}"
        )

        formatted_prompt = prompt_template.format(
            context_str=context_str, query_str=query_str
        )

        # Always use direct LLM approach for simplicity
        gemini_llm = Settings.llm
        if query_image_document:
            response = gemini_llm.complete(
                formatted_prompt, image_documents=[query_image_document]
            )
        else:
            response = gemini_llm.complete(formatted_prompt)
        return response

    except Exception as e:
        logger.error(f"Error in Gemini query: {str(e)}", exc_info=True)
        raise e


# Initialize components
client, embed_model, gemini_llm, response_synthesizer = load_components()
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
        logger.info(f"Processing uploaded image: {uploaded_image.name}")
        pil_image = Image.open(uploaded_image)
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_image.name)[1]
        ) as tmpfile:
            tmpfile.write(uploaded_image.getvalue())
            temp_image_path = tmpfile.name
            logger.info(f"Image saved to temporary file: {temp_image_path}")

        # Create image document for Gemini
        query_image_document = ImageDocument(
            image_path=temp_image_path, metadata={"file_name": uploaded_image.name}
        )
        logger.info("Image document created successfully")

    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}", exc_info=True)
        st.error(f"🚨 Error processing uploaded image: {e}")
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        query_image_document = None

if st.button("Submit Query"):
    if not query_text and not query_image_document:
        st.warning("Please enter a text query or upload an image.")
        logger.warning("Query submission attempted without text or image")
    else:
        query_str = (
            query_text
            if query_text
            else "Describe the provided image and related documents."
        )
        logger.info(
            f"Processing query: '{query_str}' with image: {query_image_document is not None}"
        )

        with st.spinner("🔍 Searching Qdrant..."):
            try:
                # Search Qdrant directly
                search_results = search_qdrant(
                    client,
                    embed_model,
                    query_text=query_str,
                    query_image_path=temp_image_path if query_image_document else None,
                    top_k=TOP_K,
                )

                logger.info(f"Found {len(search_results)} results from Qdrant")

                # Convert search results to LlamaIndex nodes
                source_nodes = convert_to_nodes(search_results)

                logger.info(f"Converted {len(source_nodes)} search results to nodes")

                with st.spinner("🧠 Querying Gemini with retrieved context..."):
                    # Query Gemini with the retrieved context
                    response = query_with_gemini(
                        query_str=query_str,
                        source_nodes=source_nodes,
                        query_image_document=query_image_document,
                        response_synthesizer=response_synthesizer,
                    )

                    logger.info("Query successful, displaying response")
                    st.subheader("Response:")
                    st.markdown(str(response))

                    # Display retrieved context
                    st.subheader("Retrieved Context Nodes:")
                    for i, result in enumerate(search_results):
                        st.text(
                            f"Result {i+1}: Score={result.score:.4f}, ID={result.id}"
                        )
                        content = result.payload.get("content", "")
                        st.text(
                            content[:200] + "..." if len(content) > 200 else content
                        )

                        # If there's an image path in the payload, try to display it
                        image_path = result.payload.get("image_path")
                        if image_path and image_path.startswith(
                            ("http://", "https://")
                        ):
                            try:
                                response = requests.get(image_path)
                                img = Image.open(BytesIO(response.content))
                                st.image(img, caption=f"Image from result {i+1}")
                            except Exception as e:
                                st.text(f"Could not load image from {image_path}: {e}")

                        st.divider()

            except Exception as e:
                logger.error(f"Query error: {str(e)}", exc_info=True)
                st.error(f"🚨 An error occurred during querying: {e}")

    if temp_image_path and os.path.exists(temp_image_path):
        try:
            logger.info(f"Removing temporary file: {temp_image_path}")
            os.remove(temp_image_path)
        except Exception as e:
            logger.warning(f"Could not remove temporary image file: {str(e)}")
            st.warning(f"Could not remove temporary image file: {e}")
