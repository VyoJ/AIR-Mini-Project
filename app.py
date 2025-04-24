import streamlit as st
import os
import qdrant_client
import logging
from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import TextNode
from PIL import Image
import requests
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Multimodal RAG with Gemini & Qdrant", layout="centered")
st.title("Multimodal RAG Query Interface 📄")
st.caption("Query your data using text with Gemini Pro and Qdrant.")

QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
QDRANT_COLLECTION_NAME = "sujet-finance-multi-vdr"
EMBED_MODEL_ID = "llamaindex/vdr-2b-multi-v1"
TOP_K = 3

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

        collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
        logger.info(f"Collection info: {collection_info}")

        logger.info(f"Loading embedding model: {EMBED_MODEL_ID}")
        embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_ID,
            trust_remote_code=True,
            embed_batch_size=1,
        )

        Settings.embed_model = embed_model

        logger.info("Initializing Gemini LLM")
        gemini_llm = GoogleGenAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
        Settings.llm = gemini_llm

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


def search_qdrant(client, embed_model, query_text, top_k=3):
    """
    Search Qdrant directly using text queries against both text and image vectors.
    Returns matched documents with their scores.
    """
    results = []
    all_ids = set()
    combined_results = []

    text_query_vector = embed_model.get_text_embedding(query_text)
    text_results = client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=("text", text_query_vector),
        limit=top_k,
        with_payload=True,
    )
    image_results = client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=(
            "image",
            text_query_vector,
        ),
        limit=top_k,
        with_payload=True,
    )

    for result in text_results + image_results:
        if result.id not in all_ids:
            all_ids.add(result.id)
            combined_results.append(result)

    results = sorted(combined_results, key=lambda x: x.score, reverse=True)[:top_k]

    return results


def convert_to_nodes(search_results):
    """
    Convert Qdrant search results to LlamaIndex nodes
    """
    nodes = []

    for result in search_results:
        payload = result.payload
        score = result.score

        node = TextNode(
            text=payload.get("content", ""),
            id_=str(result.id),
            score=score,
            metadata={
                "doc_id": payload.get("doc_id", ""),
                "score": score,
            },
        )

        image_path = payload.get("image_path")
        if image_path:
            node.metadata["image_path"] = image_path
        image_data = payload.get("image")
        if image_data:
            node.metadata["has_binary_image"] = True

        nodes.append(node)

    return nodes


def query_with_gemini(query_str, source_nodes, response_synthesizer=None):
    """
    Query Gemini with the provided query and source nodes.
    Returns the synthesized response.
    """
    try:
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

        gemini_llm = Settings.llm
        response = gemini_llm.complete(formatted_prompt)
        return response

    except Exception as e:
        logger.error(f"Error in Gemini query: {str(e)}", exc_info=True)
        raise e


client, embed_model, gemini_llm, response_synthesizer = load_components()
st.success("✅ Connected to Qdrant and models loaded.")
st.subheader("Query Your Data")

query_text = st.text_input(
    "Enter your text query:", placeholder="e.g., Describe the images related to..."
)

if st.button("Submit Query"):
    if not query_text:
        st.warning("Please enter a text query.")
        logger.warning("Query submission attempted without text")
    else:
        query_str = query_text
        logger.info(f"Processing query: '{query_str}'")

        with st.spinner("🔍 Searching Qdrant..."):
            try:
                search_results = search_qdrant(
                    client,
                    embed_model,
                    query_text=query_str,
                    top_k=TOP_K,
                )

                logger.info(f"Found {len(search_results)} results from Qdrant")

                source_nodes = convert_to_nodes(search_results)

                logger.info(f"Converted {len(source_nodes)} search results to nodes")

                with st.spinner("🧠 Querying Gemini with retrieved context..."):
                    response = query_with_gemini(
                        query_str=query_str,
                        source_nodes=source_nodes,
                        response_synthesizer=response_synthesizer,
                    )

                    logger.info("Query successful, displaying response")
                    st.subheader("Response:")
                    st.markdown(str(response))

                    st.subheader("Retrieved Context Nodes:")
                    for i, result in enumerate(search_results):
                        st.text(
                            f"Result {i+1}: Score={result.score:.4f}, ID={result.id}"
                        )
                        content = result.payload.get("content", "")
                        st.text(
                            content[:200] + "..." if len(content) > 200 else content
                        )

                        image_data = result.payload.get("image")
                        image_path = result.payload.get("image_path")

                        if image_data:
                            try:
                                img = Image.open(BytesIO(image_data))
                                st.image(img, caption=f"Image from result {i+1}")
                                logger.info(f"Displayed binary image from result {i+1}")
                            except Exception as e:
                                logger.error(f"Could not load binary image: {str(e)}")
                                st.text(f"Could not load binary image: {e}")

                        elif image_path and image_path.startswith(
                            ("http://", "https://")
                        ):
                            try:
                                response = requests.get(image_path)
                                response.raise_for_status()
                                img = Image.open(BytesIO(response.content))
                                st.image(img, caption=f"Image from result {i+1}")
                                logger.info(f"Displayed image from URL {image_path}")
                            except requests.exceptions.RequestException as e:
                                logger.error(
                                    f"Could not fetch image from URL {image_path}: {str(e)}"
                                )
                                st.text(f"Could not fetch image from {image_path}: {e}")
                            except Exception as e:
                                logger.error(
                                    f"Could not load image from URL {image_path}: {str(e)}"
                                )
                                st.text(f"Could not load image from {image_path}: {e}")
                        elif image_path and os.path.exists(image_path):
                            try:
                                img = Image.open(image_path)
                                st.image(
                                    img, caption=f"Image from result {i+1} (local path)"
                                )
                                logger.info(
                                    f"Displayed image from local path {image_path}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Could not load image from local path {image_path}: {str(e)}"
                                )
                                st.text(f"Could not load image from {image_path}: {e}")
                        elif image_path:
                            st.text(
                                f"Image path found but not displayable: {image_path}"
                            )
                            logger.warning(
                                f"Image path {image_path} is not a valid URL or local file."
                            )

                        st.divider()

            except Exception as e:
                logger.error(f"Query error: {str(e)}", exc_info=True)
                st.error(f"🚨 An error occurred during querying: {e}")
