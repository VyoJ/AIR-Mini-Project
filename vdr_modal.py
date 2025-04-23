import modal
import os
import json
import shutil
import warnings
from tqdm import tqdm

app = modal.App("vdr-indexing-app")

MODEL_ID = "llamaindex/vdr-2b-multi-v1"
DATASET_NAME = "sujet-ai/Sujet-Finance-QA-Vision-100k"
SPLIT_NAME = "train"
NUM_ITEMS_TO_INDEX = 1000
TEMP_IMAGE_DIR = "/tmp/temp_images"
QDRANT_COLLECTION_NAME = "sujet-finance-multi-vdr"
BATCH_SIZE = 16

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vdr_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "transformers",
        "fastapi[standard]",
        "llama-index-embeddings-huggingface",
        "qdrant-client>=1.7.0",
        "datasets",
        "Pillow",
        "tqdm",
        "accelerate",
    )
    .apt_install("git", "clang", "build-essential")
)

model_volume = modal.Volume.from_name("vdr-model-cache", create_if_missing=True)
# temp_image_volume = modal.Volume.from_name("temp-image-store", create_if_missing=True)


@app.function(
    image=vdr_image,
    gpu="l40s",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": model_volume},
    secrets=[modal.Secret.from_dotenv()],
)
def index_finance_data():
    """
    Loads data, generates embeddings using VDR model, and indexes into Qdrant.
    """
    from datasets import load_dataset
    from qdrant_client import QdrantClient, models
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import torch

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_api_key = os.environ["QDRANT_API_KEY"]

    if not qdrant_url or not qdrant_api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY must be set in environment variables/secrets."
        )

    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing Qdrant client...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)

    print(f"Loading embedding model: {MODEL_ID}...")
    model = HuggingFaceEmbedding(
        model_name=MODEL_ID,
        device=device,
        trust_remote_code=True,
        cache_folder="/root/.cache/huggingface/hub",
    )
    print("Embedding model loaded.")

    print(f"Loading dataset {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
        dataset = dataset.select(range(NUM_ITEMS_TO_INDEX))
    except Exception as e:
        print(f"Direct load failed: {e}. Attempting to stream...")
        try:
            dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME, streaming=True)
            dataset = dataset.take(NUM_ITEMS_TO_INDEX)
        except Exception as stream_e:
            print(f"Streaming failed: {stream_e}. Exiting.")
            return

    print(f"Processing {NUM_ITEMS_TO_INDEX} items and saving images...")
    if not os.path.exists(TEMP_IMAGE_DIR):
        os.makedirs(TEMP_IMAGE_DIR)

    image_paths = []
    contents = []
    metadata_list = []
    processed_ids = set()

    item_iterator = iter(dataset)
    for i in tqdm(range(NUM_ITEMS_TO_INDEX), desc="Processing Dataset Items"):
        try:
            item = next(item_iterator)
        except StopIteration:
            print(f"Warning: Dataset contains fewer than {NUM_ITEMS_TO_INDEX} items.")
            break

        try:
            doc_id = item.get("doc_id")
            unique_id = f"{doc_id}_{i}" if doc_id else f"item_{i}"
            if unique_id in processed_ids:
                print(f"Warning: Skipping duplicate unique_id {unique_id}")
                continue
            processed_ids.add(unique_id)

            content = item.get("content", "")
            qa_pairs = item.get("qa_pairs", [])
            image = item.get("image")

            if image is None or not hasattr(image, "save"):
                print(
                    f"Warning: Skipping item {unique_id} due to missing or invalid image."
                )
                continue

            image_filename = f"{unique_id.replace('/', '_')}.png"
            image_path = os.path.join(TEMP_IMAGE_DIR, image_filename)

            image.save(image_path)

            image_paths.append(image_path)
            contents.append(content)
            metadata_list.append(
                {
                    "doc_id": doc_id,
                    "original_index": i,
                    "qa_pairs_json": json.dumps(qa_pairs),
                }
            )

        except Exception as e:
            print(
                f"Error processing item {i} (unique_id: {unique_id if 'unique_id' in locals() else 'N/A'}): {e}"
            )
            continue

    num_processed = len(image_paths)
    if num_processed == 0:
        print("Error: No items were successfully processed. Exiting.")
        return
    print(f"Successfully processed {num_processed} items.")

    print("Generating embeddings...")
    text_embeddings = []
    image_embeddings = []

    try:
        if contents:
            for i in tqdm(
                range(0, num_processed, BATCH_SIZE), desc="Text Embedding Batches"
            ):
                batch_contents = contents[i : i + BATCH_SIZE]
                batch_embeddings = model.get_text_embedding_batch(
                    batch_contents, show_progress=False
                )
                text_embeddings.extend(batch_embeddings)
        else:
            print("Warning: No content found to embed.")

        if image_paths:
            for i in tqdm(
                range(0, num_processed, BATCH_SIZE), desc="Image Embedding Batches"
            ):
                batch_image_paths = image_paths[i : i + BATCH_SIZE]
                batch_embeddings = model.get_image_embedding_batch(
                    batch_image_paths, show_progress=False
                )
                image_embeddings.extend(batch_embeddings)
        else:
            print("Warning: No images found to embed.")

    except Exception as e:
        print(f"Error during embedding generation: {e}")
        if not text_embeddings or not image_embeddings:
            print("Critical embedding error. Exiting.")
            return

    if len(text_embeddings) != num_processed or len(image_embeddings) != num_processed:
        print(
            f"Error: Mismatch after embedding generation. Processed: {num_processed}, Text Embeddings: {len(text_embeddings)}, Image Embeddings: {len(image_embeddings)}"
        )
        print("Attempting to proceed with the minimum consistent count.")
        min_count = min(len(text_embeddings), len(image_embeddings), num_processed)
        if min_count == 0:
            print("No consistent data to index. Exiting.")
            return
        text_embeddings = text_embeddings[:min_count]
        image_embeddings = image_embeddings[:min_count]
        metadata_list = metadata_list[:min_count]
        contents = contents[:min_count]
        image_paths = image_paths[:min_count]
        num_processed = min_count
        print(f"Adjusted count to index: {num_processed}")

    if num_processed == 0:
        print("No data left to index after consistency checks. Exiting.")
        return

    print(f"Embeddings generated for {num_processed} items.")
    embed_dim = len(text_embeddings[0])
    print(f"Detected embedding dimension: {embed_dim}")

    print(f"Checking/Creating Qdrant collection: {QDRANT_COLLECTION_NAME}")
    try:
        if not client.collection_exists(QDRANT_COLLECTION_NAME):
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config={
                    "text": models.VectorParams(
                        size=embed_dim, distance=models.Distance.COSINE
                    ),
                    "image": models.VectorParams(
                        size=embed_dim,
                        distance=models.Distance.COSINE,
                    ),
                },
            )
            print(f"Collection '{QDRANT_COLLECTION_NAME}' created.")
        else:
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")
    except Exception as e:
        print(f"Error interacting with Qdrant collection: {e}")
        return

    print(
        f"Uploading {num_processed} points to collection '{QDRANT_COLLECTION_NAME}'..."
    )
    points_to_upload = []
    for idx in range(num_processed):
        points_to_upload.append(
            models.PointStruct(
                id=idx,
                vector={
                    "text": text_embeddings[idx],
                    "image": image_embeddings[idx],
                },
                payload={
                    "doc_id": metadata_list[idx]["doc_id"],
                    "content": contents[idx],
                    "qa_pairs_json": metadata_list[idx]["qa_pairs_json"],
                },
            )
        )

    qdrant_batch_size = 100
    for i in tqdm(
        range(0, num_processed, qdrant_batch_size), desc="Uploading Points to Qdrant"
    ):
        batch_points = points_to_upload[i : i + qdrant_batch_size]
        try:
            client.upload_points(
                collection_name=QDRANT_COLLECTION_NAME,
                points=batch_points,
                wait=False,
            )
        except Exception as e:
            print(f"Error uploading batch starting at index {i}: {e}")
    print("Point upload requests submitted.")

    print(f"Removing temporary image directory: {TEMP_IMAGE_DIR}")
    if os.path.exists(TEMP_IMAGE_DIR):
        try:
            shutil.rmtree(TEMP_IMAGE_DIR)
            print("Temporary directory removed.")
        except OSError as e:
            print(f"Error removing temporary directory {TEMP_IMAGE_DIR}: {e}")

    print("--- Indexing process finished ---")


@app.local_entrypoint()
def main():
    """
    Local entrypoint to run the indexing function.
    Requires QDRANT_URL and QDRANT_API_KEY to be set in the local environment.
    """
    print("Running indexing function locally...")
    index_finance_data.remote()
    print("Indexing function call completed.")
