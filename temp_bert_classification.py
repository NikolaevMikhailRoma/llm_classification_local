from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F

import json

def classify_with_bert(data_path: str = "examples/1/messages.json"):
    """
    Performs text classification using Sentence-BERT by finding the closest
    category based on cosine similarity.

    Args:
        data_path (str): The path to the JSON file containing categories and messages.
    """
    # 1. Data Loading
    # Load categories and messages from the specified JSON file
    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        categories = data["categories"]
        messages = data["messages"]
        print("Data loaded successfully.")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading data file: {e}")
        return

    # 2. Model Loading
    # The model will be downloaded on the first run and cached automatically.
    # Subsequent runs will load from the cache.
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    print(f"Loading model: {model_name}...")
    print("(This might take a moment on the first run as the model is downloaded)")
    model = SentenceTransformer(model_name)
    print("Model loaded successfully.")

    # 3. Encoding
    # Convert categories and messages into numerical vectors (embeddings)
    print("\nEncoding categories and messages into vectors...")
    category_embeddings = model.encode(categories, convert_to_tensor=True)
    message_embeddings = model.encode(messages, convert_to_tensor=True)
    print("Encoding complete.")

    # 4. Classification via Cosine Similarity
    # Compute cosine similarity between each message vector and all category vectors
    cosine_scores = util.cos_sim(message_embeddings, category_embeddings)

    # 5. Output Results
    print("\n--- Classification Results ---\n")
    for i, message in enumerate(messages):
        # Get raw cosine similarity scores for the current message
        raw_scores = cosine_scores[i]

        # Apply softmax to convert scores to a probability distribution
        normalized_scores = F.softmax(raw_scores, dim=0)

        # Pair each category with its normalized score (probability)
        score_pairs = []
        for j, category in enumerate(categories):
            score_pairs.append((category, normalized_scores[j].item()))

        # Sort the pairs by score in descending order
        sorted_scores = sorted(score_pairs, key=lambda x: x[1], reverse=True)

        print(f'Message: "{message}"')
        for category, score in sorted_scores:
            # Highlight the top score and format as percentage
            if category == sorted_scores[0][0]:
                print(f'  -> {category:<10} ({score:.2%})')
            else:
                print(f'     {category:<10} ({score:.2%})')
        print()  # Add a newline for better readability

    print("--- End of Report ---")

if __name__ == "__main__":
    classify_with_bert()
