import torch
import numpy as np
from datasets import load_dataset
from clip_eval.utils import compute_embeddings


def evaluate_retrieval(image_embeddings, text_embeddings, top_k=5):
    """
    Evaluate text retrieval and image retrieval.
    """
    # Compute similarity matrix
    similarity_matrix = text_embeddings @ image_embeddings.T
    similarity_matrix = torch.tensor(similarity_matrix)

    # Text-to-Image Retrieval (text query to find corresponding images)
    t2i_ranks = torch.argsort(similarity_matrix, dim=1, descending=True)
    t2i_recall = []
    for i, rank in enumerate(t2i_ranks):
        t2i_recall.append((rank[:top_k] == i).any().item())
    t2i_recall_at_k = np.mean(t2i_recall)

    # Image-to-Text Retrieval (image query to find corresponding texts)
    i2t_ranks = torch.argsort(similarity_matrix.T, dim=1, descending=True)
    i2t_recall = []
    for i, rank in enumerate(i2t_ranks):
        i2t_recall.append((rank[:top_k] == i).any().item())
    i2t_recall_at_k = np.mean(i2t_recall)

    return t2i_recall_at_k, i2t_recall_at_k


if __name__ == "__main__":
    from clip_eval.model import line_clip

    ds = load_dataset(
        "webdataset", data_files="crossmodal-3600/data.tar", split="train"
    )
    model, processor, tokenizer = line_clip.load("line-corporation/clip-japanese-base")

    images = ds["jpg"]
    texts = ds["txt"]

    image_emb_list = []
    text_emb_list = []

    images = [processor(image) for image in images]
    images = torch.stack(images).to(model.device)

    image_embeddings, text_embeddings = compute_embeddings(
        model, images, texts, tokenizer
    )

    print(evaluate_retrieval(image_embeddings, text_embeddings, top_k=1))
    print(evaluate_retrieval(image_embeddings, text_embeddings, top_k=5))
