import torch
import numpy as np
import matplotlib.pyplot as plt
from clip_eval.utils import compute_embeddings
import datasets
import argparse
from clip_eval.utils import get_dataset, load_model
import os
import japanize_matplotlib  # noqa # pylint: disable=unused-import


def compute_similarity_matrix(embeddings):
    normalized_embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    return similarity_matrix


def plot_similarity_matrix(similarity_matrix, texts, output_file):
    fig_size = min(50, len(texts) // 2)
    plt.figure(figsize=(fig_size, fig_size))
    # Plot the similarity matrix
    plt.imshow(similarity_matrix, cmap="viridis")
    plt.xticks(range(len(texts) * 2), texts + texts, rotation=90, fontsize=14)
    plt.yticks(range(len(texts) * 2), texts + texts, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Analysis")
    parser.add_argument(
        "--model_name", type=str, default="line-corporation/clip-japanese-base"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imagenet-1k",
    )
    parser.add_argument("--subcategory", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, process, tokenizer = load_model(args.model_name, args.device)
    dataset, classnames = get_dataset(args.dataset_name, args.subcategory)
    if args.num_classes:
        classnames = classnames[: args.num_classes]
        dataset = dataset.filter(
            lambda x: classnames[x["label"]],
            num_proc=32,
        )
    # pick only one example per class
    seen = set()
    new_dataset = [
        data
        for data in dataset
        if (classname := classnames[data["label"]]) not in seen
        and not seen.add(classname)
    ]
    dataset = datasets.Dataset.from_dict(
        {
            "image": [data["image"] for data in new_dataset],
            "category": [classnames[data["label"]] for data in new_dataset],
        }
    )
    assert len(dataset) == len(classnames)

    texts = dataset["category"]
    images = [process(image) for image in dataset["image"]]

    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images).to(model.device)

    # Compute embeddings for images and texts
    image_embeddings, text_embeddings = compute_embeddings(
        model, images, texts, tokenizer, args.batch_size
    )

    # Combine embeddings
    embeddings = np.concatenate([text_embeddings, image_embeddings], axis=0)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    result_dir = f"{args.result_dir}/{args.dataset_name}"
    if args.subcategory:
        result_dir = f"{result_dir}/{args.subcategory}"
    os.makedirs(result_dir, exist_ok=True)

    # Similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    plot_similarity_matrix(
        similarity_matrix,
        texts,
        f"{result_dir}/{args.model_name.split('/')[-1]}.png",
    )
