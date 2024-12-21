import torch
import numpy as np
import matplotlib.pyplot as plt
from clip_eval.utils import compute_image_embeddings, compute_text_embeddings
import argparse
from clip_eval.utils import get_dataset, load_model
import os
import japanize_matplotlib  # noqa # pylint: disable=unused-import
from sklearn.manifold import TSNE


def plot_tsne(embedding: np.ndarray, texts: list[str], output_file: str):
    plt.figure(figsize=(10, 10))
    for i, category in enumerate(set(texts)):
        mask = np.array(texts) == category
        plt.scatter(embedding[mask, 0], embedding[mask, 1], label=category)

    plt.legend()
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
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, process, tokenizer = load_model(args.model_name, args.device)
    dataset, classnames = get_dataset(args.dataset_name, args.subcategory)
    dataset = dataset.map(
        lambda x: {"category": classnames[x["label"]]},
        remove_columns=["label"],
    )
    # only first 10 classes
    classnames = classnames[: args.num_classes]
    dataset = dataset.filter(
        lambda x: x["category"] in classnames,
        num_proc=32,
    )
    texts = dataset["category"]
    images = [process(image) for image in dataset["image"]]
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images).to(model.device)

    # Compute embeddings for images and texts
    image_embeddings = compute_image_embeddings(model, images, args.batch_size)
    text_embeddings = compute_text_embeddings(
        model, classnames, tokenizer, args.batch_size
    )

    # Combine embeddings
    text_embeddings_tsne = TSNE(
        n_components=2, perplexity=3, random_state=42
    ).fit_transform(text_embeddings)
    image_embeddings_tsne = TSNE(n_components=2, random_state=42).fit_transform(
        image_embeddings
    )

    result_dir = f"{args.result_dir}/{args.dataset_name}/tsne"
    if args.subcategory:
        result_dir = f"{result_dir}/{args.subcategory}/tsne"
    os.makedirs(result_dir, exist_ok=True)
    # TSNE plot
    plot_tsne(
        text_embeddings_tsne,
        classnames,
        f"{result_dir}/{args.model_name.split('/')[-1]}_text_tsne.png",
    )
    plot_tsne(
        image_embeddings_tsne,
        texts,
        f"{result_dir}/{args.model_name.split('/')[-1]}_image_tsne.png",
    )
