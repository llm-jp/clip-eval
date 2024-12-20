from clip_eval.eval import load_model
import io
import requests
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from clip_eval.zeroshot_retrieval import compute_embeddings
import datasets
import argparse
from clip_eval.eval import get_dataset
import os
import japanize_matplotlib  # noqa # pylint: disable=unused-import


def encode_texts(model, tokenizer, texts, device):
    tokens = tokenizer(texts).to(device)
    return model.get_text_features(tokens).cpu().detach().numpy()


def encode_images(model, process, image_urls, device):
    features = []
    for url in image_urls:
        try:
            image = Image.open(io.BytesIO(requests.get(url).content))
            image = process(image).unsqueeze(0).to(device)
            features.append(model.get_image_features(image).cpu().detach().numpy())
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            features.append(
                np.zeros((1, model.visual.output_dim))
            )  # Placeholder in case of error
    return np.vstack(features)


def plot_tsne(embeddings_tsne, texts, output_file):
    plt.figure(figsize=(50, 50))
    text_embeddings = embeddings_tsne[: len(texts)]
    image_embeddings = embeddings_tsne[len(texts) :]
    # plot each category with different color
    for i, category in enumerate(set(texts)):
        mask = np.array(texts) == category
        plt.scatter(
            text_embeddings[mask, 0], text_embeddings[mask, 1], label=category, s=100
        )
    for i, category in enumerate(set(texts)):
        mask = np.array(texts) == category
        plt.scatter(
            image_embeddings[mask, 0],
            image_embeddings[mask, 1],
            label=f"Image - {category}",
            s=100,
            marker="x",
        )

    plt.title("TSNE Plot of Text and Image Embeddings")
    # plot legend only text label
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=14)

    plt.savefig(output_file)


def compute_similarity_matrix(embeddings):
    size = embeddings.shape[0]
    similarity_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
    return similarity_matrix


def plot_similarity_matrix(similarity_matrix, texts, output_file):
    plt.figure(figsize=(len(texts), len(texts)))
    # Plot the similarity matrix
    plt.imshow(similarity_matrix, cmap="viridis")
    # log scale colorbar
    plt.xticks(range(len(texts) * 2), texts + texts, rotation=90, fontsize=14)
    plt.yticks(range(len(texts) * 2), texts + texts, fontsize=14)

    plt.colorbar()
    plt.title("Similarity Matrix of Text and Image Embeddings")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, process, tokenizer = load_model(args.model_name, args.device)
    dataset, classnames = get_dataset(args.dataset_name, args.subcategory)

    # dedupulicate by category
    new_dataset = []
    seen = set()
    for data in dataset:
        classname = classnames[data["label"]]
        if classname not in seen:
            new_dataset.append(data)
            seen.add(classname)

    dataset = datasets.Dataset.from_dict(
        {
            "image": [data["image"] for data in new_dataset],
            "category": [classnames[data["label"]] for data in new_dataset],
        }
    )
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
    # TSNE plot
    # embeddings_tsne = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    # plot_tsne(
    #     embeddings_tsne,
    #     texts,
    #     f"{result_dir}/tsne_{args.model_name.split('/')[-1]}.png",
    # )

    # Similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    plot_similarity_matrix(
        similarity_matrix,
        texts,
        f"{result_dir}/{args.model_name.split('/')[-1]}.png",
    )
