import datasets
from datasets import load_dataset
import pandas as pd
import torch


def load_model(model_name: str, device) -> tuple:
    if model_name == "line-corporation/clip-japanese-base":
        from clip_eval.model.line_clip import load
    elif model_name.startswith("rinna"):
        from clip_eval.model.rinna import load
    elif model_name.startswith("hf-hub:"):
        from clip_eval.model.open_clip_model import load
    elif model_name == "stabilityai/japanese-stable-clip-vit-l-16":
        from clip_eval.model.stability_clip import load
    elif model_name.startswith("openai"):
        from clip_eval.model.clip import load
    elif model_name.startswith("jinaai"):
        from clip_eval.model.jina import load
    elif model_name.startswith("google"):
        from clip_eval.model.siglip import load
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    wrap_model, preprocess, tokenizer = load(model_name, device=device)
    return wrap_model, preprocess, tokenizer


def get_dataset(
    dataset_name: str, subcategory: str
) -> tuple[datasets.Dataset, list[str]]:
    """Get dataset for image classification
    Dataset: dataset column: image, label
    classnames: list of classnames
    """
    if dataset_name == "imagenet-1k":
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            num_proc=32,
            trust_remote_code=True,
        )
        from clip_eval.dataset.imagenet_zeroshot_data import imagenet_classnames

        classes_df = pd.DataFrame.from_dict(imagenet_classnames)
        classnames = classes_df["ja"].values.tolist()
    elif dataset_name == "recruit":
        dataset = load_dataset(
            "speed/japanese-image-classification-evaluation-datasetv2",
            split="train",
            num_proc=32,
            trust_remote_code=True,
        )
        if subcategory:
            dataset = dataset.filter(
                lambda x: x["subcategory"] == subcategory,
                num_proc=32,
            )
        classnames = dataset.unique("category")
        # category to id
        category_to_id = {category: i for i, category in enumerate(classnames)}
        dataset = dataset.map(
            lambda x: {"label": category_to_id[x["category"]]},
            remove_columns=["category"],
        )
        dataset = dataset.map(
            lambda x: {"image": x["jpg"]}, remove_columns=["jpg"], num_proc=32
        )
    elif dataset_name == "cifar100":
        from clip_eval.dataset.cifar100 import LABEL_MAPPING

        dataset = load_dataset(
            "uoft-cs/cifar100",
            split="test",
            num_proc=32,
            trust_remote_code=True,
        )
        classnames_en = dataset.features["fine_label"].names
        classnames = [LABEL_MAPPING[cls] for cls in classnames_en]
        dataset = dataset.map(
            lambda x: {"label": x["fine_label"]},
            remove_columns=["fine_label"],
            num_proc=32,
        )
        dataset = dataset.rename_column("img", "image")
    elif dataset_name == "cifar10":
        from clip_eval.dataset.cifar10 import LABEL_MAPPING

        dataset = load_dataset(
            "uoft-cs/cifar10",
            split="test",
            num_proc=32,
            trust_remote_code=True,
        )
        classnames_en = dataset.features["label"].names
        classnames = [LABEL_MAPPING[cls] for cls in classnames_en]
        dataset = dataset.rename_column("img", "image")
    elif dataset_name == "food101":
        from clip_eval.dataset.food101 import LABEL_MAPPING

        dataset = load_dataset("ethz/food101", num_proc=32, split="validation")
        classnames_en = dataset.features["label"].names
        classnames = [LABEL_MAPPING[cls] for cls in classnames_en]
    elif dataset_name == "caltech101":
        from clip_eval.dataset.caltech101 import LABEL_MAPPING

        dataset = load_dataset("flwrlabs/caltech101", num_proc=32, split="train")
        classnames_en = dataset.features["label"].names
        classnames = [LABEL_MAPPING[cls] for cls in classnames_en]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    return dataset, classnames


def compute_image_embeddings(model, images, batch_size=10):
    """
    Compute image embeddings.
    """
    image_embeddings = []

    # Process images in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        with torch.no_grad():
            image_embeddings.append(model.get_image_features(batch_images).cpu())

    # Concatenate all embeddings
    image_embeddings = torch.cat(image_embeddings, dim=0)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    return image_embeddings


def compute_text_embeddings(model, texts: list[str], tokenizer, batch_size=10):
    """
    Compute text embeddings.
    """
    text_embeddings = []

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            batch_texts = tokenizer(batch_texts).to(model.device)
        except AttributeError:
            batch_texts = batch_texts
        with torch.no_grad():
            text_embeddings.append(model.get_text_features(batch_texts).cpu())

    # Concatenate all embeddings
    text_embeddings = torch.cat(text_embeddings, dim=0)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings


def compute_embeddings(model, images, texts: list[str], tokenizer, batch_size=10):
    """
    Compute image and text embeddings.
    """
    image_embeddings = compute_image_embeddings(model, images, batch_size)
    text_embeddings = compute_text_embeddings(model, texts, tokenizer, batch_size)
    return image_embeddings, text_embeddings
