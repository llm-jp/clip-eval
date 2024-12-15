import pandas as pd
import torch
from datasets import load_dataset
from japanese_clip.utils.callbacks import ClassificationCallback
import os
import argparse
from logging import getLogger, basicConfig
import json

logger = getLogger(__name__)
logger.setLevel("INFO")
# basic config
basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args():
    parser = argparse.ArgumentParser(description="Zero-shot Image Classification")
    parser.add_argument(
        "--model_name",
        type=str,
        default="xlm-roberta-large-ViT-H-14",
        help="Model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imagenet-1k",
        help="Dataset name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Result directory",
    )
    args = parser.parse_args()
    return args


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
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    wrap_model, preprocess, tokenizer = load(model_name, device=device)
    return wrap_model, preprocess, tokenizer


if __name__ == "__main__":
    args = get_args()
    wrap_model, preprocess, tokenizer = load_model(args.model_name, args.device)
    from clip_eval.japanese_clip.utils.imagenet_zeroshot_data import imagenet_templates

    templates_df = pd.DataFrame.from_dict(imagenet_templates)
    templates = templates_df["ja"].values.tolist()

    if args.dataset_name == "imagenet-1k":
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            num_proc=32,
            trust_remote_code=True,
        )
        from japanese_clip.utils.imagenet_zeroshot_data import imagenet_classnames

        classes_df = pd.DataFrame.from_dict(imagenet_classnames)
        classnames = classes_df["ja"].values.tolist()

    elif args.dataset_name == "recruit":
        dataset = load_dataset(
            "speed/japanese-image-classification-evaluation-dataset",
            split="train",
            num_proc=32,
            trust_remote_code=True,
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
    elif args.dataset_name == "cifar100":
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
    elif args.dataset_name == "cifar10":
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
    elif args.dataset_name == "food101":
        from clip_eval.dataset.food101 import LABEL_MAPPING

        dataset = load_dataset("ethz/food101", num_proc=32, split="validation")
        classnames_en = dataset.features["label"].names
        classnames = [LABEL_MAPPING[cls] for cls in classnames_en]
    elif args.dataset_name == "caltech101":
        from clip_eval.dataset.caltech101 import LABEL_MAPPING

        dataset = load_dataset("flwrlabs/caltech101", num_proc=32, split="train")
        classnames_en = dataset.features["label"].names
        classnames = [LABEL_MAPPING[cls] for cls in classnames_en]
    else:
        raise ValueError(f"Unknown dataset_name: {args.dataset_name}")

    def collate_fn(batch):
        # images = [transform(x["image"].convert("RGB")) for x in batch]
        images = [preprocess(x["image"].convert("RGB")) for x in batch]
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        targets = torch.tensor([x["label"] for x in batch])
        return images, targets

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    logger.info(
        f"Start Zero-shot Image Classification: {args.model_name} on {args.dataset_name}"
    )
    logger.info(f"{len(classnames)} classes, {len(templates)} templates")

    callback = ClassificationCallback(classnames, templates, dataloader)
    result_dict = callback.zeroshot(wrap_model, tokenizer)
    result_dir = f"{args.result_dir}/{args.dataset_name}"
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/{args.model_name.replace('/', '-')}.json", "w") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
