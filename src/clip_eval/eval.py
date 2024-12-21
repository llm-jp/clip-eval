import pandas as pd
import torch
from datasets import load_dataset
from clip_eval.zeroshot_classification import ClassificationCallback
import os
import argparse
from logging import getLogger, basicConfig
import json
from clip_eval.utils import load_model, get_dataset


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
        "--subcategory",
        type=str,
        default=None,
        help="Subcategory (i.e. jafacility20) for recruit dataset",
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


if __name__ == "__main__":
    args = get_args()
    wrap_model, preprocess, tokenizer = load_model(args.model_name, args.device)
    dataset2task = {
        "imagenet-1k": "classification",
        "recruit": "classification",
        "cifar100": "classification",
        "cifar10": "classification",
        "food101": "classification",
        "caltech101": "classification",
        "crossmodal3600": "retrieval",
    }
    task = dataset2task[args.dataset_name]

    if task == "classification":
        from clip_eval.dataset.imagenet_zeroshot_data import imagenet_templates

        templates_df = pd.DataFrame.from_dict(imagenet_templates)
        templates = templates_df["ja"].values.tolist()

        dataset, classnames = get_dataset(args.dataset_name, args.subcategory)

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
    elif task == "retrieval":
        from clip_eval.zeroshot_retrieval import evaluate_retrieval, compute_embeddings

        # TODO: Publish the dataset to HF
        ds = load_dataset(
            "webdataset", data_files="crossmodal-3600/data.tar", split="train"
        )
        images = ds["jpg"]
        texts = ds["txt"]

        images = [preprocess(image) for image in images]
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images).to(wrap_model.device)

        image_embeddings, text_embeddings = compute_embeddings(
            wrap_model, images, texts, tokenizer, args.batch_size
        )
        top_k_list = [1, 5, 10]
        result_dict = {"t2i_recall": {}, "i2t_recall": {}}
        for top_k in top_k_list:
            t2i_recall_at_k, i2t_recall_at_k = evaluate_retrieval(
                image_embeddings, text_embeddings, top_k=top_k
            )
            result_dict["t2i_recall"][f"top{top_k}"] = t2i_recall_at_k
            result_dict["i2t_recall"][f"top{top_k}"] = i2t_recall_at_k

    result_dir = f"{args.result_dir}/{args.dataset_name}"
    if args.subcategory:
        result_dir = f"{result_dir}/{args.subcategory}"
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/{args.model_name.replace('/', '-')}.json", "w") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
