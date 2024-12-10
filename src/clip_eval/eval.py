import pandas as pd
import torch
from datasets import load_dataset
from japanese_clip.utils.callbacks import ImagenetClassificationCallback

import argparse


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
        default="ILSVRC/imagenet-1k",
        help="Dataset name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    device = args.device
    # load model, tokenizer
    # model_name = "line-corporation/clip-japanese-base"
    # model_name = "rinna/japanese-clip-vit-b-16"
    # model_name = "rinna/japanese-cloob-vit-b-16"
    # model_name = "xlm-roberta-large-ViT-H-14"

    model_name = args.model_name
    if model_name == "line-corporation/clip-japanese-base":
        from line_clip import load

        wrap_model, preprocess, tokenizer = load(
            "line-corporation/clip-japanese-base", device=device
        )
    elif (
        model_name == "rinna/japanese-clip-vit-b-16"
        or model_name == "rinna/japanese-cloob-vit-b-16"
    ):
        from rinna import load

        wrap_model, preprocess, tokenizer = load(model_name, device=device)
    elif (
        model_name
        == "hf-hub:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k"
        or model_name
        == "hf-hub:speed/llm-jp-roberta-pretrained-ViT-B-16-relaion-1.5B-lr1e-4-bs8k-accum4-2024112-epoch87"
    ):
        from open_clip_model import load

        wrap_model, preprocess, tokenizer = load(model_name, device=device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    print("Model: ", model_name)

    if args.dataset_name == "ILSVRC/imagenet-1k":
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            num_proc=32,
            trust_remote_code=True,
        )
        from japanese_clip.utils.imagenet_zeroshot_data import (
            imagenet_templates,
            imagenet_classnames,
        )

        templates_df = pd.DataFrame.from_dict(imagenet_templates)
        classes_df = pd.DataFrame.from_dict(imagenet_classnames)
        imagenet_classes = classes_df["ja"].values.tolist()
        imagenet_templates_lan = templates_df["ja"].values.tolist()
        print(
            f"{len(imagenet_classes)} classes, {len(imagenet_templates_lan)} templates"
        )
    elif args.dataset_name == "speed/japanese-image-classification-evaluation-dataset":
        dataset = load_dataset(
            "speed/japanese-image-classification-evaluation-dataset",
            split="train",
            num_proc=32,
            trust_remote_code=True,
        )
        from japanese_clip.utils.imagenet_zeroshot_data import (
            imagenet_templates,
            imagenet_classnames,
        )

        templates_df = pd.DataFrame.from_dict(imagenet_templates)
        # imagenet_classes = dataset.features["label"].names
        imagenet_classes = dataset.unique("category")
        imagenet_templates_lan = templates_df["ja"].values.tolist()
        # category to id
        category_to_id = {category: i for i, category in enumerate(imagenet_classes)}
        dataset = dataset.map(
            lambda x: {"label": category_to_id[x["category"]]},
            remove_columns=["category"],
        )
        dataset = dataset.map(
            lambda x: {"image": x["jpg"]}, remove_columns=["jpg"], num_proc=32
        )
        dataset = dataset.filter(lambda example: example["image"] is not None)
        print(len(dataset))

        print(
            f"{len(imagenet_classes)} classes, {len(imagenet_templates_lan)} templates"
        )

    # transform = transforms.Compose([
    #         transforms.Resize((224, 224)),  # Resize to 224x224
    #         transforms.ToTensor(),  # Convert PIL Image to tensor
    # ])
    def collate_fn(batch):
        # images = [transform(x["image"].convert("RGB")) for x in batch]
        images = [preprocess(x["image"].convert("RGB")) for x in batch]
        images = torch.stack(images)
        targets = torch.tensor([x["label"] for x in batch])
        return images, targets

    imagenet_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    # print(next(iter(imagenet_dataloader)))
    # print(imagenet_dataloader)
    # print(next(iter(imagenet_dataloader))[0].shape)
    imagenet_callback = ImagenetClassificationCallback(
        imagenet_classes, imagenet_templates_lan, imagenet_dataloader
    )
    result_dict = imagenet_callback.zeroshot(wrap_model, tokenizer)
    print(result_dict)
    # prints: {"top1": xx, "top5": xx, "top10": xx, "top100": xx}
