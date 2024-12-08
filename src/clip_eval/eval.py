import pandas as pd
import torch
from datasets import load_dataset
from japanese_clip.utils.callbacks import ImagenetClassificationCallback


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model, tokenizer
    # model_name = "line-corporation/clip-japanese-base"
    #model_name = "rinna/japanese-clip-vit-b-16"
    # model_name = "rinna/japanese-cloob-vit-b-16"
    model_name = "hf-hub:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k"

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
        model_name == "hf-hub:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k"
    ):
        from open_clip_model import load
        wrap_model, preprocess, tokenizer = load(model_name, device=device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    print("Model: ", model_name)

    ##################################
    # Zero-shot Image Classification #
    ##################################
    from japanese_clip.utils.imagenet_zeroshot_data import (
        imagenet_templates,
        imagenet_classnames,
    )

    templates_df = pd.DataFrame.from_dict(imagenet_templates)
    classes_df = pd.DataFrame.from_dict(imagenet_classnames)
    imagenet_classes = classes_df["ja"].values.tolist()
    imagenet_templates_lan = templates_df["ja"].values.tolist()
    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates_lan)} templates")

    dataset = load_dataset(
        "ILSVRC/imagenet-1k", split="validation", num_proc=32, trust_remote_code=True
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
        batch_size=64,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    print(next(iter(imagenet_dataloader)))

    # dataset = torchvision.datasets.ImageNet("imagenet_val", split="val", transform=preprocess)
    # imagenet_dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=64,
    #     shuffle=False,
    #     num_workers=2,
    #     persistent_workers=True,
    #     drop_last=False,
    #     collate_fn=None,
    # )

    # print(imagenet_dataloader)
    # print(next(iter(imagenet_dataloader))[0].shape)
    # import os
    # os._exit(0)
    imagenet_callback = ImagenetClassificationCallback(
        imagenet_classes, imagenet_templates_lan, imagenet_dataloader
    )
    result_dict = imagenet_callback.zeroshot(wrap_model, tokenizer)
    print(result_dict)
    # prints: {"top1": xx, "top5": xx, "top10": xx, "top100": xx}
