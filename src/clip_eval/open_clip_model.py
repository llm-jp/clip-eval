import open_clip
import torch


class WrapModel:
    """
    enable to do model.get_text_features(dict_tensor)
    """

    def __init__(self, model):
        self.model = model

    def get_text_features(self, dict_tensor):
        return self.model.encode_text(dict_tensor)

    def get_image_features(self, image):
        return self.model.encode_image(image)

    @property
    def device(self):
        return "cuda"


def load(
    model_name: str = "ViT-B-32-quickgelu",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if (
        model_name
        == "hf-hub:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k"
    ):
        model, _, transform = open_clip.create_model_and_transforms(
            "xlm-roberta-large-ViT-H-14", pretrained="frozen_laion5b_s13b_b90k"
        )
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer("xlm-roberta-large-ViT-H-14")
        return WrapModel(model), transform, tokenizer
    else:
        model, _, transform = open_clip.create_model_and_transforms(model_name)
        model = model.to(device)
        model.eval()
        model = WrapModel(model)
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, transform, tokenizer


if __name__ == "__main__":
    model, preprocess, tokenizer = load(
        "hf-hub:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k"
    )
    print(sum(p.numel() for p in model.model.parameters()))
