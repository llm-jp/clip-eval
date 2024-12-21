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
        # MEMO: OpenCLIP's CustomTextCLIP object has no attribute 'device'
        return next(self.model.parameters()).device


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
    import io
    import requests
    from PIL import Image
    import torch

    model_name = "hf-hub:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k"
    # model_name = "hf-hub:speed/llm-jp-roberta-pretrained-ViT-B-16-relaion-1.5B-lr1e-4-bs8k-accum4-2024112-epoch87"
    model, processor, tokenizer = load(model_name)

    image = Image.open(
        io.BytesIO(
            requests.get(
                "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260"
            ).content
        )
    )

    images = processor(image).unsqueeze(0)
    print(images.shape)
    # text = tokenizer(["doc", "cat", "elephant"])
    text = tokenizer(["犬", "猫", "象"])
    with torch.no_grad():
        image_features = model.get_image_features(images.to(model.device))
        text_features = model.get_text_features(text.to(model.device))
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print(text_probs)
