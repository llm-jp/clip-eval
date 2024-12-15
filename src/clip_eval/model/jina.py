# !pip install transformers einops timm pillow
from transformers import AutoModel
import torch


class WrapModel:
    def __init__(self, model):
        self.model = model

    def get_text_features(self, dict_tensor):
        return torch.tensor(self.model.encode_text(dict_tensor, truncate_dim=512))

    def get_image_features(self, image):
        return torch.tensor(self.model.encode_image(image, truncate_dim=512))

    @property
    def device(self):
        return self.model.device


def load(
    model_name: str = "jinaai/jina-clip-v2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Initialize the model
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    def tokenizer(x):
        return x

    def processor(x):
        return x

    return WrapModel(model), processor, tokenizer


if __name__ == "__main__":
    import io
    import requests
    from PIL import Image
    import torch

    model, processor, tokenizer = load("jinaai/jina-clip-v2")
    image = Image.open(
        io.BytesIO(
            requests.get(
                "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260"
            ).content
        )
    )
    images = [image, image]
    images = processor(images)
    text = tokenizer(["犬", "猫", "象"])
    with torch.no_grad():
        image_features = model.get_image_features(images)
        text_features = model.get_text_features(text)
        print(image_features.shape, text_features.shape)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print(text_probs)
