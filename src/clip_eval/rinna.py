import torch
import japanese_clip as ja_clip
from typing import Union
from typing import Dict


class DictTensor:
    """
    enable to do `tokenizer(texts).to(device)`
    """

    def __init__(self, d: Dict[str, torch.Tensor]):
        self.d = d

    def to(self, device):
        return {k: v.to(device) for k, v in self.d.items()}


class ModelWrap:
    def __init__(self, model):
        self.model = model

    def get_image_features(self, image):
        return self.model.get_image_features(image)

    def get_text_features(self, text):
        return self.model.get_text_features(**text)

    @property
    def device(self):
        return self.model.device


def load(
    model_name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    model, preprocess = ja_clip.load(model_name, device=device)
    tokenizer = ja_clip.load_tokenizer()

    def tokenizer_wrapper(x):
        return DictTensor(ja_clip.tokenize(x, tokenizer=tokenizer, device=device))

    wrap_model = ModelWrap(model)
    return wrap_model, preprocess, tokenizer_wrapper


if __name__ == "__main__":
    import io
    import requests
    from PIL import Image
    import torch

    model, preprocess, tokenizer = load("rinna/japanese-clip-vit-b-16")
    image = Image.open(
        io.BytesIO(
            requests.get(
                "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260"
            ).content
        )
    )
    image = preprocess(image).unsqueeze(0)
    encodings = tokenizer(["犬", "猫", "象"])
    with torch.no_grad():
        image_features = model.get_image_features(image.to(model.device))
        text_features = model.get_text_features(encodings)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print(text_probs)
