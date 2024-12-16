from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch


class WrapModel:
    def __init__(self, model):
        self.model = model

    def get_text_features(self, dict_tensor):
        return self.model.get_text_features(**dict_tensor)

    def get_image_features(self, image):
        return self.model.get_image_features(image)

    @property
    def device(self):
        return self.model.device


def load(
    model_name: str = "google/siglip-base-patch16-256-multilingual",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    image_processor = processor.image_processor

    def processor_wrapper(x):
        return image_processor(x, return_tensors="pt")["pixel_values"].squeeze(0)

    def tokenizer_wrapper(x):
        return processor(x, return_tensors="pt", padding="max_length")

    return WrapModel(model), processor_wrapper, tokenizer_wrapper


if __name__ == "__main__":
    import io
    import requests
    from PIL import Image
    import torch

    model, processor, tokenizer = load("google/siglip-base-patch16-256-multilingual")
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
        image_features = model.get_image_features(images.to(model.device))
        text_features = model.get_text_features(text.to(model.device))
        print(image_features.shape, text_features.shape)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print(text_probs)
