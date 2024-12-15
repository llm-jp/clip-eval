from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel


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


def load(model_name: str, device: str = "cuda"):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = processor.tokenizer

    def tokenizer_wrapper(x):
        return tokenizer(x, return_tensors="pt", padding=True, truncation=True)

    def processor_wrapper(x):
        return processor(images=x, return_tensors="pt")["pixel_values"].squeeze(0)

    return WrapModel(model), processor_wrapper, tokenizer_wrapper


if __name__ == "__main__":
    import io
    import requests
    from PIL import Image
    import torch

    model, processor, tokenizer = load("openai/clip-vit-base-patch16")
    image = Image.open(
        io.BytesIO(
            requests.get(
                "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260"
            ).content
        )
    )
    images = [image, image]
    images = processor(images)
    # text = tokenizer(["doc", "cat", "elephant"])
    text = tokenizer(["犬", "猫", "象"])
    with torch.no_grad():
        image_features = model.get_image_features(images.to(model.device))
        text_features = model.get_text_features(text.to(model.device))
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print(text_probs)
