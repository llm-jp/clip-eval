from typing import Union, List
import ftfy
import html
import re
import io
import requests
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, BatchFeature


class JaCLIPForBenchmark:
    """
    enable to do model.encode_text(dict_tensor)
    """

    def __init__(self, model):
        self.model = model

    def get_text_features(self, dict_tensor):
        return self.model.get_text_features(**dict_tensor)

    def get_image_features(self, image):
        print(image)
        return self.model.get_image_features(image)


# taken from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py#L65C8-L65C8
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def tokenize(
    tokenizer,
    texts: Union[str, List[str]],
    max_seq_len: int = 77,
):
    """
    This is a function that have the original clip's code has.
    https://github.com/openai/CLIP/blob/main/clip/clip.py#L195
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [whitespace_clean(basic_clean(text)) for text in texts]

    inputs = tokenizer(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[tokenizer.bos_token_id] + ids for ids in inputs["input_ids"]]
    attention_mask = [[1] + am for am in inputs["attention_mask"]]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    return BatchFeature(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }
    )


def load(model_name: str = "stabilityai/japanese-stable-clip-vit-l-16", device="cpu"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

    def processor_wrapper(x):
        return processor(x, return_tensors="pt")["pixel_values"].squeeze(0)

    def tokenizer_wrapper(x):
        return tokenize(tokenizer, x)

    return JaCLIPForBenchmark(model), processor_wrapper, tokenizer_wrapper


if __name__ == "__main__":
    import io
    import requests
    from PIL import Image
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    model, processor, tokenizer = load("stabilityai/japanese-stable-clip-vit-l-16")
    print(model.model.num_parameters())
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
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print(text_probs)
