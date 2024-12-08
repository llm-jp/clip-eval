import open_clip
import torch



def load(model_name: str = "ViT-B-32-quickgelu", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model_name)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer

