import os
from datasets import load_dataset
import webdataset as wds
import hashlib

# データセットを読み込む
ds = load_dataset("json", data_files="crossmodal-3600/captions.jsonl", split="train")
print(ds)

# WebDatasetのターゲットディレクトリ
output_dir = "crossmodal-3600"
os.makedirs(output_dir, exist_ok=True)

# 画像ファイルのパス
image_dir = "crossmodal-3600/images/"

# WebDataset形式のデータを構築
with wds.TarWriter(f"{output_dir}/data.tar") as writer:
    for item in ds:
        image_key = item["image/key"]
        captions = item["ja"]["caption"]

        # 画像の読み込み
        image_path = f"{image_dir}{image_key}.jpg"
        with open(image_path, "rb") as f:
            image_data = f.read()

        # 各キャプションについて、画像と一緒にWebDatasetに書き込む
        # for caption in captions:
        #     key = f"{image_key}-{hashlib.md5(caption.encode()).hexdigest()}"
        #     sample = {
        #         "__key__": key,    # キー
        #         "jpg": image_data,    # 画像のバイナリデータ
        #         "txt": caption      # キャプション
        #     }
        #     writer.write(sample)
        key = f"{image_key}-{hashlib.md5(captions[0].encode()).hexdigest()}"
        sample = {
            "__key__": key,  # キー
            "jpg": image_data,  # 画像のバイナリデータ
            "txt": captions[0],
        }
        writer.write(sample)

print(f"WebDatasetが保存されました: {output_dir}")
