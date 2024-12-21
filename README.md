# clip-eval
CLIP evaluation code.

## Installation
```bash
rye sync
```

## Usage

### Zero-shot image classification tasks.

Evaluate CLIP on imagenet-1k dataset.
```bash
python src/clip_eval/eval.py --model  openai/clip-vit-base-patch16 --dataset imagenet-1k
```

Supported models:
- line-corporation/clip-japanese-base
- rinna/japanese-cloob-vit-b-16
- rinna/japanese-clip-vit-b-16
- hf-hub:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k
- stabilityai/japanese-stable-clip-vit-l-16
- openai/clip-vit-base-patch16
- openai/clip-vit-large-patch14
- jinaai/jina-clip-v2
- google/siglip-base-patch16-256-multilingual

Supported evaluation datasets:
- `imagenet-1k`: ImageNet-1k image classification dataset
- `recruit`: Japanese-culture related image classification dataset
- `cifar100`: CIFAR-100 image classification dataset
- `cifar10`: CIFAR-10 image classification dataset
- `food101`: Food-101 image classification dataset
- `caltech101`: Caltech-101 image classification dataset

When evaluating on `Recruit` dataset, you can specify the `--subcategory` option to evaluate on a specific subcategory.
```bash
python src/clip_eval/eval.py --model  openai/clip-vit-base-patch16 --dataset recruit --subcategory "jafacility20"
```

### Zero-shot image-to-text and text-to-image retrieval tasks.

Evaluate CLIP on crossmodal3600 dataset.
```bash
python src/clip_eval/eval.py --model  openai/clip-vit-base-patch16 --dataset crossmodal3600
```


Supported evaluation datasets:
- `crossmodal3600`: Cross-modal image-text retrieval dataset


### Embedding Analysis
You can calculate the similarity matrix of the embeddings (Only first (image,text) pair per class is used).
```bash
python src/clip_eval/embedding_analysis.py --model line-corporation/clip-japanese-base --dataset recruit --batch_size 16
```
The generated image is like this:
<figure>
  <img src="./images/clip-japanese-base.png" alt="similarity_matrix" style="width:40%">
  <figcaption>Similarity matrix of embeddings</figcaption>
</figure>
This matrix is calculated by
```python
text_embeddings # (num_classes, embedding_dim)
image_embeddings # (num_classes, embedding_dim)
embeddings = torch.cat([text_embeddings, image_embeddings], dim=0) # (2*num_classes, embedding_dim)
similarity_matrix = torch.mm(embeddings, embeddings.T) # (2*num_classes, 2*num_classes)
```
So, the Left-Top submatrix is the similarity matrix of text embeddings, and the Right-Bottom submatrix is the similarity matrix of image embeddings. The Right-Top and Left-Bottom submatrices are the similarity between text and image embeddings.



You can also visualize the embeddings using t-SNE (Only first 10 classes are used).
```bash
python src/clip_eval/tsne_plot.py --model $model --dataset $dataset --batch_size 16
```
The generated image is like this:
<figure>
  <img src="./images/tsne.png" alt="similarity_matrix" style="width:40%">
  <figcaption>Similarity matrix of embeddings</figcaption>
</figure>


## Reference
- https://github.com/rinnakk/japanese-clip
- https://huggingface.co/datasets/recruit-jp/japanese-image-classification-evaluation-dataset
- https://github.com/LAION-AI/CLIP_benchmark