# coding=utf-8
# Copyright 2022 rinna Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tqdm.auto import tqdm
import numpy as np
import torch
from logging import getLogger

logger = getLogger(__name__)
logger.setLevel("INFO")


def accuracy(output, target, topk=(1,)):
    output = torch.from_numpy(np.asarray(output))
    target = torch.from_numpy(np.asarray(target))
    pred = output.topk(max(topk), dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


class ClassificationCallback:
    def __init__(
        self,
        classes,
        templates,
        dataloader,
    ):
        self.classes = classes
        self.templates = templates
        self.dataloader = dataloader

    def zeroshot_classifier(self, model, tokenizer, classnames, templates):
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            tokenized_texts = tokenizer(texts)
            try:
                tokenized_texts = tokenized_texts.to(model.device)
            except AttributeError:
                tokenized_texts = tokenized_texts
            class_embeddings = (
                model.get_text_features(tokenized_texts).detach().cpu().numpy()
            )
            class_embeddings = class_embeddings / np.linalg.norm(
                class_embeddings, axis=-1, keepdims=True
            )
            class_embedding = np.mean(class_embeddings, axis=0)
            class_embedding /= np.linalg.norm(class_embedding, axis=-1)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = np.stack(zeroshot_weights, axis=1)
        return zeroshot_weights

    def zeroshot(self, model, tokenizer) -> dict:
        logger.info("Zeroshot Classification...")
        zeroshot_weights = self.zeroshot_classifier(
            model, tokenizer, self.classes, self.templates
        )
        top_ns = [1, 5, 10, 100]
        # if target classes is < 100, top_ns is adjusted
        top_ns = [min(top_n, len(self.classes)) for top_n in top_ns]
        acc_counters = [0.0 for _ in top_ns]
        n = 0.0

        for i, (images, target) in enumerate(tqdm(self.dataloader)):
            target = target.numpy()
            # predict
            try:
                images = images.to(model.device)
            except AttributeError:
                images = images
            image_features = model.get_image_features(images).detach().cpu().numpy()
            image_features = image_features / np.linalg.norm(
                image_features, axis=-1, keepdims=True
            )
            logits = 100.0 * image_features @ zeroshot_weights
            # measure accuracy
            accs = accuracy(logits, target, topk=top_ns)
            for j in range(len(top_ns)):
                acc_counters[j] += accs[j]
            n += len(images)

        tops = {
            f"top{top_ns[i]}": acc_counters[i] / n * 100 for i in range(len(top_ns))
        }

        return tops
