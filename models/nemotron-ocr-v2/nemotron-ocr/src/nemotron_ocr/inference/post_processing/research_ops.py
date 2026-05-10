# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def parse_relational_results(result, level="sentence"):
    """
    Parses the relational results from the OCR model.
    Supported levels:
    - "word" returns a list of words, each with a bounding box, text, and confidence.
    - "sentence" returns a list of sentences, each with a bounding box, text, and confidence.
    - "paragraph" returns a list of paragraphs, each with a bounding box, text, and confidence.

    Args:
        result (object): The result object from the OCR model.
        level (str, optional): The level to parse the results to. Defaults to "sentence".

    Returns:
        np array [N x 4 x 2]: The bounding boxes of the OCR results.
        np array [N]: The text of the OCR results
        np array [N]: The confidence scores of the OCR results
    """
    if level not in ["word", "sentence", "paragraph"]:
        raise ValueError(
            f"Invalid level: {level}. Supported levels are 'word', 'sentence', and 'paragraph'."
        )
    results = []
    for block_ids in result.relation_graph:
        sentences = []
        for sentence_ids in block_ids:
            regions = [result.regions[idx] for idx in sentence_ids]
            bboxes = [r.region.cpu().numpy() if hasattr(r.region, 'cpu') else np.asarray(r.region) for r in regions]
            texts = [region.text for region in regions]
            confs = [region.confidence for region in regions]

            if level == "word":
                for bbox, text, conf in zip(bboxes, texts, confs):
                    results.append(
                        {
                            "bbox": bbox,
                            "text": text,
                            "confidence": conf,
                        }
                    )
            else:
                bboxes = np.stack(bboxes)
                xmin = bboxes[:, :, 0].min().item()
                ymin = bboxes[:, :, 1].min().item()
                xmax = bboxes[:, :, 0].max().item()
                ymax = bboxes[:, :, 1].max().item()

                sentences.append(
                    {
                        "bbox": [
                            [xmin, ymin],
                            [xmax, ymin],
                            [xmax, ymax],
                            [xmin, ymax],
                        ],
                        "text": " ".join(texts),
                        "confidence": np.mean(confs),
                    }
                )
        if level == "word":
            pass
        elif level == "sentence":
            results += sentences
        else:
            bboxes = np.stack([s["bbox"] for s in sentences])
            texts = [s["text"] for s in sentences]
            confs = [s["confidence"] for s in sentences]
            xmin = bboxes[:, :, 0].min().item()
            ymin = bboxes[:, :, 1].min().item()
            xmax = bboxes[:, :, 0].max().item()
            ymax = bboxes[:, :, 1].max().item()
            results.append(
                {
                    "bbox": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                    "text": " ".join(texts),
                    "confidence": np.mean(confs),
                }
            )

    boxes = np.array([r["bbox"] for r in results])
    texts = np.array([r["text"] for r in results])
    confidences = np.array([r["confidence"] for r in results])
    return boxes, texts, confidences


def reorder_boxes(boxes, texts, confs, mode="center", dbscan_eps=10):
    """
    Reorders the boxes in reading order.
    If mode is "center", the boxes are reordered using bbox center.
    If mode is "top_left", the boxes are reordered using the top left corner.
    If dbscan_eps is not 0, the boxes are clustered by Y-coordinate proximity
    (equivalent to 1-D DBSCAN with min_samples=1) then sorted by cluster
    center Y, then X within each cluster.

    Args:
        boxes (np array [n x 4 x 2]): The bounding boxes of the OCR results.
        texts (np array [n]): The text of the OCR results.
        confs (np array [n]): The confidence scores of the OCR results.
        mode (str, optional): The mode to reorder the boxes. Defaults to "center".
        dbscan_eps (float, optional): The epsilon parameter for Y-clustering. Defaults to 10.

    Returns:
        np array [n x 4 x 2]: The reordered bounding boxes.
        np array [n]: The reordered texts.
        np array [n]: The reordered confidence scores.
    """
    n = len(boxes)
    if n == 0:
        return boxes, texts, confs

    boxes_arr = np.asarray(boxes).reshape(n, 4, 2)

    if mode == "center":
        xs = (boxes_arr[:, 0, 0] + boxes_arr[:, 2, 0]) / 2.0
        ys = (boxes_arr[:, 0, 1] + boxes_arr[:, 2, 1]) / 2.0
    elif mode == "top_left":
        xs = boxes_arr[:, 0, 0].copy()
        ys = boxes_arr[:, 0, 1].copy()
    else:
        xs = (boxes_arr[:, 0, 0] + boxes_arr[:, 2, 0]) / 2.0
        ys = (boxes_arr[:, 0, 1] + boxes_arr[:, 2, 1]) / 2.0

    if dbscan_eps and n > 1:
        sort_idx = np.argsort(ys)
        sorted_ys = ys[sort_idx]

        sorted_labels = np.zeros(n, dtype=np.int32)
        cluster_id = 0
        for i in range(1, n):
            if sorted_ys[i] - sorted_ys[i - 1] > dbscan_eps:
                cluster_id += 1
            sorted_labels[i] = cluster_id

        labels = np.empty(n, dtype=np.int32)
        labels[sort_idx] = sorted_labels

        n_clusters = cluster_id + 1
        centers = np.zeros(n_clusters, dtype=np.float64)
        counts = np.zeros(n_clusters, dtype=np.float64)
        np.add.at(centers, labels, ys)
        np.add.at(counts, labels, 1)
        centers /= np.maximum(counts, 1)

        cluster_y = centers[labels].astype(np.int32)
        order = np.lexsort((xs, cluster_y))
    else:
        binned_y = np.round((ys - ys.min()) / 5.0, 0)
        order = np.lexsort((xs, binned_y))

    boxes = [boxes[i] for i in order]
    texts = [texts[i] for i in order]
    confs = [confs[i] for i in order]
    return boxes, texts, confs
