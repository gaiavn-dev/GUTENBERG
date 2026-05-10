#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2


def main(image_path, merge_level, no_visualize, model_dir, lang,
         detector_only, skip_relational):
    kwargs = {}
    if model_dir is not None:
        kwargs["model_dir"] = model_dir
    else:
        kwargs["lang"] = lang
    if detector_only:
        kwargs["detector_only"] = True
    if skip_relational:
        kwargs["skip_relational"] = True

    ocr = NemotronOCRV2(**kwargs)

    predictions = ocr(image_path, merge_level=merge_level)

    print(f"Found {len(predictions)} text regions.")
    for pred in predictions:
        if "text" in pred:
            print(
                f"  - Text: '{pred['text']}', "
                f"Confidence: {pred['confidence']:.2f}, "
                f"Bbox: [left={pred['left']:.4f}, upper={pred['upper']:.4f}, "
                f"right={pred['right']:.4f}, lower={pred['lower']:.4f}]"
            )
        else:
            print(
                f"  - Confidence: {pred['confidence']:.2f}, "
                f"Bbox: [left={pred['left']:.4f}, upper={pred['upper']:.4f}, "
                f"right={pred['right']:.4f}, lower={pred['lower']:.4f}]"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR inference on an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "--merge-level",
        type=str,
        choices=["word", "sentence", "paragraph"],
        default="paragraph",
        help="Merge level for OCR output (default: paragraph).",
    )
    parser.add_argument("--no-visualize", action="store_true", help="(unused, kept for compat)")
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Local checkpoint directory. If omitted, downloads from Hugging Face.",
    )
    parser.add_argument(
        "--lang", type=str, choices=["en", "multi", "v1"], default=None,
        help="Hub checkpoint: en, multi (default), or v1.",
    )
    parser.add_argument(
        "--detector-only", action="store_true",
        help="Run detector only — returns boxes without text.",
    )
    parser.add_argument(
        "--skip-relational", action="store_true",
        help="Skip relational model — returns per-word text without reading order.",
    )
    args = parser.parse_args()

    main(
        args.image_path,
        merge_level=args.merge_level,
        no_visualize=args.no_visualize,
        model_dir=args.model_dir,
        lang=args.lang,
        detector_only=args.detector_only,
        skip_relational=args.skip_relational,
    )
