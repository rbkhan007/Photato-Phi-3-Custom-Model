#!/usr/bin/env python3
"""
Multi-modal Support (Image Understanding) for Local LLMs.

Features:
- Image loading and preprocessing
- Vision-language integration
- Image description generation
- OCR capabilities

Usage:
    from capabilities.multimodal import MultiModalProcessor

    processor = MultiModalProcessor()
    description = processor.describe_image("path/to/image.jpg")
"""

import base64
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImageInfo:
    """Image information."""
    path: str
    format: str
    size: tuple[int, int]
    mode: str
    file_size: int
    metadata: dict = field(default_factory=dict)


class ImageLoader:
    """Image loading utilities."""

    @staticmethod
    def load_image(path: str) -> Optional[bytes]:
        """
        Load image from path.

        Args:
            path: Image path

        Returns:
            Image bytes or None
        """
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    @staticmethod
    def get_image_info(path: str) -> Optional[ImageInfo]:
        """
        Get image information.

        Args:
            path: Image path

        Returns:
            ImageInfo or None
        """
        try:
            file_path = Path(path)
            file_size = file_path.stat().st_size

            # Get format from extension
            format_map = {
                ".jpg": "JPEG",
                ".jpeg": "JPEG",
                ".png": "PNG",
                ".gif": "GIF",
                ".webp": "WEBP",
                ".bmp": "BMP",
            }
            fmt = format_map.get(file_path.suffix.lower(), "UNKNOWN")

            return ImageInfo(
                path=str(file_path.absolute()),
                format=fmt,
                size=(0, 0),  # Would need PIL to get actual size
                mode="unknown",
                file_size=file_size,
            )
        except Exception:
            return None

    @staticmethod
    def to_base64(path: str) -> Optional[str]:
        """
        Convert image to base64.

        Args:
            path: Image path

        Returns:
            Base64 string or None
        """
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None


class MultiModalProcessor:
    """
    Multi-modal processing for local LLMs.

    Features:
    - Image loading
    - Image description
    - OCR integration
    - Vision-language prompts
    """

    def __init__(self, model_path: str = ""):
        """
        Initialize processor.

        Args:
            model_path: Path to vision model
        """
        self.model_path = model_path
        self.loader = ImageLoader()

    def describe_image(
        self,
        image_path: str,
        detail_level: str = "medium",
    ) -> str:
        """
        Generate image description.

        Args:
            image_path: Path to image
            detail_level: Detail level (low, medium, high)

        Returns:
            Image description
        """
        # Load image info
        info = self.loader.get_image_info(image_path)
        if not info:
            return "Unable to load image."

        # Create description prompt
        prompt = self._create_description_prompt(info, detail_level)

        # Generate description (placeholder - use actual model)
        return f"Image description for {info.format} image ({info.file_size} bytes)."

    def _create_description_prompt(self, info: ImageInfo, detail_level: str) -> str:
        """Create prompt for image description."""
        detail_instructions = {
            "low": "Provide a brief one-sentence description.",
            "medium": "Describe the main elements and their relationships.",
            "high": "Provide a detailed description including colors, objects, and composition.",
        }

        return f"""Describe this {info.format} image.

{detail_instructions.get(detail_level, detail_instructions['medium'])}

Image properties:
- Format: {info.format}
- Size: {info.size[0]}x{info.size[1]}
- File size: {info.file_size} bytes

Description:"""

    def create_vision_prompt(
        self,
        image_path: str,
        question: str,
    ) -> str:
        """
        Create prompt for vision question answering.

        Args:
            image_path: Path to image
            question: Question about image

        Returns:
            Vision prompt
        """
        info = self.loader.get_image_info(image_path)
        format_info = f"{info.format}, {info.file_size} bytes" if info else "unknown"

        return f"""Look at this image ({format_info}) and answer the question.

Question: {question}

Answer based on the image content:"""

    def extract_text_prompt(self, image_path: str) -> str:
        """
        Create prompt for OCR.

        Args:
            image_path: Path to image

        Returns:
            OCR prompt
        """
        info = self.loader.get_image_info(image_path)
        format_info = f"{info.format}" if info else "image"

        return f"""Extract all visible text from this {format_info} image.

List the text exactly as it appears, preserving formatting where possible.

Text:"""

    def analyze_image(
        self,
        image_path: str,
        analysis_type: str = "general",
    ) -> str:
        """
        Analyze image.

        Args:
            image_path: Path to image
            analysis_type: Type of analysis (general, objects, colors, text)

        Returns:
            Analysis result
        """
        info = self.loader.get_image_info(image_path)
        if not info:
            return "Unable to load image."

        analysis_prompts = {
            "general": "Provide a comprehensive analysis of this image.",
            "objects": "List and describe all objects visible in this image.",
            "colors": "Describe the color palette and dominant colors.",
            "text": "Extract and list all visible text.",
        }

        prompt = f"""{analysis_prompts.get(analysis_type, analysis_prompts['general'])}

Image: {info.format} ({info.file_size} bytes)

Analysis:"""

        return f"Analysis of {info.format} image..."

    def batch_describe(self, image_paths: list[str]) -> dict[str, str]:
        """
        Describe multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            Dict of path -> description
        """
        results = {}
        for path in image_paths:
            results[path] = self.describe_image(path)
        return results


class VisionPromptBuilder:
    """Build prompts for vision tasks."""

    @staticmethod
    def describe(image_path: str, style: str = "detailed") -> str:
        """Build description prompt."""
        styles = {
            "brief": "In one sentence, describe this image.",
            "detailed": "Provide a detailed description of this image, including objects, colors, and composition.",
            "creative": "Describe this image in a creative and evocative way.",
        }
        return styles.get(style, styles["detailed"])

    @staticmethod
    def question(image_path: str, question: str) -> str:
        """Build question-answering prompt."""
        return f"Look at this image and answer: {question}"

    @staticmethod
    def compare(path1: str, path2: str) -> str:
        """Build comparison prompt."""
        return "Compare these two images. What are the similarities and differences?"

    @staticmethod
    def count(image_path: str, object_name: str) -> str:
        """Build counting prompt."""
        return f"How many {object_name} are visible in this image?"


def main(argv=None):
    """CLI for multi-modal (image understanding) support."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="capabilities.multimodal",
        description="Multi-modal (image understanding) support for local LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_image(p):
        p.add_argument("--image", required=True, help="Path to image file")

    p_desc = sub.add_parser("describe", help="Describe an image")
    add_image(p_desc)
    p_desc.add_argument("--detail-level", choices=["low", "medium", "high"], default="medium")

    p_an = sub.add_parser("analyze", help="Analyze an image")
    add_image(p_an)
    p_an.add_argument("--type", choices=["general", "objects", "colors", "text"], default="general")

    p_vp = sub.add_parser("vision-prompt", help="Build a vision QA prompt")
    add_image(p_vp)
    p_vp.add_argument("--question", required=True)

    p_op = sub.add_parser("ocr-prompt", help="Build an OCR prompt")
    add_image(p_op)

    p_info = sub.add_parser("info", help="Show image info")
    add_image(p_info)

    p_b64 = sub.add_parser("base64", help="Convert image to base64")
    add_image(p_b64)

    p_build = sub.add_parser("build", help="Build a prompt via VisionPromptBuilder")
    p_build.add_argument("--kind", choices=["describe", "question", "compare", "count"], default="describe")
    p_build.add_argument("--image", help="Image path")
    p_build.add_argument("--question", help="Question (for question)")
    p_build.add_argument("--image2", help="Second image path (for compare)")
    p_build.add_argument("--object", help="Object name (for count)")
    p_build.add_argument("--style", default="detailed", choices=["brief", "detailed", "creative"])

    args = parser.parse_args(argv)

    try:
        proc = MultiModalProcessor()
        if args.command == "describe":
            desc = proc.describe_image(args.image, detail_level=args.detail_level)
            info = ImageLoader.get_image_info(args.image)
            print(json.dumps({
                "image": args.image,
                "description": desc,
                "info": info.__dict__ if info else None,
            }, indent=2, default=str))
        elif args.command == "analyze":
            res = proc.analyze_image(args.image, analysis_type=args.type)
            print(json.dumps({
                "image": args.image,
                "type": args.type,
                "analysis": res,
            }, indent=2, default=str))
        elif args.command == "vision-prompt":
            prompt = proc.create_vision_prompt(args.image, args.question)
            print(json.dumps({"prompt": prompt}, indent=2, default=str))
        elif args.command == "ocr-prompt":
            prompt = proc.extract_text_prompt(args.image)
            print(json.dumps({"prompt": prompt}, indent=2, default=str))
        elif args.command == "info":
            info = ImageLoader.get_image_info(args.image)
            print(json.dumps(
                info.__dict__ if info else {"error": "unable to load image"},
                indent=2, default=str))
        elif args.command == "base64":
            b64 = ImageLoader.to_base64(args.image)
            if b64 is None:
                print(json.dumps({"error": "unable to load image"}, indent=2))
            else:
                print(json.dumps({"path": args.image, "base64": b64}, indent=2))
        elif args.command == "build":
            if args.kind == "describe":
                out = VisionPromptBuilder.describe(args.image or "", style=args.style)
            elif args.kind == "question":
                out = VisionPromptBuilder.question(args.image or "", args.question or "")
            elif args.kind == "compare":
                out = VisionPromptBuilder.compare(args.image or "", args.image2 or "")
            else:
                out = VisionPromptBuilder.count(args.image or "", args.object or "")
            print(json.dumps({"prompt": out}, indent=2, default=str))
        return 0
    except (OSError, ValueError) as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
