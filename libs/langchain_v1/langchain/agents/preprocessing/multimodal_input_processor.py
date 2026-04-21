"""Utilities for converting multimodal inputs into unified text."""

from __future__ import annotations

from typing import Any

import pytesseract
from PIL import Image
from PyPDF2 import PdfReader


class MultiModalInputProcessor:
    """Convert text, image, and PDF inputs into plain text.

    This class provides a small, extensible preprocessing layer for normalizing
    different input modalities before downstream processing.
    """

    def process_text(self, text: str) -> str:
        """Return text input as-is.

        Args:
            text: Raw text content.

        Returns:
            The input text unchanged.
        """
        return text

    def process_image(self, image_path: str) -> str:
        """Extract text from an image using OCR.

        Args:
            image_path: Path to an image file.

        Returns:
            Extracted text from the image. Returns an empty string if no text
            is found.
        """
        with Image.open(image_path) as image:
            extracted = pytesseract.image_to_string(image)
        return extracted.strip() if extracted else ""

    def process_pdf(self, pdf_path: str) -> str:
        """Extract text from all pages in a PDF file.

        Args:
            pdf_path: Path to a PDF file.

        Returns:
            Combined extracted text from all pages. Returns an empty string if
            no text is found.
        """
        reader = PdfReader(pdf_path)
        parts = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(parts).strip()

    def process(self, input_data: dict[str, Any]) -> str:
        """Process multimodal input and return normalized text.

        Expected input format:
            {
                "type": "text" | "image" | "pdf",
                "data": <string or file path>
            }

        Args:
            input_data: Input payload containing modality type and data.

        Returns:
            Extracted or normalized text. Returns an empty string when no text
            is extracted.

        Raises:
            ValueError: If input type is unsupported.
        """
        input_type = input_data.get("type")
        data = input_data.get("data")

        if input_type == "text":
            return self.process_text(str(data or ""))
        if input_type == "image":
            return self.process_image(str(data or ""))
        if input_type == "pdf":
            return self.process_pdf(str(data or ""))

        msg = f"Unsupported input type: {input_type}"
        raise ValueError(msg)
