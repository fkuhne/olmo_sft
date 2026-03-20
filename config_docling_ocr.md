# Docling OCR Engine Configuration Guide

This document explains the trade-offs of the default OCR engine and provides instructions on how to switch to more powerful alternatives, including an automated selection script for your environment.

---

## 1. Understanding the Default: RapidOCR

**RapidOCR** is the default engine in Docling because it is designed for **high-throughput production environments**. It is based on the PaddleOCR framework and uses the ONNX runtime for cross-platform compatibility.

### Advantages
* **Performance:** It is significantly faster than Tesseract and EasyOCR on CPU-bound workloads.
* **Resource Efficiency:** It has a smaller memory footprint and lower latency, making it ideal for scaling across many document pages.
* **Dependency-Lite:** It runs locally without requiring complex system-level drivers like those often needed for Tesseract.

### Trade-offs
* **Accuracy Jitter:** While fast, it may struggle with highly stylized fonts, very low-resolution scans, or complex mathematical formulas compared to modern deep-learning models.
* **Multilingual Support:** While it supports multiple languages, its recognition quality for non-Latin scripts can sometimes be less robust than specialized engines.


## 2. Switching to EasyOCR
**Best for:** General-purpose OCR where higher accuracy than RapidOCR is needed, especially if you have a **GPU** available.

### Prerequisites
Install the EasyOCR extra:
```bash
pip install docling[easyocr]
ImplementationPythonfrom docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# 1. Configure the pipeline to use EasyOCR
pipeline_options = PdfPipelineOptions()
pipeline_options.ocr_options = EasyOcrOptions(
    lang=["en", "es"],  # Set your target languages
    use_gpu=True        # Highly recommended for EasyOCR
)

# 2. Initialize the converter
converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# 3. Convert
result = converter.convert("path/to/your/document.pdf")
print(result.document.export_to_markdown())
```
---

## 3. Switching to VLM (Vision-Language Models)

**Best for:** The highest possible extraction quality. VLMs like Granite-Docling or Qwen2.5-VL "understand" the document layout and context rather than just reading characters.

### Prerequisites

Install the VLM-specific extensions:
```Bash
pip install docling[vlm]
```

### Implementation

```Python
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# 1. Define VLM options (Note: This will download weights on first run)
pipeline_options = VlmPipelineOptions()
pipeline_options.vlm_model = "ibm-granite/granite-3.0-8b-instruct" 

# 2. Initialize converter with the VLM pipeline
converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# 3. Convert
result = converter.convert("messy_complex_document.pdf")
print(result.document.export_to_markdown())
```

## 4. Automated Engine Selection Script

For advanced projects like doctune, you can use this wrapper to automatically select the most powerful engine available based on your system's hardware.
```Python
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    EasyOcrOptions, 
    VlmPipelineOptions
)

def get_optimized_converter():
    """
    Selects the best available OCR engine:
    1. VLM (if high-end GPU is detected)
    2. EasyOCR (if standard GPU is detected)
    3. RapidOCR (Default fallback for CPU)
    """
    
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # High VRAM: Use VLM for maximum intelligence
        if vram_gb > 12:
            print("🚀 Hardware match: Using VLM (Granite)")
            options = VlmPipelineOptions()
            options.vlm_model = "ibm-granite/granite-3.0-8b-instruct"
        
        # Standard GPU: Use EasyOCR for high accuracy
        else:
            print("🏎️ Hardware match: Using EasyOCR")
            options = PdfPipelineOptions()
            options.ocr_options = EasyOcrOptions(use_gpu=True)
            
    else:
        print("⚡ Hardware match: Using RapidOCR (CPU Optimized)")
        options = PdfPipelineOptions() # Defaults to RapidOCR
        
    return DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=options)}
    )

# Usage
converter = get_optimized_converter()
```


## Engine Comparison Summary

| Feature | RapidOCR (Default) | EasyOCR | VLM (e.g., Granite) |
| :--- | :--- | :--- | :--- |
| **Primary Strength** | Speed & Efficiency | Multilingual Accuracy | Contextual Understanding |
| **Hardware** | CPU Optimized | GPU Recommended | GPU Required (High VRAM) |
| **Accuracy** | Good | High | State-of-the-Art |
| **Typical Use Case** | Bulk processing / Speed | Scanned PDFs | Hand-drawn tables / Complex layout |
