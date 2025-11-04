# Speed Limit Detection

This project detects and reads speed limit signs using OpenCV and Tesseract OCR.

## Overview
The program automatically locates digits within an image of a speed sign and uses OCR to recognize the speed value. It supports rectangular (North American) and orange construction zone signs.

### Example:

**Input**

<p align="center">
  <img src="data/110.jpg" alt="Input speed sign" width="320">
</p>

**Output**

<p align="center">
  <img src="results/speed_limit_result_robust.png" alt="Output with MAX 110" width="320">
</p>

Result: **MAX 110**

## ⚙️ Requirements
- Python 3.9+
- OpenCV
- NumPy
- Pillow
- pytesseract
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

Install dependencies:
```bash
pip install opencv-python numpy pillow pytesseract
