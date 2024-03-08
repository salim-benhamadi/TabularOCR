TableOCR
========

.. image:: https://img.shields.io/pypi/v/tableocr.svg
    :target: https://pypi.python.org/pypi/tableocr

.. image:: https://readthedocs.org/projects/tableocr/badge/?version=latest
    :target: https://tableocr.readthedocs.io/en/latest/?version=latest
    :alt: Documentation Status

TableOCR is a powerful and versatile Python library that provides an easy-to-use Optical Character Recognition (OCR) solution for extracting tables from images and PDFs. It offers flexible output options, allowing you to export the extracted data in CSV, XLSX, or other spreadsheet formats.

* Free software: MIT license
* Documentation: https://tableocr.readthedocs.io.

Features
--------

- **Accurate Table Detection**: TableOCR uses advanced computer vision algorithms to accurately detect and extract tables from images and PDFs, even in challenging scenarios with complex layouts or low-quality scans.

- **Multiple Input Formats**: Supports a wide range of input formats, including PNG, JPG, BMP, TIFF, and PDF files.

- **Customizable Output**: Offers flexible output options, allowing you to export the extracted data in CSV, XLSX, or other spreadsheet formats of your choice.

- **Batch Processing**: Easily process multiple files in a directory or a folder structure, making it ideal for high-volume data extraction tasks.

- **Multi-language Support**: Leverages state-of-the-art OCR engines to support a wide range of languages, enabling accurate table extraction from documents in various languages.

- **Parallel Processing**: Utilizes multi-threading and parallel processing capabilities to speed up the table extraction process, significantly reducing processing times for large datasets.

- **Configurable Settings**: Provides a range of configuration options to fine-tune the table extraction process, including options for adjusting image pre-processing, OCR engine settings, and output formatting.

- **Embedded OCR Engines**: TableOCR comes bundled with several popular OCR engines, including Tesseract and LSTM-based models, ensuring high accuracy and flexibility in table extraction.

- **Seamless Integration**: Designed with a user-friendly API, TableOCR can be easily integrated into your existing Python projects, allowing for efficient table data extraction and analysis workflows.

Installation
------------

TableOCR can be installed from PyPI using pip:

```
pip install tableocr
```

Usage
-----

Here's a simple example of how to use TableOCR to extract tables from an image file:

```python
from tableocr import TableOCR

# Initialize the TableOCR instance
ocr = TableOCR()

# Path to the input image file
image_path = "path/to/image.png"

# Extract tables from the image
tables = ocr.extract_tables(image_path)

# Export the extracted tables to a CSV file
ocr.export_to_csv("output.csv", tables)
```

For more advanced usage, including batch processing, configuring OCR settings, and handling PDF files, refer to the [documentation](https://tableocr.readthedocs.io).

Contributing
------------

Contributions to TableOCR are welcome! If you encounter any issues or have ideas for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/salim-benhamadi/tableocr).

Credits
-------

TableOCR was created and is maintained by [Salim Benhamadi](https://github.com/salim-benhamadi).

License
-------

This project is licensed under the terms of the MIT license.
