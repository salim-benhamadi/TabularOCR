TabularOCR
========

.. image:: https://img.shields.io/pypi/v/tabularOCR.svg
    :target: https://pypi.python.org/pypi/tabularOCR

.. image:: https://readthedocs.org/projects/tabularocr/badge/?version=latest
    :target: https://tabularocr.readthedocs.io/en/latest/?version=latest
    :alt: Documentation Status

TabularOCR is a powerful and versatile Python library that provides an easy-to-use Optical Character Recognition (OCR) solution for extracting tables from images and PDFs. It offers flexible output options, allowing you to export the extracted data in CSV, XLSX, or other spreadsheet formats.

* Free software: MIT license
* Documentation: https://tabularocr.readthedocs.io.

Features
--------

- **Accurate Table Detection**: TabularOCR uses advanced computer vision algorithms to accurately detect and extract tables from images and PDFs, even in challenging scenarios with complex layouts or low-quality scans. It employs techniques such as edge detection, connected component analysis, and deep learning-based object detection to locate and isolate tables within the input document.

- **Multiple Input Formats**: Supports a wide range of input formats, including PNG, JPG, BMP, TIFF, and PDF files, allowing for flexibility in processing various types of document sources.

- **Customizable Output**: Offers flexible output options, allowing you to export the extracted data in CSV, XLSX, or other spreadsheet formats of your choice, ensuring seamless integration with your existing data processing workflows.

- **Batch Processing**: Easily process multiple files in a directory or a folder structure, making it ideal for high-volume data extraction tasks, such as digitizing large archives or processing scanned documents at scale.

- **Multi-language Support**: Leverages state-of-the-art OCR engines to support a wide range of languages, enabling accurate table extraction from documents in various languages, including English, Spanish, French, German, Chinese, Arabic, and many more.

- **Parallel Processing**: Utilizes multi-threading and parallel processing capabilities to speed up the table extraction process, significantly reducing processing times for large datasets or complex documents.

- **Configurable Settings**: Provides a range of configuration options to fine-tune the table extraction process, including options for adjusting image pre-processing (e.g., deskewing, denoising, and binarization), OCR engine settings (e.g., language packs, character whitelists), and output formatting (e.g., column delimiters, date formats).

- **Embedded OCR Engines**: TabularOCR comes bundled with several popular OCR engines, including Tesseract and LSTM-based models, ensuring high accuracy and flexibility in table extraction. Additional OCR engines can be easily integrated, thanks to the modular design of the library.

- **Seamless Integration**: Designed with a user-friendly API, TabularOCR can be easily integrated into your existing Python projects, allowing for efficient table data extraction and analysis workflows, enabling applications in areas such as data mining, research, and business intelligence.

Installation
------------

TabularOCR can be installed from PyPI using pip:

.. code-block:: bash

    pip install TabularOCR

Usage
-----

Here's a simple example of how to use TabularOCR to extract tables from an image file:

.. code-block:: python

    from tabularocr import TabularOCR

    # Initialize the TabularOCR instance
    ocr = TabularOCR()

    # Path to the input image or PDF file
    image_path = "path/to/image.png"

    # Extract tables from the image
    tables = ocr.extract(image_path)

    # Export the extracted tables to a CSV file
    ocr.to_csv("output.csv", tables)

Usage Examples
-----------------------

1. **Batch Processing**:

TabularOCR supports batch processing of multiple files in a directory or folder structure. Here's an example:

.. code-block:: python

    from tabularocr import TabularOCR
    import os

    # Initialize the TabularOCR instance
    ocr = TabularOCR()

    # Directory containing input files
    input_dir = "path/to/input/directory"

    # Iterate over files in the directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        tables = ocr.extract(file_path)

        # Export tables to individual CSV files
        output_file = f"output_{filename}.csv"
        ocr.to_csv(output_file, tables)

2. **Configuring OCR Settings**:

You can fine-tune the OCR engine settings to optimize performance for specific document types or languages:

.. code-block:: python

    from tabularocr import TabularOCR, OCRSettings

    # Initialize the TabularOCR instance
    ocr = TabularOCR()

    # Configure OCR settings
    settings = OCRSettings(language="fra", whitelist="0123456789")
    ocr.set_ocr_settings(settings)

    # Extract tables using the configured settings
    image_path = "path/to/image.png"
    tables = ocr.extract(image_path)

3. **Customizing Output Formatting**:

TabularOCR allows you to customize the output format by specifying column delimiters, date formats, and other formatting options:

.. code-block:: python

    from tabularocr import TabularOCR, OutputSettings

    # Initialize the TabularOCR instance
    ocr = TabularOCR()

    # Configure output settings
    output_settings = OutputSettings(delimiter="|", date_format="%Y-%m-%d")
    ocr.set_output_settings(output_settings)

    # Extract tables and export to CSV with custom settings
    image_path = "path/to/image.png"
    tables = ocr.extract(image_path)
    ocr.to_csv("output.csv", tables)

For more advanced usage, such as handling PDF files, table structure analysis, and table merging, refer to the `documentation <https://tabularocr.readthedocs.io>`_.

Contributing
------------

Contributions to TabularOCR are welcome! If you encounter any issues or have ideas for improvements, please open an issue or submit a pull request on the `GitHub repository <https://github.com/salim-benhamadi/tabularocr>`_.

Credits
-------

TabularOCR was created and is maintained by `Salim Benhamadi <https://github.com/salim-benhamadi>`_.

License
-------

This project is licensed under the terms of the MIT license.
