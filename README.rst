========
TableOCR
========


.. image:: https://img.shields.io/pypi/v/tableocr.svg
        :target: https://pypi.python.org/pypi/tableocr

.. image:: https://img.shields.io/travis/salim-benhamadi/tableocr.svg
        :target: https://travis-ci.com/salim-benhamadi/tableocr

.. image:: https://readthedocs.org/projects/tableocr/badge/?version=latest
        :target: https://tableocr.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




TableOCR is a Python library that provides an easy-to-use Optical Character Recognition (OCR) solution for extracting tables from images and PDFs. It offers flexible output options, allowing you to export the extracted data in CSV, XLSX, or other spreadsheet formats.


* Free software: MIT license
* Documentation: https://tableocr.readthedocs.io.


Features
--------

* TODO

Credits
-------


1. **Create a virtual environment**

   - Use the `venv` module (included in Python 3) to create a new virtual environment:

     ```bash
     python3 -m venv myenv
     ```

     This will create a new directory `myenv` containing the virtual environment files.

2. **Activate the virtual environment**

   - On Windows:

     ```bash
     myenv\Scripts\activate
     ```

   - On Unix or macOS:

     ```bash
     source myenv/bin/activate
     ```

   Your shell prompt will now have a prefix indicating the active virtual environment.

3. **Install packages**

   - Install packages using `pip`:

     ```bash
     pip install package_name
     ```

   - For example, to install `paddleocr` and `layoutparser`:

     ```bash
     pip install paddleocr layoutparser
     ```

4. **Create a requirements file**

   - Generate a `requirements.txt` file containing the installed packages and their versions:

     ```bash
     pip freeze > requirements.txt
     ```

   - This will create a `requirements.txt` file in the current directory, listing all installed packages and their versions.

5. **Install packages from the requirements file**

   - To install all packages from the `requirements.txt` file in a new environment:

     ```bash
     pip install -r requirements.txt
     ```

     This will install all packages listed in the `requirements.txt` file.

6. **Deactivate the virtual environment (when done)**

   - To deactivate the virtual environment:

     ```bash
     deactivate
     ```
