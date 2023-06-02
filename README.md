![](https://img.shields.io/badge/status-finished-green?style=flat-square)
![](https://img.shields.io/badge/Python-blue?style=flat-square&logo=python&color=blue&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/dependency-numpy_+_scipy-blue?style=flat-square)

# CHAR-POL

In this repository we provide the implementation of the project "Computing the characteristic polynomial" of the class Computational Linear Algebra, MATH-453  <br/>
_Authors: Anna Paulish, Fabio Matti_

## Instructions

You can reproduce our results with
```[bash]
git clone https://github.com/FMatti/CHAR-POL.git
cd CHAR-POL
python main.py
```

If dependency problems arise, you can mirror our Python environment using
```[bash]
python -m venv .venv

source .venv/bin/activate   # on Linux, macOS
.venv\Scripts\activate.bat  # in Windows command prompt (recommended)
.venv\Scripts\Activate.ps1  # in Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Our implementations require a Python version $\geq$ 3.8.

## File structure
Our implementations are located in the `src/` directory. Our results can be found in the Jupyter notebook `main.ipynb` or equivalently reproduced by running the Python script `main.py`.

```
CHAR-POL
│   README.md
|   main.ipynb             (Jupyter notebook with our results)
|   main.py                (equivalent Python script with our results)
|
└───src
|   |   methods.py      (implementations of the four methods)
|   |   helpers.py         (helper functions for plotting)
|   |   matrices.py        (definition of the example matrices)
```
