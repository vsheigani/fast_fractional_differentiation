# Fast Fractional Differentiation

Fast, numba-accelerated fractional differentiation for time series, plus a
notebook example that applies it to dollar bars. The implementation is based on
the approach described in "Advances in Financial Machine Learning" by Marcos
Lopez de Prado.

## Overview

Fractional differentiation helps make a series more stationary while retaining
memory. This repo provides:

- `fast_frac_diff()` for fast fractional differentiation
- `calc_min_d()` to estimate a minimum fractional order using the ADF test
- A reference notebook demonstrating the workflow on AAPL dollar bars

## Project Structure

- `utils/fractional_diff.py` - core implementation
- `fast_fractional_differentiation.ipynb` - example notebook
- `data/aapl_dollar_bars.h5` - sample data used in the notebook

## Installation

Python 3.12+ is required.

Using `uv` (recommended):

```bash
uv venv
source .venv/bin/activate
uv sync
```

Using `pip`:

```bash
pip install -e .
```

## Usage

Run the notebook:

```bash
jupyter notebook fast_fractional_differentiation.ipynb
```

Or use the functions directly:

```python
import pandas as pd
from utils.fractional_diff import calc_min_d, fast_frac_diff

series = pd.Series(...)
min_d = calc_min_d(series, col_name="close", thresh=1e-4)
diffed = fast_frac_diff(series, col_name="close", diff_amt=min_d, thresh=1e-4)
```

## Notes

- `calc_min_d()` uses the ADF test to find the smallest `d` that achieves
  stationarity at the 5% threshold.
- The notebook uses `data/aapl_dollar_bars.h5` and expects the `key` HDF key.