# Installation

EvoGrad requires Python 3.12+ and PyTorch 2.0+.

## From PyPI

The distribution name is `evograd-diff`; the import name is `evograd`:

```bash
pip install evograd-diff
```

## From source

```bash
git clone https://github.com/andreatangherloni/EvoGrad.git
cd EvoGrad
pip install -e .
```

Or directly:

```bash
pip install "git+https://github.com/andreatangherloni/EvoGrad.git"
```

## Verify

```python
import evograd
print(evograd.__version__)
```
