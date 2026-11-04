# Predictive Modelling with Applications

Open [the course reader](index.html). This independent study edition follows
Prof. Sayantee Jana's IIT Hyderabad NPTEL course: 39 teaching recordings,
18 chapters,108 worked solutions and 18 executable Python laboratories.

All inventoried R methods are implemented in Python. Missing original inputs
and source-option uncertainties are explained in the
[conversion ledger](python-conversions.html). Scientific, source, execution and
browser evidence have separate scopes; see [Sources and verification](about.html).

## Run and rebuild

Install `requirements.txt` in a Python environment. The verified versions are
in `requirements-lock.txt`. Run `python code_lab.py 1` through
`python code_lab.py 18`, or execute the corresponding `labs/chNN.py` directly.

```text
python tools/build_book.py
python tools/check_book.py
python tools/check_reader.py
python tools/check_export.py
```

The reader includes local mathematics rendering and search. External source
links need the internet. The portable package is generated at
`dist/predictive-modelling-notes.zip`; it excludes raw media and transcripts.
No R installation is required to run the final laboratories.
