# Channel coding course project

Course is outlined [here](http://en.didattica.unipd.it/off/2017/LM/IN/IN2371/001PD/INP6075320/N0).

# Project structure
- `simulation.py` is the executable file that actually test codes for various SNR

- `report/` contains LaTeX file and dependencies (plots, ...) needed for project presentation

- `Makefile` makes straightforward to generate relevant plots in `report/figures/` from simulation result file with `make all`

- `specs/` is the place where compressed encoding matrix for the code should be put, in order to perform simulations. A suitable Python module is embedded to ease matrix decompression for various code lengths.

- `LDPC/` is a Python module that contains relevant procedures to encode and decode, given specification. Most expensive operations are written in the much faster Cython, a Python-like language that compiles to C.

- `results/` contains all relevant CSV output of `simulation.py` execution

- `test-matrices.py` checks if matrices are symmetric in some sense, with a standard output report

# How to run project
- put relevant compressed encoding matrices in `specs/`, as specified `specs/README.md`

- install needed Python packages, for example creating a virtual environment (see Unix example, where `virtualenv` command is required)
```bash
virtualenv venv/
source venv/bin/activate
pip install -r requirements.txt
```
- run wanted simulation, after checking its parameters (in the code)
```bash
python3 simulation.py
```

- check result file is created, namely `SNRvsPe.csv.gz` in our context

# Missing `specs/` files

Private version (with licensed `specs/` matrices) is [here](https://bitbucket.org/lobisquit/channel_coding); I have done this because of parity check matrices licensing by IEEE.
