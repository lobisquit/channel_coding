Put here all matrices and code rate specifications

Matrices should be available in the compressed format specified in our reference, i.e. "IEEE Std 802.16e-2005", section 8.4.9.2.5.1.

For example then, parity check matrix for rate 1/2 should be stored in this folder with the name `H-rate-12.csv`, removing then the `/`.

The user should provide also the specifications for the wanted code rate, mainly codeword length, in files in the form `block-size-rate-12.csv` for code rate 1/2.

Those should be CSV files as well with following columns.

- n (bits)
- n (bytes)
- z factor
- k (bytes)
- Subchannels QPSK
- Subchannels 16QAM
- Subchannels 64QAM
