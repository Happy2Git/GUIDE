# GUIDE
Implementation of the GUIDE framework for Inductive Graph Unlearning, just accepted at USENIX Security'23. For more information on GUIDE, please refer to the following paper link: https://arxiv.org/abs/2304.03093.

## Installation

```
git clone https://github.com/Happy2Git/GUIDE.git  
cd GUIDE  
mamba env create -n GUIDE -f environment.yml  
mkdir data checkpoints
```

## Usage
See 'inductive_BTC.ipynb' and inductive_BTC_abalation.ipynb'

# Citation
If you find GUIDE useful for your research, please consider citing this paper:
```
@inproceedings{DBLP:conf/uss/WangH023,
  author       = {Cheng{-}Long Wang and
                  Mengdi Huai and
                  Di Wang},
  editor       = {Joseph A. Calandrino and
                  Carmela Troncoso},
  title        = {Inductive Graph Unlearning},
  booktitle    = {32nd {USENIX} Security Symposium, {USENIX} Security 2023, Anaheim,
                  CA, USA, August 9-11, 2023},
  pages        = {3205--3222},
  publisher    = {{USENIX} Association},
  year         = {2023},
  url          = {https://www.usenix.org/conference/usenixsecurity23/presentation/wang-cheng-long},
  timestamp    = {Sun, 04 Aug 2024 19:38:59 +0200},
  biburl       = {https://dblp.org/rec/conf/uss/WangH023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
## Contact
-   Cheng-Long Wang - X.Y@Z where X = chenglong, Y = wang and Z = kaust.edu.sa
