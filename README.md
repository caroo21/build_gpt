# build_gpt

A from-scratch GPT-style language model built with PyTorch. Trained on character-level text to generate new text.

## Files
- `decoder.py` — model + training loop
- `selfattention.ipynb` — self-attention walkthrough
- `data_processing.ipynb` — data exploration
- `input.txt` — training data

## Setup & Run
```bash
pip install -r requirements.txt
python decoder.py

## Based on 
Andrej Karpathy's Let's build GPT YouTube Video