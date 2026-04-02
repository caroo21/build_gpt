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
```

## Scores
| | Andrej Karpathy | My Results |
|---|---|---|
| Train Loss |1.0884 | ... |
| Val Loss | 1.4778 | ... |

## Improvements over base implementation

Starting from Andrej Karpathy's base implementation, the following changes were made:

- **Gradient Clipping** — prevents exploding gradients during training
- **Learning Rate Scheduler** — cosine annealing decay for better convergence
- **GeLU Activation** — smoother gradient flow compared to ReLU
- **Flash Attention** — faster and memory-efficient attention (PyTorch 2.0+)
- **Weight Tying** — shared weights between token embedding and output layer

## Based on 
Andrej Karpathy's Let's build GPT YouTube Video