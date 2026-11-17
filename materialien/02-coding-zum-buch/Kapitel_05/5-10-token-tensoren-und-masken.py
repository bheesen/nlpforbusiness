# Token-IDs, Masken und Embeddings als PyTorch-Tensoren
import torch
torch.manual_seed(7)
token_ids = torch.tensor(
    [[101, 2057, 2003, 102, 0],
     [101, 3185, 102, 0, 0]],
    dtype=torch.long,
)
print(f"Token-IDs: {token_ids.shape}, {token_ids.dtype}")
for nr, doc in enumerate(token_ids, start=1):
    print(f"Tensor-{nr}: {doc}")
maske = (token_ids != 0).to(torch.long)
embedding = torch.nn.Embedding(5000, 8)
vektoren = embedding(token_ids)
print(f"\nMaske: {maske.shape}, {maske.dtype}\n{maske}")
print(f"\nEmbeddings: {vektoren.shape}, {vektoren.dtype}")

