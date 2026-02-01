import torch


def collate_batch(batch, task_vocab, device=None):
    x = torch.stack([b["x"] for b in batch], dim=0)
    q0 = torch.stack([b["q0"] for b in batch], dim=0)
    emotion = torch.stack([b["emotion"] for b in batch], dim=0)
    task_ids = torch.tensor([task_vocab[b["task_type"]] for b in batch], dtype=torch.long)

    if device is not None:
        x = x.to(device)
        q0 = q0.to(device)
        emotion = emotion.to(device)
        task_ids = task_ids.to(device)

    return {
        "x": x,
        "emotion": emotion,
        "task_id": task_ids,
        "q0": q0,
        "task_type": [b["task_type"] for b in batch],
        "context_json": [b["context_json"] for b in batch],
    }
