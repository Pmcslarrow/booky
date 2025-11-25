import torch

def encode_user(user_id, age, encoders):
    user_id = encoders['User-ID'].get(str(user_id), 0)
    user_age = encoders['User-Age'].get(str(age), 0)
    return torch.tensor([user_id], dtype=torch.long), torch.tensor([user_age], dtype=torch.long)
