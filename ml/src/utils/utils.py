import torch

def bucketize_age(age):
    if age < 18:
        return "<18"
    elif age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 50:
        return "36-50"
    else:
        return"50+"

def encode_user(user_id, age, encoders):
    user_id = encoders['User-ID'].get(str(user_id), 0)
    user_age = encoders['User-Age'].get(str(age), 0)
    return torch.tensor([user_id], dtype=torch.long), torch.tensor([user_age], dtype=torch.long)
