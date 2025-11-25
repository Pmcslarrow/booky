from src.data.loaders import load_model_and_artifacts
from src.inference.encode_user import encode_user
import torch.nn.functional as F
import torch
import os 

def recommend_books(model, encoders, index, dataset, user_id, age, top_k=100):
    uid, uage = encode_user(user_id, age, encoders)

    with torch.no_grad():
        user_emb = model(uid, uage)
        user_emb = F.normalize(user_emb, p=2, dim=1)
        user_emb = user_emb.numpy()

    distances, indices = index.search(user_emb, top_k)

    results = []
    seen = set()

    for idx, dist in zip(indices[0], distances[0]):
        row = dataset.iloc[idx]
        title = row['Book-Title']
        author = row['Book-Author']
        isbn = row['ISBN']

        if title in seen or isbn in seen:
            continue

        seen.add(title)
        seen.add(isbn)

        results.append({
            "title": title,
            "author": author,
            "isbn": isbn,
            "score": float(dist)
        })

    return results


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    model, encoders, index, dataset = load_model_and_artifacts()
    print("Loaded everything successfully!", end="\n\n")
    recs = recommend_books(model, encoders, index, dataset, "1234567890", "18-25", top_k=1000)
    
    for i in range(100):
        title = recs[i]['title']
        author = recs[i]['author']
        distance = recs[i]['score']
        print(f"Title: {title}, Author: {author}, Score: {distance:.4f}")
