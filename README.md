# booky -- Two-Tower Book Recommender

I use a Two-Tower model to generate personalized book recommendations. During training, I include my own book interaction data so the model can learn a personalized User embedding that reflects my reading preferences.

After training, I extract and store all learned item embeddings in a vector index to enable fast similarity search. At inference time, the system computes a user’s embedding vector and performs an inner product with the item embeddings in the index to generate similarity scores for each book.

The model achieves approximately 31% Recall@25 on the test set across all users, and in my own manual evaluations, the recommendations are qualitatively strong and aligned with *most* of my reading preferences.

## Directory

booky/
├── src/
│   ├── data/
│   │   └── __init__.py            # load book_items.index, encoders, metadata
│   │
│   ├── model/
│   │   ├── two_towers.py         # model definitions (UserTower, ItemTower, TwoTowers)
│   │   └── __init__.py
│   │
│   ├── inference/
│   │   ├── inference.py          # main ranking + recommendation entrypoint
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── models/
│   ├── two_towers_epoch50_test6.3_train2.38.pt
│   ├── user_tower.pth            
│
├── data/
│   ├── book_items.index
│   ├── dataset_metadata.pkl
│   ├── encoders.pkl
│
├── notebooks/
│   ├── good_reads_two_towers_first_edition.ipynb
│   ├── two_towers_final_edition.ipynb
│
├── README.md
├── requirements.txt

## Setup

If you are testing the application locally, make sure git LFS is initialized correctly. 

To initialize the python project, navigate to `booky/` and type:
```
pip install -e .
```

To run the inference function with the pretrained weights on my user profile, run:
```
python -m src.inference.inference
```

