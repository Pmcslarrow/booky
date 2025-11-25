# booky
Application that helps suggest books based on my own profile. I train and maintain a model offline and then utilize online methods to quickly adjust the embeddings based on my user profile. 

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
