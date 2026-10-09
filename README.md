# booky

## Summary

I use a Two-Tower model to generate personalized book recommendations. During training, I include my own book interaction data so the model can learn a personalized User embedding that reflects my reading preferences.

After training, I extract and store all learned item embeddings in a vector index to enable fast similarity search. At inference time, the system computes a user’s embedding vector and performs an inner product with the item embeddings in the index to generate similarity scores for each book.

The model achieves approximately 31% Recall@25 on the test set across all users, and in my own manual evaluations, the recommendations are qualitatively strong and aligned with *most* of my reading preferences.

## In Development

Currently developing a full-stack application for **Booky** — a platform for people to discover new books. The old code for this project lives in `ml/notebooks/two_towers_final_edition.ipynb`. 

- CI/CD:
    - Continuous integration (CI) with GitHub Actions (linting and tests where applicable)
    - Contunuous deployment (CD) is one of the last steps that I will focus on. But I hope to use this project as a way to learn how to automatically build and deploy my work.

- UI/Backend:
    - The `homepage` will display personalized recommendations powered by a two-towers model serving as the online recommender system. It will also surface other recommendation types such as trending books, bestsellers, and books filtered by favorite genre(s).
    - The `admin page` will allow users to track details about books they have read along with their ratings. This data will inform future recommendation systems and be used for offline model training to keep recommendations current.

- Architecture / Machine Learning:
    - *Scalability and latency are the two primary areas of focus*.
    - To improve scalability, I plan to implement a multi-stage recommender pipeline that first narrows a large candidate pool down to 500–5,000 items, then applies a richer feature set to the reduced candidates in a ranking model. A final reranking pass will apply any necessary business logic before surfacing recommendations to the user.
    - Latency will benefit from the restructuring described above, and will be further improved through caching — rather than running inference on every request, user-item pairs will be cached for efficiency. Additionally, an approximate nearest-neighbor (ANN) search will eliminate the need for exhaustive similarity comparisons between user embeddings and all items.

Note: My full-time job takes priority over this side project, and I will do my best to develop this out when I have chances on weekends. 


## Directory

```
.
├── amazon-books                    // ML / Replaces the former `ml/` folder with the amazon-books dataset training logic
│   ├── data
│   │   ├── Books_5.json.gz
│   │   ├── isbn_image.csv
│   │   └── meta_Books.json.gz
│   ├── main.ipynb
│   └── state
├── booky_api                       // Backend / API endpoints that are fed to GCP `booky_api/<model>/*`
│   └── two_towers_recommend
│       ├── dockerfile
│       ├── makefile
│       ├── pyproject.toml
│       ├── requirements.txt
│       └── src
│           ├── main.py
│           └── models.py
├── booky_recommender.png
├── booky_ui                        // UI
│   ├── Dockerfile
│   ├── eslint.config.js
│   ├── index.html
│   ├── nginx.conf
│   ├── package-lock.json
│   ├── package.json
│   ├── src
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── makefile
├── pyproject.toml
└── README.md
```

## Booky Design

<img src="booky_recommender.png" width="750px">
