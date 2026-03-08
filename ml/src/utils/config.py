class Config:
    def __init__(self, dataset):
        EMBEDDING_SIZE = 128

        NUM_USERS = len(dataset.encoders['User-ID']) + 1
        NUM_AGES = len(dataset.encoders['User-Age']) + 1
        NUM_ISBN = len(dataset.encoders['ISBN']) + 1
        NUM_TITLES = len(dataset.encoders['Book-Title']) + 1
        NUM_AUTHORS = len(dataset.encoders['Book-Author']) + 1
        NUM_PUBLISHERS = len(dataset.encoders['Publisher']) + 1
        NUM_YEAR_OF_PUBLICATIONS = len(dataset.encoders['Book-Year-Of-Publication']) + 1

        EPOCHS = 50
        LEARNING_RATE = 5e-4
        EARLY_STOPPING_PATIENCE = 15
        TEMPERATURE = 0.1
        WEIGHT_DECAY = 1e-5
