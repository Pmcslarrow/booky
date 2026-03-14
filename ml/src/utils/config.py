class Config:
    def __init__(self, dataset):
        self.MODEL_SAVE_PATH = 'ml/artifacts/models/batch_training'

        self.EMBEDDING_SIZE = 128
        self.EPOCHS = 50
        self.LEARNING_RATE = 5e-4
        self.EARLY_STOPPING_PATIENCE = 15
        self.TEMPERATURE = 0.1
        self.WEIGHT_DECAY = 1e-5

        self.NUM_USERS = len(dataset.encoders['User-ID']) + 1
        self.NUM_AGES = len(dataset.encoders['User-Age']) + 1
        self.NUM_ISBN = len(dataset.encoders['ISBN']) + 1
        self.NUM_TITLES = len(dataset.encoders['Book-Title']) + 1
        self.NUM_AUTHORS = len(dataset.encoders['Book-Author']) + 1
        self.NUM_PUBLISHERS = len(dataset.encoders['Publisher']) + 1
        self.NUM_YEAR_OF_PUBLICATIONS = len(dataset.encoders['Book-Year-Of-Publication']) + 1

    def __str__(self):
        return (
            f"EMBEDDING_SIZE: {self.EMBEDDING_SIZE}\n"
            f"EPOCHS: {self.EPOCHS}\n"
            f"LEARNING_RATE: {self.LEARNING_RATE}\n"
            f"EARLY_STOPPING_PATIENCE: {self.EARLY_STOPPING_PATIENCE}\n"
            f"TEMPERATURE: {self.TEMPERATURE}\n"
            f"WEIGHT_DECAY: {self.WEIGHT_DECAY}\n"
            f"NUM_USERS: {self.NUM_USERS}\n"
            f"NUM_AGES: {self.NUM_AGES}\n"
            f"NUM_ISBN: {self.NUM_ISBN}\n"
            f"NUM_TITLES: {self.NUM_TITLES}\n"
            f"NUM_AUTHORS: {self.NUM_AUTHORS}\n"
            f"NUM_PUBLISHERS: {self.NUM_PUBLISHERS}\n"
            f"NUM_YEAR_OF_PUBLICATIONS: {self.NUM_YEAR_OF_PUBLICATIONS}"
        )      



