from posixpath import splitdrive
from re import L
import numpy as np
from transformers import Trainer
from datasets import concatenate_datasets

class CustomTrainer(Trainer):

    def splitEval(self, dataset):
        preds = self.predict(dataset)
        incorrect_data = dataset.filter(lambda ex, indx: bool(np.argmax(preds.predictions, axis=1)[indx] != preds.label_ids[indx]), with_indices=True)
        return  incorrect_data
    
    def trainHard(self):
        self.train_dataset = concatenate_datasets([self.train_dataset, self.splitEval(self.train_dataset)])
        self.train()

    # Trains the model by only looking at data that is hard to learn (i.e. data that is wrong, does this for multiple epochs)
    def trainChecklist(self, set1, set2, epochs):
        combined = concatenate_datasets([set1, set2])
        for e in range(epochs):
            self.train_dataset = self.splitEval(combined)
            self.train()