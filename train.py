from data_prep import prepare_dataset, tokenizer, addTokenLength
from model import FakeNewsClassifierModel, FakeNewsClassifierConfig

import numpy as np

from datasets import load_metric
from transformers import Trainer, TrainingArguments

tokenizedDataset = prepare_dataset("data/train.csv", tokenizer = tokenizer)
fullDataset = tokenizedDataset.map(addTokenLength)

tokenizedDevDataset = prepare_dataset("data/dev.csv", tokenizer = tokenizer)
fullDatasetDev = tokenizedDevDataset.map(addTokenLength)

hyperparams = {
    'bert_model_name': 'distilbert-base-uncased',
    'dropout_rate': 0.5,
    'num_classes': 2
}

config = FakeNewsClassifierConfig(**hyperparams)
model = FakeNewsClassifierModel(config)

training_args = TrainingArguments(
    output_dir='results/dbu-0.5drop',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.01,
    evaluation_strategy='steps',
    metric_for_best_model='accuracy',
    greater_is_better=True,
)

# metric = load_metric("accuracy")
metric = load_metric('glue', 'mrpc')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=fullDataset,
    eval_dataset=fullDatasetDev,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()