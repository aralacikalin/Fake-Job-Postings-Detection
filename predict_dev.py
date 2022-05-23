from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset, Dataset, load_metric
from transformers import AutoModel, PretrainedConfig
import pandas as pd
from transformers import TrainingArguments, Trainer
import numpy as np

from model import FakeNewsClassifierModel, FakeNewsClassifierConfig
from data_prep import prepare_dataset, tokenizer, addTokenLength


eval_dataset = prepare_dataset("data/dev.csv", tokenizer = tokenizer, test_set=True)
eval_dataset = eval_dataset.map(addTokenLength)

AutoConfig.register("fakenews", FakeNewsClassifierConfig)
AutoModelForSequenceClassification.register(FakeNewsClassifierConfig, FakeNewsClassifierModel)

model = AutoModelForSequenceClassification.from_pretrained('results/dbu-0.25drop/checkpoint-44500')
tokenizer = AutoTokenizer.from_pretrained('results/dbu-0.25drop/checkpoint-44500')

training_args = TrainingArguments(
	output_dir='results/4-res',
	learning_rate=1e-5,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	num_train_epochs=300,
	weight_decay=0.01,
	evaluation_strategy='steps',
	metric_for_best_model='accuracy',
	greater_is_better=True,
)

metric = load_metric('glue', 'mrpc')


def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
	model=model,
	args=training_args,
	# train_dataset=fullDataset,
	# eval_dataset=fullDataset_dev,
	tokenizer=tokenizer,
	compute_metrics=compute_metrics
)

# results=trainer.evaluate()
results=trainer.predict(eval_dataset)[0]
data = pd.read_csv("data/dev.csv")
data=data.drop(columns=['title', 'location', "company_profile","telecommuting","has_company_logo","has_questions","description","requirements","benefits","employment_type","required_experience","industry","function","salary_range","department","required_education"])
data=data.drop(data.columns[0], axis=1)

resultLabels=[]
for res in results:
	resultLabels.append(np.argmax(res))

data["fraudulent"]=resultLabels
data.to_csv("data/dev_predicted.csv")


# print(results)
