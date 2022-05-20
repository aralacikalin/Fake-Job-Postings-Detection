import torch
import numpy
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import string
import re
data = pd.read_csv("/gpfs/space/home/aral/nlpProject/train.csv")
data = data.drop(columns=["job_id","salary_range","department","required_education"])
data=data.drop(data.columns[0], axis=1)

def cleaner(x,maxLenght):
    if(type(x)!=str):
      return x
    fullText=x
    newFullText=""
    # example['text']=fullText.encode('ascii',errors='ignore')
    printable = set(string.printable)
    for char in fullText:
      if(not (char in printable)):
        newFullText+=" "
      else:
        newFullText+=char

    newFullText=re.sub('\(#[^#]*#\)', '', newFullText,  flags=re.DOTALL)
    newFullText=re.sub('#EMAIL[^#]*#', '', newFullText,  flags=re.DOTALL)
    newFullText=re.sub('#URL[^#]*#', '', newFullText,  flags=re.DOTALL)
    newFullText=re.sub('#PHONE[^#]*#', '', newFullText,  flags=re.DOTALL)
    newFullText=re.sub('{[^}]*}', '', newFullText,  flags=re.DOTALL)
    return ' '.join(newFullText.split(maxsplit=maxLenght)[:maxLenght]) 

max_size_des=200
data['description'] = data['description'].apply(lambda x: cleaner(x,200))
# data['description'] = data['description'].apply(lambda x: "")
data['requirements'] = data['requirements'].apply(lambda x: cleaner(x,30))
data['benefits'] = data['benefits'].apply(lambda x: cleaner(x,30))
data['company_profile'] = data['company_profile'].apply(lambda x: cleaner(x,30))

data['text'] = data[['title', 'location', "company_profile","description","requirements","employment_type","industry","function"]].astype(str).agg(' [SEP] '.join, axis=1)
data["label"]=data["fraudulent"]
data=data.drop(columns=['title', 'location', "company_profile","description","requirements","benefits","employment_type","required_experience","industry","function","fraudulent"])
dataset = Dataset.from_pandas(data)
def addTokenLenght(example):
  tokenLength=0
  for tok in example["input_ids"]:
    if(tok!=0):
      tokenLength+=1
  # tokenLength=len(example["input_ids"])
  example['text_lenght']=tokenLength
  return example

import string
import re
def charCleaning(example):
  # specialChars=[]
  fullText=example['text']
  newFullText=""
  # example['text']=fullText.encode('ascii',errors='ignore')
  printable = set(string.printable)
  for char in fullText:
    if(not (char in printable)):
      newFullText+=" "
    else:
      newFullText+=char

  newFullText=re.sub('\(#[^#]*#\)', '', newFullText,  flags=re.DOTALL)
  newFullText=re.sub('#EMAIL[^#]*#', '', newFullText,  flags=re.DOTALL)
  newFullText=re.sub('#URL[^#]*#', '', newFullText,  flags=re.DOTALL)
  newFullText=re.sub('#PHONE[^#]*#', '', newFullText,  flags=re.DOTALL)
  newFullText=re.sub('{[^}]*}', '', newFullText,  flags=re.DOTALL)
  example['text']=newFullText

  return example
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

cleanedDataset=dataset.map(charCleaning)
tokenizedDataset = cleanedDataset.map(lambda examples: tokenizer(examples['text'],truncation=True,padding=True), batched=True)
fullDataset=tokenizedDataset.map(addTokenLenght)

data = pd.read_csv("/gpfs/space/home/aral/nlpProject/dev.csv")
data = data.drop(columns=["job_id","salary_range","department","required_education"])
data=data.drop(data.columns[0], axis=1)

def cleaner(x,maxLenght):
    if(type(x)!=str):
      return x
    fullText=x
    newFullText=""
    # example['text']=fullText.encode('ascii',errors='ignore')
    printable = set(string.printable)
    for char in fullText:
      if(not (char in printable)):
        newFullText+=" "
      else:
        newFullText+=char

    newFullText=re.sub('\(#[^#]*#\)', '', newFullText,  flags=re.DOTALL)
    newFullText=re.sub('#EMAIL[^#]*#', '', newFullText,  flags=re.DOTALL)
    newFullText=re.sub('#URL[^#]*#', '', newFullText,  flags=re.DOTALL)
    newFullText=re.sub('#PHONE[^#]*#', '', newFullText,  flags=re.DOTALL)
    newFullText=re.sub('{[^}]*}', '', newFullText,  flags=re.DOTALL)
    return ' '.join(newFullText.split(maxsplit=maxLenght)[:maxLenght]) 

max_size_des=200
data['description'] = data['description'].apply(lambda x: cleaner(x,200))
# data['description'] = data['description'].apply(lambda x: "")
data['requirements'] = data['requirements'].apply(lambda x: cleaner(x,30))
data['benefits'] = data['benefits'].apply(lambda x: cleaner(x,30))
data['company_profile'] = data['company_profile'].apply(lambda x: cleaner(x,30))

data['text'] = data[['title', 'location', "company_profile","description","requirements","employment_type","industry","function"]].astype(str).agg(' [SEP] '.join, axis=1)
data["label"]=data["fraudulent"]
data=data.drop(columns=['title', 'location', "company_profile","description","requirements","benefits","employment_type","required_experience","industry","function","fraudulent"])
dataset = Dataset.from_pandas(data)
def addTokenLenght(example):
  tokenLength=0
  for tok in example["input_ids"]:
    if(tok!=0):
      tokenLength+=1
  # tokenLength=len(example["input_ids"])
  example['text_lenght']=tokenLength
  return example

import string
import re
def charCleaning(example):
  # specialChars=[]
  fullText=example['text']
  newFullText=""
  # example['text']=fullText.encode('ascii',errors='ignore')
  printable = set(string.printable)
  for char in fullText:
    if(not (char in printable)):
      newFullText+=" "
    else:
      newFullText+=char

  newFullText=re.sub('\(#[^#]*#\)', '', newFullText,  flags=re.DOTALL)
  newFullText=re.sub('#EMAIL[^#]*#', '', newFullText,  flags=re.DOTALL)
  newFullText=re.sub('#URL[^#]*#', '', newFullText,  flags=re.DOTALL)
  newFullText=re.sub('#PHONE[^#]*#', '', newFullText,  flags=re.DOTALL)
  newFullText=re.sub('{[^}]*}', '', newFullText,  flags=re.DOTALL)
  example['text']=newFullText

  return example
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

cleanedDataset=dataset.map(charCleaning)
tokenizedDataset = cleanedDataset.map(lambda examples: tokenizer(examples['text'],truncation=True,padding=True), batched=True)
fullDataset_dev=tokenizedDataset.map(addTokenLenght)
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from typing import Optional
import numpy as np

class FakeNewsClassifierConfig(PretrainedConfig):
    model_type = "fakenews"

    def __init__(
            self,
            bert_model_name: str = 'distilbert-base-uncased',
            dropout_rate: float = 0.5,
            num_classes: int = 2,
            **kwargs) -> None:
        """Initialize the Fake News Classifier Confing.

        Args:
            bert_model_name (str, optional): Name of pretrained BERT model. Defaults to 'distilbert-base-uncased'.
            dropout_rate (float, optional): Dropout rate for the classification head. Defaults to 0.5.
            num_classes (int, optional): Number of classes to predict. Defaults to 2.
        """
        self.bert_model_name = bert_model_name
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        super().__init__(**kwargs)


class FakeNewsClassifierModel(PreTrainedModel):
    """DistilBERT based model for fake news classification."""

    config_class = FakeNewsClassifierConfig

    def __init__(self, config: PretrainedConfig) -> None:
        """Initialize the Fake News Classifier Model.

        Args:
            config (PretrainedConfig): Config with model's hyperparameters.
        """
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = AutoModel.from_pretrained(config.bert_model_name)
        self.clf = nn.Sequential(
            nn.Linear(self.bert.config.dim+4, self.bert.config.dim+4),
            nn.ELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.bert.config.dim+4, config.num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,has_company_logo,has_questions,text_lenght,telecommuting,
                labels: Optional[torch.Tensor] = None) -> SequenceClassifierOutput:
        bert_output = self.bert(input_ids, attention_mask)

        # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        last_hidden_state = bert_output[0]

        # torch.FloatTensor of shape (batch_size, hidden_size)
        pooled_output = last_hidden_state[:, 0]

        has_company_logo=torch.reshape(has_company_logo,(has_company_logo.size(0),1))
        telecommuting=torch.reshape(telecommuting,(telecommuting.size(0),1))
        text_lenght=torch.reshape(text_lenght,(text_lenght.size(0),1))
        has_questions=torch.reshape(has_questions,(has_questions.size(0),1))

        # torch.FloatTensor of shape (batch_size, num_labels)
        # print(pooled_output.size(),has_company_logo.size(),has_questions.size(),telecommuting.size(),text_lenght.size())
        logits = self.clf(torch.cat((pooled_output,has_company_logo,has_questions,telecommuting,text_lenght),dim=-1))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

hyperparams = {
    'bert_model_name': 'distilbert-base-uncased',
    'dropout_rate': 0.5,
    'num_classes': 2
}
config = FakeNewsClassifierConfig(**hyperparams)
model = FakeNewsClassifierModel(config)

training_args = TrainingArguments(
    output_dir='/gpfs/space/home/aral/nlpProject/results/6-0.5drop',
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
    eval_dataset=fullDataset_dev,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()