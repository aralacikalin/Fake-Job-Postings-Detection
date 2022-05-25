import re
import string
import pandas as pd

import numpy as np
from typing import Optional

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertTokenizerFast
from datasets import Dataset, load_metric, load_dataset

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def cleaner(x, maxLength):
	if(type(x)!=str):
		return x
	fullText = x
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
	return ' '.join(newFullText.split(maxsplit=maxLength)[:maxLength])

def addTokenLength(example):
	tokenLength = 0
	for tok in example["input_ids"]:
		if tok != 0:
			tokenLength += 1
	# tokenLength=len(example["input_ids"])
	example['text_length'] = tokenLength
	return example

def charCleaning(example):
	fullText = example['text']
	newFullText = ""
	# example['text']=fullText.encode('ascii',errors='ignore')
	printable = set(string.printable)
	for char in fullText:
		if (not(char in printable)):
			newFullText += " "
		else:
			newFullText += char

	newFullText=re.sub('\(#[^#]*#\)', '', newFullText,  flags=re.DOTALL)
	newFullText=re.sub('#EMAIL[^#]*#', '', newFullText,  flags=re.DOTALL)
	newFullText=re.sub('#URL[^#]*#', '', newFullText,  flags=re.DOTALL)
	newFullText=re.sub('#PHONE[^#]*#', '', newFullText,  flags=re.DOTALL)
	newFullText=re.sub('{[^}]*}', '', newFullText,  flags=re.DOTALL)
	example['text']=newFullText
	return example

def prepare_dataset(csv_path = "/content/train.csv", tokenizer = tokenizer, test_set = False, max_size_des = 200, max_size_other = 30):
	# Reading csv file, and dropping irrelevant columns
	data = pd.read_csv(csv_path)
	data = data.drop(columns = ["job_id","salary_range","department","required_education"])
	data = data.drop(data.columns[0], axis = 1)

	# cleaning input text and slicing length to model input size 
	data['description'] = data['description'].apply(lambda x: cleaner(x, max_size_des))
	data['requirements'] = data['requirements'].apply(lambda x: cleaner(x, max_size_other))
	data['benefits'] = data['benefits'].apply(lambda x: cleaner(x, max_size_other))
	data['company_profile'] = data['company_profile'].apply(lambda x: cleaner(x, max_size_other))

	# Merging Input columns into `text` and creating output column as `label`
	data['text'] = data[['title', 'location', "company_profile","description","requirements","employment_type","industry","function"]].astype(str).agg(' [SEP] '.join, axis=1)
	if not test_set:
		data["label"] = data["fraudulent"]
		data = data.drop(columns=["fraudulent"])
	# Removing all columns except 'text' and 'label'
	data = data.drop(columns=['title', 'location', "company_profile","description","requirements","benefits","employment_type","required_experience","industry","function"])

	# Building dataset from pandas frame
	dataset = Dataset.from_pandas(data)
	dataset = dataset.map(charCleaning)
	dataset = dataset.map(lambda examples: tokenizer(examples['text'],truncation=True,padding=True), batched=True)
	return dataset


tokenizedDataset = prepare_dataset("data/train.csv", tokenizer = tokenizer)
fullDataset = tokenizedDataset.map(addTokenLength)

