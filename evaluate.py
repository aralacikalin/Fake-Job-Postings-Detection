import pandas as pd
from sklearn.metrics import f1_score
import argparse

if __name__ == '__main__':
	expected_version = '1.1'
	parser = argparse.ArgumentParser(
		description='Evaluation for SQuAD ' + expected_version)
	parser.add_argument('dataset_file', help='Dataset file')
	parser.add_argument('prediction_file', help='Prediction File')
	args = parser.parse_args()

	dataset_df = pd.read_csv(args.dataset_file)
	dataset = {}
	for row in dataset_df.iterrows():
		dataset[row[1]["job_id"]] = row[1]["fraudulent"]

	predictions_df = pd.read_csv(args.prediction_file)
	predictions = {}
	for row in predictions_df.iterrows():
		predictions[row[1]["job_id"]] = row[1]["fraudulent"]

	y_true = []
	y_pred = []
	for k, v in dataset.items():
		if k in predictions:
			y_true.append(v)
			y_pred.append(predictions[k])
		else:
			print(f'Prediction with job_id {k} is missing. Skipping...')

	f1_pos = f1_score(y_true, y_pred, average='binary')
	f1_micro = f1_score(y_true, y_pred, average='micro')
	f1_macro = f1_score(y_true, y_pred, average='macro')

	results = {
		'f1_pos': f1_pos,
		'f1_micro': f1_micro,
		'f1_macro': f1_macro
	}

	print(results)
