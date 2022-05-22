import torch
import torch.nn as nn
from typing import Optional

from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,has_company_logo,has_questions,text_length,telecommuting,
                labels: Optional[torch.Tensor] = None) -> SequenceClassifierOutput:
        bert_output = self.bert(input_ids, attention_mask)

        # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        last_hidden_state = bert_output[0]

        # torch.FloatTensor of shape (batch_size, hidden_size)
        pooled_output = last_hidden_state[:, 0]

        has_company_logo=torch.reshape(has_company_logo,(has_company_logo.size(0),1))
        telecommuting=torch.reshape(telecommuting,(telecommuting.size(0),1))
        text_length=torch.reshape(text_length,(text_length.size(0),1))
        has_questions=torch.reshape(has_questions,(has_questions.size(0),1))

        # torch.FloatTensor of shape (batch_size, num_labels)
        # print(pooled_output.size(),has_company_logo.size(),has_questions.size(),telecommuting.size(),text_length.size())
        logits = self.clf(torch.cat((pooled_output,has_company_logo,has_questions,telecommuting,text_length),dim=-1))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

