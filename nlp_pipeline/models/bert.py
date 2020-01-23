"""
Creating models for Bert. 
Features the model class and the learner (used to split layers.
"""
from nlp_pipeline.models.pooler import *
from fastai.text import *
from transformers import BertPreTrainedModel, BertModel


class BertSequenceClassification(BertPreTrainedModel):
    """
    Edit the Default Bert For Sequence Classification
    """

    def __init__(self, config, dropout_rate, hidden_layer_output):
        # best was dropout 0.15 -> classifier
        super(BertSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.hidden_layer_output = hidden_layer_output
        # Adjust the number here based on the number of features to make
        hidden_size_and_features = config.hidden_size

        # Adding a custom pooler
        self.pooler = BertPooler(config.hidden_size)
        # self.avgpool = nn.AdaptiveAvgPool1d(hidden_size_and_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size_and_features, self.config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        engineered_features=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_state = outputs[0]
        pooled_output = outputs[1]
        all_hid = outputs[2]
        pooled_output = self.pooler(all_hid[-self.hidden_layer_output])
        # Feeding
        if engineered_features is not None:
            pooled_output = torch.cat((pooled_output, engineered_features), dim=1)

        # Using global average pooling over the dataset first
        pooled_output = self.dropout(pooled_output)
        # We apply sigmoid here because we can't use Relu and then we batchnorm
        # Afterwards to get a proper responds
        logits = self.classifier(pooled_output)
        # Consider applying Sigmoid to function
        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        return logits.unsqueeze(0)  # (loss), logits, (hidden_states), (attentions)


class BertSequenceClassification(BertPreTrainedModel):
    """
    Edit the Default Bert For Sequence Classification
    """

    def __init__(self, config, dropout_rate, hidden_layer_output):
        # best was dropout 0.15 -> classifier
        super(BertSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.hidden_layer_output = hidden_layer_output
        # Adjust the number here based on the number of features to make
        hidden_size_and_features = config.hidden_size

        # Adding a custom pooler
        self.pooler = BertPooler(config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size_and_features, self.config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        engineered_features=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_state = outputs[0]
        pooled_output = outputs[1]
        all_hid = outputs[2]
        pooled_output = self.pooler(all_hid[-self.hidden_layer_output])
        if engineered_features is not None:
            pooled_output = torch.cat((pooled_output, engineered_features), dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]
        return logits.unsqueeze(0)  # (loss), logits, (hidden_states), (attentions)


class BertLearner(Learner):
    def __init__(self, databunch, model, **kwargs):
        super(BertLearner, self).__init__(databunch, model, **kwargs)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.model = model
        self.split(self.get_layer_groups())

    def get_layer_groups(self):
        group_splits = [
            self.model.transformer.bert.encoder.layer[0:1],
            self.model.transformer.bert.encoder.layer[2:3],
            self.model.transformer.bert.encoder.layer[4:5],
            self.model.transformer.bert.encoder.layer[6:7],
            self.model.transformer.bert.encoder.layer[8:9],
            self.model.transformer.bert.encoder.layer[10:11],
            self.model.transformer.bert.pooler,
            self.model.transformer.classifier,
        ]
        return group_splits


class BertFeatLearner(Learner):
    def __init__(self, databunch, model, **kwargs):
        super(BertFeatLearner, self).__init__(databunch, model, **kwargs)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.model = model
        self.split(self.get_layer_groups())

    def get_layer_groups(self):
        group_splits = [
            self.model.transformer.bert.encoder.layer[0:1],
            self.model.transformer.bert.encoder.layer[2:3],
            self.model.transformer.bert.encoder.layer[4:5],
            self.model.transformer.bert.encoder.layer[6:7],
            self.model.transformer.bert.encoder.layer[8:9],
            self.model.transformer.bert.encoder.layer[10:11],
            self.model.pooler,
            self.model.classifier,
        ]
        return group_splits


class BertForFeatures(BertPreTrainedModel):
    """
    Edit the Default Bert For Sequence Classification
    """

    def __init__(self, config, dropout_rate, hidden_layer_output):
        # best was dropout 0.15 -> classifier
        super(BertForFeatures, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.hidden_layer_output = hidden_layer_output
        self.dropout_rate = dropout_rate
        
        # Adjust the number here based on the number of features to make
        hidden_size_and_features = config.hidden_size

        # Adding a custom pooler
        self.pooler = BertPooler(config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        engineered_features=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        hidden_state = outputs[0]
        pooled_output = outputs[1]
        all_hid = outputs[2]
        pooled_output = self.pooler(all_hid[-self.hidden_layer_output])
        return pooled_output  # (loss), logits, (hidden_states), (attentions)

class BertWithEmbeds(BertPreTrainedModel):
    """
    Edit the Default Bert For Sequence Classification
    """

    def __init__(self, config, dropout_rate, hidden_layer_output):
        # best was dropout 0.15 -> classifier
        super(BertWithEmbeds, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.hidden_layer_output = hidden_layer_output
        self.dropout_rate = dropout_rate
        
        # Adjust the number here based on the number of features to make
        hidden_size_and_features = config.hidden_size

        # Adding a custom pooler
        self.pooler = BertPooler(config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def forward(
        self,
        inputs_embeds=None,
        engineered_features=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_state = outputs[0]
        pooled_output = outputs[1]
        all_hid = outputs[2]
        pooled_output = self.pooler(all_hid[-self.hidden_layer_output])
        return pooled_output  # (loss), logits, (hidden_states), (attentions)