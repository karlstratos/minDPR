import collections
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.serialization import default_restore_location
from transformers import AutoModel


class DualEncoder(torch.nn.Module):

    def __init__(self, tokenizer, query_encoder, passage_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder

    def forward(self, Q=None, Q_mask=None, Q_type=None, P=None, P_mask=None,
                P_type=None):
        X = self.query_encoder(Q, Q_mask, Q_type) if Q is not None else None
        Y = self.passage_encoder(P, P_mask, P_type) if P is not None else None
        return X, Y


class BertEncoder(torch.nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            'bert-base-uncased', attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        last_hidden_state = output[0]  # (B, L, d)
        embeddings = last_hidden_state[:, 0, :]  # (B, d)
        return embeddings


def make_bert_model(tokenizer, dropout=0.1):
    query_encoder = BertEncoder(dropout)
    passage_encoder = BertEncoder(dropout)
    model = DualEncoder(tokenizer, query_encoder, passage_encoder)
    return model


def load_model(model_path, tokenizer, device):
    saved_pickle = torch.load(model_path, map_location=device)
    model = make_bert_model(tokenizer).to(device)
    is_dpr = not 'sd' in saved_pickle

    if is_dpr:
        saved_state = load_states_from_checkpoint(model_path)
        question_state = {
            key[len("question_model."):]: value for (key, value) in
            saved_state.model_dict.items() if key.startswith("question_model.")
        }
        passage_state = {
            key[len("ctx_model."):]: value for (key, value) in
            saved_state.model_dict.items() if key.startswith("ctx_model.")
        }
        model.query_encoder.encoder.load_state_dict(question_state,
                                                    strict=False)
        model.passage_encoder.encoder.load_state_dict(passage_state,
                                                      strict=False)
        ArgsMock = collections.namedtuple('ArgsMock', ['max_length'])
        args_saved = ArgsMock(saved_state.encoder_params['sequence_length'])
    else:
        model.load_state_dict(saved_pickle['sd'])
        args_saved = saved_pickle['args']

    return model, args_saved, is_dpr


CheckpointState = collections.namedtuple('CheckpointState',
                                         ['model_dict', 'optimizer_dict',
                                          'scheduler_dict', 'offset',
                                          'epoch', 'encoder_params'])

def load_states_from_checkpoint(model_file):
    state_dict = torch.load(model_file, map_location=lambda s, l:
                            default_restore_location(s, 'cpu'))
    return CheckpointState(**state_dict)
