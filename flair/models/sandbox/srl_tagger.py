import logging

from pathlib import Path
from typing import List, Union, Optional

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Sentence, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Metric, Result, store_embeddings

log = logging.getLogger("flair")

class SemanticRoleTagger(flair.nn.Model):
    def __init__(
            self,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            tag_type: str,
            embedding_size:int,
            rnn_hidden_size : int,
            beta: float = 1.0,
    ):
        super(SemanticRoleTagger, self).__init__()

        print("Embedding_size: "+str(embedding_size))
        print("rnn_hidden_size: "+str(rnn_hidden_size))

        # embeddings
        self.embeddings = embeddings

        # dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)


        # self.embedding = torch.nn.Embedding(len(vocabulary), embedding_size)
        # self.hidden2tag = torch.nn.Linear(rnn_hidden_size, len(vocabulary))
        # self.linear = torch.nn.Linear(self.embeddings.embedding_length, len(tag_dictionary))
        # Multihead Attention
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim = embedding_size,
                                                     num_heads = 8,
                                                     dropout = 0.8)

        self.lstm = torch.nn.LSTM(input_size = embedding_size,
                                 hidden_size = rnn_hidden_size,
                                 batch_first = True,
                                 bidirectional = True,
                                 dropout = 0.8)


        # F-beta score
        self.beta = beta
     
        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        features = self.forward(data_points)
        return self._calculate_loss(features, data_points)


    def forward(self, sentences: List[Sentence]):
        pass
        # Word & Predicate
        # Nonlinear Sublayer: RNN
        # Attention Sublayer: Self-Attention
        # Repeat
        # Softmax

        # Initial Embedding ?
        # Code-Schnipsel: 
        # embeds = self.embedding(one_hot_sentence)

        # lstm_out, _ = self.lstm(embeds)
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        # return F.log_softmax(features, dim=1)

    def make_tag_dictionary(data) -> Dict[str, int]::
        # A dictionary of tags available (e.g. A0)
        label_to_ix = {}
        for label in data: # depends on where data comes from
            if label not in label_to_ix:
                label_to_ix[label] = len(label_to_ix)
        return label_to_ix

    def make_word_dictionary(data) -> Dict[str, int]:
        # A dictionary of words available
        word_to_ix = {}
        for sent in data: # depends on where data comes from
            for word in sent:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        return word_to_ix

    # Questions to myself:
    # How does the data come in? (flair?)
    # Do I need a third method for frames?
    # What format will it be? (changing for loops)



    """def forward(self, sentences: List[Sentence]):

        self.embeddings.embed(sentences)

        names = self.embeddings.get_names()

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding(names)
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                    ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        features = self.linear(sentence_tensor)

        return features"""

    def _calculate_loss(
            self, features: torch.tensor, sentences: List[Sentence]
    ) -> float:

        # TODO
        return 0