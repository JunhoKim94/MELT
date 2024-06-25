from transformers import BertConfig, ElectraConfig
import argparse

Bert_Small_Config = BertConfig(vocab_size = 31090,
                                hidden_size = 256,
                                num_hidden_layers = 12,
                                num_attention_heads = 4,
                                intermediate_size = 1024,
                                hidden_act = "gelu",
                                hidden_dropout_prob = 0.1,
                                attention_probs_dropout_prob = 0.1,
                                max_position_embeddings = 512,
                                initializer_range = 0.02,
                                layer_norm_eps = 1e-12,
                                pad_token_id = 0,
                                gradient_checkpointing = False)

