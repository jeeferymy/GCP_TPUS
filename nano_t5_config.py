# nano_t5_config.py
from transformers import T5Config

class NanoT5Config(T5Config):
    def __init__(self,
                 vocab_size=32000,
                 d_model=128,       # Tiny!
                 d_kv=32,
                 d_ff=256,
                 num_layers=2,
                 num_heads=4,
                 **kwargs):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            **kwargs
        )
