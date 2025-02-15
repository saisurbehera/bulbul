import torch
from torch import nn
from typing import Callable, List, Optional, Tuple, Union
import math
from functools import partial
from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock


from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import Qwen2Config

from knn_memory import KNNMemoryList, DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY
from contextlib import contextmanager

class ANNQwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with the ANN extension"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.knn_memory = self.create_knn_memories()



    def forward(self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            mem_kv, mem_mask = knn_memory.search(q, self.num_retrieved_memories)

    def create_knn_memories(
        self,
        *,
        batch_size
    ):
            return KNNMemoryList.create_memories(
                batch_size = batch_size,
                num_memory_layers = self.num_memory_layers,
                memories_directory = self.knn_memories_directory,
            )(**self.knn_mem_kwargs)

    @contextmanager
    def knn_memories_context(
        self,
        **kwargs
    ):
        knn_dir = Path(self.knn_memories_directory)
        knn_dir.mkdir(exist_ok = True, parents = True)
        lock = FileLock(str(knn_dir / 'mutex'))

        with lock:
            knn_memories = self.create_knn_memories(**kwargs)
            yield knn_memories
            knn_memories.cleanup()
