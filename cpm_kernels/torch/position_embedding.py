import torch
from .embedding import Embedding
from ..kernels import position_bucket
import math

def batched_position_bucket(batch, key_len, query_len, num_buckets, max_distance, bidirectional, device) -> torch.Tensor:
    pos = torch.empty((1, key_len * query_len), dtype=torch.int32, device=device)
    position_bucket(
        query_len,
        key_len,
        num_buckets,
        max_distance,
        pos.data_ptr(),
        bidirectional,
        torch.cuda.current_stream().cuda_stream
    )
    pos = pos.repeat(batch, 1)
    return pos

class PositionEmbedding(torch.nn.Module):
    def __init__(self, num_heads, num_buckets, max_distance, bidirectional=True):
        super(PositionEmbedding, self).__init__()
        self.embedding = Embedding(num_buckets, num_heads)
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional


    def forward(self, batch, key_len, query_len):
        pos = batched_position_bucket(
            batch,
            key_len,
            query_len,
            self.num_buckets,
            self.max_distance,
            self.bidirectional,
            self.embedding.weight.device
        )
        assert pos.is_contiguous()
        return self.embedding(pos).view(batch, self.num_heads, key_len, query_len)



class PositionEmbeddingTH(torch.nn.Module):
    def __init__(self, num_heads, num_buckets, max_distance, bidirectional=True) -> None:
        super(PositionEmbeddingTH, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.embedding = torch.nn.Embedding(self.num_buckets, self.num_heads)

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        device = self.embedding.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self.relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.embedding(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 1, 0]).unsqueeze(0)  # shape (1, num_heads, key_length, query_length)
        return values
    
    def forward(self, batch, key_length, query_length):
        return self.compute_bias(query_length, key_length).repeat(batch, 1, 1, 1)
