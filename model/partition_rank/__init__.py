# type: ignore
"""
PartitionRank Models: Ensemble models based on variable partitioning.

Each term partitions one variable from the others, creating a rank decomposition.
"""

from model.partition_rank.partition_rank_models import PartitionRank3, distribute_partition_rank
