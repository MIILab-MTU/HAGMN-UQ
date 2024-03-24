import numpy as np
import tensorflow as tf

from .config import BaseConfig
from ..hypergraph import HypergraphsTuple
from .. import hypergraph


class ArteryConfig(BaseConfig):
    def __init__(self, feature_per_node):

        super(ArteryConfig, self).__init__()

        self.DATABASE_NAME = "artery"

        input_info = HypergraphsTuple(**{
            hypergraph.NODES: ((None, feature_per_node*2), 'float32'),
            hypergraph.N_NODE: ((None,), 'int32'),
            hypergraph.EDGES: ((None, feature_per_node*6), 'float32'),
            hypergraph.N_EDGE: ((None,), 'int32'),
            hypergraph.HYPEREDGES: ((None, 3), 'int32'),
            hypergraph.ROWS: ((None, 1), 'float32'),
            hypergraph.N_ROW: ((None,), 'int32'),
            hypergraph.ROW_ID: ((None,), 'int32'),
            hypergraph.COLS: ((None, 1), 'float32'),
            hypergraph.N_COL: ((None,), 'int32'),
            hypergraph.COL_ID: ((None,), 'int32'),
            hypergraph.GLOBALS: ((None, 1), 'float32'),
            hypergraph.N_GLOBAL: ((None,), 'int32'),
        })
        self.INPUT_SHAPE = tuple(shape for (shape, _) in input_info)
        self.INPUT_DTYPE = tuple(dtype for (_, dtype) in input_info)
        self.INPUT_SIGNATURE = HypergraphsTuple(*(tf.TensorSpec(shape=shape, dtype=dtype)
                                                  for (shape, dtype) in input_info))

        target_info = input_info.replace(nodes=((None, 1), 'float32'))
        self.TARGET_SHAPE = tuple(shape for (shape, _) in target_info)
        self.TARGET_DTYPE = tuple(dtype for (_, dtype) in target_info)
        self.TARGET_SIGNATURE = HypergraphsTuple(*(tf.TensorSpec(shape=shape, dtype=dtype)
                                                   for (shape, dtype) in target_info))
        # ==================================
