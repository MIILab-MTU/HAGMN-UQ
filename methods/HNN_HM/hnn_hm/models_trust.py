import itertools

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Layer, LayerNormalization

from . import hypergraph, utils
from .hypergraph import HypergraphsTuple
from .hypergraph_utils_tf import concat_graphs


def mlp(layer_sizes, output_use_activation='sigmoid', output_use_bias=True):
    assert len(layer_sizes) >= 1

    layers = Sequential()
    for latent_size in layer_sizes[:-1]:
        layers.add(Dense(units=latent_size, activation='relu', use_bias=True))

    # output_activation = 'relu' if output_use_activation else None
    layers.add(Dense(layer_sizes[-1], activation=output_use_activation, use_bias=output_use_bias))

    return layers


def mlp_with_normalization(layer_sizes, output_use_activation=True, output_use_bias=True, use_normalization=True):
    """Instantiates a new MLP, followed by LayerNorm.  """
    layers = mlp(layer_sizes, output_use_activation=output_use_activation, output_use_bias=output_use_bias)
    if use_normalization:
        layers.add(LayerNormalization())
    return layers


class HypergraphIndependent(Layer):
    def __init__(self, layer_sizes_dict, output_use_activation='sigmoid', output_use_bias=True, use_normalization=True, name="hypergraph_independent"):
        super(HypergraphIndependent, self).__init__(name=name)
        self.network_names = {}
        for field, layer_sizes in layer_sizes_dict.items():
            network = mlp_with_normalization(layer_sizes=layer_sizes, output_use_activation=output_use_activation,
                                             output_use_bias=output_use_bias, use_normalization=use_normalization)
            network_name = '{}_network'.format(field)
            setattr(self, network_name, network)
            self.network_names[field] = network_name

    def call(self, graph):
        feats = {}
        for field, network_name in self.network_names.items():
            feat = getattr(graph, field)
            network = getattr(self, network_name)
            feats[field] = network(feat)

        return graph.replace(**feats)


class HypergraphOutputSoftmax(Layer):
    def __init__(self, name="hypergraph_output_softmax", prob_vec=False):
        super(HypergraphOutputSoftmax, self).__init__(name=name)
        self.prob_vec = prob_vec

    def call(self, graph):
        feat = getattr(graph, hypergraph.NODES)
        if self.prob_vec:
            feat = tf.nn.softmax(feat)
        else:
            feat = tf.nn.softmax(feat)[:, 1]
        feats = {hypergraph.NODES: feat}
        return graph.replace(**feats)

class HypergraphOutputScaler2Onehot(Layer):
    def __init__(self, name="hypergraph_output_reverse"):
        super(HypergraphOutputScaler2Onehot, self).__init__(name=name)
    
    def call(self, graph):
        feat = getattr(graph, hypergraph.NODES)
        # one_hot_feat = tf.zeros([feat.shape[0], 2])
        # one_hot_feat[:, 1] = feat
        # one_hot_feat[:, 0] = 1-feat
        feat = tf.stack([1-feat, feat], axis=1)
        feat = tf.squeeze(feat)
        feats = {hypergraph.NODES: feat}
        return graph.replace(**feats)


class HypergraphOutputNormalByRow(Layer):
    def __init__(self, name="HypergraphOutputNormalByRow"):
        super(HypergraphOutputNormalByRow, self).__init__(name=name)
    
    def call(self, graph):
        feat = getattr(graph, hypergraph.NODES)
        row_ids = getattr(graph, hypergraph.ROW_ID)
        num_rows = tf.reduce_sum(graph.n_row)

        segment_max = tf.math.unsorted_segment_max(feat, row_ids, num_rows)
        feat_max = tf.gather(segment_max, row_ids)
        feat_residule = feat_max - feat

        feat = tf.stack([feat_residule, feat], axis=1)
        feat = tf.squeeze(feat)
        feats = {hypergraph.NODES: feat}
        return graph.replace(**feats)


class HypergraphNodeBlock(Layer):
    def __init__(self,
                 layer_sizes,
                 output_use_activation=True,
                 output_use_bias=True,
                 use_normalization=True,
                 name="hypergraph_node_block"):
        super(HypergraphNodeBlock, self).__init__(name=name)

        self.network = mlp_with_normalization(layer_sizes=layer_sizes,
                                              output_use_activation=output_use_activation,
                                              output_use_bias=output_use_bias,
                                              use_normalization=use_normalization)

    def call(self, graph):
        globals = utils._one_to_segments(graph.globals, graph.n_node)
        input_list = [graph.nodes, globals]
        for (feature_field, index_field, is_index_of_nodes) in zip(hypergraph.HYPEREDGES_FEATURE_FIELDS,
                                                                   hypergraph.HYPEREDGES_INDEX_FIELDS,
                                                                   hypergraph.HYPEREDGES_INDEX_OF_NODES):
            feat = getattr(graph, feature_field)
            ind = getattr(graph, index_field)
            if is_index_of_nodes:
                feat_agg = utils._one_to_fixed_number(feat, ind, tf.reduce_sum(graph.n_node))
            else:
                feat_agg = utils._one_to_unsorted_segments(feat, ind)
            input_list.append(feat_agg)

        inputs = tf.concat(input_list, axis=1)
        nodes = self.network(inputs)
        return graph.replace(nodes=nodes)


class HypergraphNodeAttentionBlock(Layer):
    def __init__(self,
                 layer_sizes,
                 attention_key_size=8,
                 output_use_activation=True,
                 output_use_bias=True,
                 use_normalization=True,
                 name="hypergraph_node_attention_block"):
        """node block with edge attention"""

        super(HypergraphNodeAttentionBlock, self).__init__(name=name)

        attention_head_number = 8
        self.node2query = Dense(units=attention_key_size, use_bias=True)
        self.edge2key = Dense(units=attention_key_size, use_bias=True)
        self.multihead = tf.keras.layers.Conv1D(filters=attention_head_number, kernel_size=4, padding='same')

        self.network = mlp_with_normalization(layer_sizes=layer_sizes,
                                              output_use_activation=output_use_activation,
                                              output_use_bias=output_use_bias,
                                              use_normalization=use_normalization)

        self.flatten = tf.keras.layers.Flatten()

    def call(self, graph):
        globals = utils._one_to_segments(graph.globals, graph.n_node)
        input_list = [graph.nodes, globals]
        for (feature_field, index_field, is_index_of_nodes) in zip(hypergraph.HYPEREDGES_FEATURE_FIELDS,
                                                                   hypergraph.HYPEREDGES_INDEX_FIELDS,
                                                                   hypergraph.HYPEREDGES_INDEX_OF_NODES):
            feat = getattr(graph, feature_field)
            ind = getattr(graph, index_field)
            if feature_field == hypergraph.EDGES:
                node_queries = self.node2query(graph.nodes)
                edge_keys = self.edge2key(feat)

                node_queries_multihead = self.multihead(node_queries[:, None, :])
                edge_keys_multihead = self.multihead(edge_keys[:, None, :])
                edge_values_multihead = edge_keys_multihead

                edges_multihead = utils._one_to_fixed_number_attention(node_queries_multihead, edge_keys_multihead, edge_values_multihead, ind)
                feat_agg = self.flatten(edges_multihead)
            elif is_index_of_nodes:
                feat_agg = utils._one_to_fixed_number(feat, ind, tf.reduce_sum(graph.n_node))
            else:
                feat_agg = utils._one_to_unsorted_segments(feat, ind)
            input_list.append(feat_agg)

        inputs = tf.concat(input_list, axis=1)
        nodes = self.network(inputs)
        return graph.replace(nodes=nodes)


class HypergraphEdgeBlock(Layer):
    def __init__(self,
                 layer_sizes,
                 output_use_activation=True,
                 output_use_bias=True,
                 use_normalization=True,
                 field=hypergraph.EDGES,
                 name="hypergraph_edge_block"):
        super(HypergraphEdgeBlock, self).__init__(name=name)

        idx = hypergraph.HYPEREDGES_FEATURE_FIELDS.index(field)
        self.feature_field = hypergraph.HYPEREDGES_FEATURE_FIELDS[idx]
        self.index_field = hypergraph.HYPEREDGES_INDEX_FIELDS[idx]
        self.number_field = hypergraph.HYPEREDGES_NUMBER_FIELDS[idx]
        self.is_index_of_nodes = hypergraph.HYPEREDGES_INDEX_OF_NODES[idx]

        self.network = mlp_with_normalization(layer_sizes=layer_sizes,
                                              output_use_activation=output_use_activation,
                                              output_use_bias=output_use_bias,
                                              use_normalization=use_normalization)

    def call(self, graph):
        feat = getattr(graph, self.feature_field)
        num = getattr(graph, self.number_field)
        ind = getattr(graph, self.index_field)

        if self.is_index_of_nodes:
            nodes = utils._fixed_number_to_one(graph.nodes, indices=ind)
        else:
            num_segments = tf.reduce_sum(num)
            nodes = utils._unsorted_segments_to_one(graph.nodes,
                                                    segment_ids=ind,
                                                    num_segments=num_segments)

        globals = utils._one_to_segments(graph.globals, num)

        inputs = tf.concat([feat, nodes, globals], axis=1)
        feat_new = self.network(inputs)
        return graph.replace(**{self.feature_field: feat_new})


class HypergraphEdgeAttentionBlock(Layer):
    def __init__(self,
                 layer_sizes,
                 output_use_activation=True,
                 output_use_bias=True,
                 use_normalization=True,
                 field=hypergraph.EDGES,
                 name="hypergraph_edge_block"):
        super(HypergraphEdgeAttentionBlock, self).__init__(name=name)

        idx = hypergraph.HYPEREDGES_FEATURE_FIELDS.index(field)
        self.feature_field = hypergraph.HYPEREDGES_FEATURE_FIELDS[idx]
        self.index_field = hypergraph.HYPEREDGES_INDEX_FIELDS[idx]
        self.number_field = hypergraph.HYPEREDGES_NUMBER_FIELDS[idx]
        self.is_index_of_nodes = hypergraph.HYPEREDGES_INDEX_OF_NODES[idx]

        assert self.is_index_of_nodes

        attention_key_size = 8
        attention_head_number = 8
        self.edge2query = Dense(units=attention_key_size, use_bias=True)
        self.node2key = Dense(units=attention_key_size, use_bias=True)
        self.multihead_edge = tf.keras.layers.Conv1D(
            filters=attention_head_number,
            kernel_size=4,
            # Use 'same' padding so outputs have the same shape as inputs.
            padding='same')

        self.network = mlp_with_normalization(layer_sizes=layer_sizes,
                                              output_use_activation=output_use_activation,
                                              output_use_bias=output_use_bias,
                                              use_normalization=use_normalization)

        self.flatten = tf.keras.layers.Flatten()

    def call(self, graph):
        feat = getattr(graph, self.feature_field)
        num = getattr(graph, self.number_field)
        ind = getattr(graph, self.index_field)

        edge_queries = self.edge2query(feat)
        node_keys = self.node2key(graph.nodes)

        edge_queries_multihead = self.multihead_edge(edge_queries[:, None, :])
        node_keys_multihead = self.multihead_edge(node_keys[:, None, :])
        node_values_multihead = node_keys_multihead

        nodes_multihead = utils._fixed_number_to_one_attention(edge_queries_multihead,
                                                               node_keys_multihead,
                                                               node_values_multihead,
                                                               ind)
        nodes = self.flatten(nodes_multihead)

        globals = utils._one_to_segments(graph.globals, num)

        inputs = tf.concat([feat, nodes, globals], axis=1)
        feat_new = self.network(inputs)
        return graph.replace(**{self.feature_field: feat_new})


class HypergraphGlobalBlock(Layer):
    def __init__(self,
                 layer_sizes,
                 output_use_activation=True,
                 output_use_bias=True,
                 use_normalization=True,
                 name="hypergraph_global_block"):
        super(HypergraphGlobalBlock, self).__init__(name=name)

        self.network = mlp_with_normalization(layer_sizes=layer_sizes,
                                              output_use_activation=output_use_activation,
                                              output_use_bias=output_use_bias,
                                              use_normalization=use_normalization)

    def call(self, graph):
        input_list = [graph.globals]
        for feat_field, num_field in zip(hypergraph.HYPERGRAPH_FEATURE_FIELDS,
                                         hypergraph.HYPERGRAPH_NUMBER_FIELDS):
            if feat_field == hypergraph.GLOBALS:
                continue
            feat = getattr(graph, feat_field)
            num = getattr(graph, num_field)
            feat_agg = utils._segments_to_one(feat, num)
            input_list.append(feat_agg)
        inputs = tf.concat(input_list, axis=1)
        globals = self.network(inputs)
        return graph.replace(globals=globals)


class HypergraphNetwork(Layer):
    def __init__(self,
                 layer_sizes_dict,
                 output_use_activation=True,
                 output_use_bias=True,
                 use_normalization=True,
                 node_attention=True,
                 edge_attention=True,
                 name="hypergraph_network"):
        super(HypergraphNetwork, self).__init__(name=name)


        # Node (Vertex in hyper assocaition graph)
        node_layer_sizes = layer_sizes_dict[hypergraph.NODES]
        
        if node_attention:
            self.node_block = HypergraphNodeAttentionBlock(layer_sizes=node_layer_sizes,
                                                        output_use_activation=output_use_activation,
                                                        output_use_bias=output_use_bias,
                                                        use_normalization=use_normalization,
                                                        name="node_block")
        else:
            self.node_block = HypergraphNodeBlock(layer_sizes=node_layer_sizes,
                                                  output_use_activation=output_use_activation,
                                                  output_use_bias=output_use_bias,
                                                  use_normalization=use_normalization,
                                                  name="node_block")

        # Hyperedges
        edge_layer_sizes = layer_sizes_dict[hypergraph.EDGES]

        if edge_attention:
            self.edge_block = HypergraphEdgeAttentionBlock(layer_sizes=edge_layer_sizes,
                                                        output_use_activation=output_use_activation,
                                                        output_use_bias=output_use_bias,
                                                        use_normalization=use_normalization,
                                                        field=hypergraph.EDGES,
                                                        name="edge_block")
        else:
            self.edge_block = HypergraphEdgeBlock(layer_sizes=edge_layer_sizes,
                                                  output_use_activation=output_use_activation,
                                                  output_use_bias=output_use_bias,
                                                  use_normalization=use_normalization,
                                                  field=hypergraph.EDGES,
                                                  name="edge_block")

        # Row hyperedge
        row_layer_sizes = layer_sizes_dict[hypergraph.ROWS]
        self.row_block = HypergraphEdgeBlock(layer_sizes=row_layer_sizes,
                                             output_use_activation=output_use_activation,
                                             output_use_bias=output_use_bias,
                                             use_normalization=use_normalization,
                                             field=hypergraph.ROWS,
                                             name='row_block')

        # Column hyperedge
        col_layer_sizes = layer_sizes_dict[hypergraph.COLS]
        self.col_block = HypergraphEdgeBlock(layer_sizes=col_layer_sizes,
                                             output_use_activation=output_use_activation,
                                             output_use_bias=output_use_bias,
                                             use_normalization=use_normalization,
                                             field=hypergraph.COLS,
                                             name='col_block')

        global_layer_sizes = layer_sizes_dict[hypergraph.GLOBALS]
        self.global_block = HypergraphGlobalBlock(layer_sizes=global_layer_sizes,
                                                  output_use_activation=output_use_activation,
                                                  output_use_bias=output_use_bias,
                                                  use_normalization=use_normalization,
                                                  name="global_block")

    def call(self, graph):
        graph = self.row_block(graph)
        graph = self.col_block(graph)
        graph = self.edge_block(graph)
        graph = self.node_block(graph)
        graph = self.global_block(graph)

        return graph


class HypergraphModelTrust3(Model):
    def __init__(self,
                 num_processing_steps,
                 node_output_size,
                 hidden_layer_size=16,
                 num_hidden_layer=2,
                 node_attention=True,
                 edge_attention=True,
                 name="hypergraph_model"):

        super(HypergraphModelTrust3, self).__init__(name=name)

        self.num_processing_steps = num_processing_steps

        output_size_dict = {hypergraph.NODES: node_output_size}

        layer_sizes = [hidden_layer_size]*num_hidden_layer

        layer_sizes_dict = {field: layer_sizes for field in hypergraph.HYPERGRAPH_FEATURE_FIELDS}
        self.encoder = HypergraphIndependent(layer_sizes_dict, output_use_activation='relu', output_use_bias=True, use_normalization=True, name="encoder")

        layer_sizes_dict = {field: layer_sizes for field in hypergraph.HYPERGRAPH_FEATURE_FIELDS}
        self.core = HypergraphNetwork(layer_sizes_dict, output_use_activation='relu', output_use_bias=True, use_normalization=True, name="core", node_attention=node_attention, edge_attention=edge_attention)

        layer_sizes_dict = {field: layer_sizes for field in output_size_dict}
        self.decoder = HypergraphIndependent(layer_sizes_dict, output_use_activation='relu', output_use_bias=True, use_normalization=True, name="decoder")

        # layer_sizes_dict = {field: [output_size] for field, output_size in output_size_dict.items()}
        # self.output_transform = HypergraphIndependent(layer_sizes_dict, output_use_activation='sigmoid', output_use_bias=True, use_normalization=False, name="output")

        layer_sizes_dict = {hypergraph.NODES: [1]}
        self.tcp_classfier_layer = HypergraphIndependent(layer_sizes_dict, output_use_activation='relu', output_use_bias=True, use_normalization=False, name="tcp_clf")

        layer_sizes_dict = {hypergraph.NODES: [1]}
        # layer_sizes_dict = {field: [output_size] for field, output_size in output_size_dict.items()}
        self.tcp_confidence_layer = HypergraphIndependent(layer_sizes_dict, output_use_activation='relu', output_use_bias=True, use_normalization=False, name="tcp_conf")

        self.softmax_gmn = HypergraphOutputSoftmax(name="softmax_gmn", prob_vec=False)

        self.softmax_reverse = HypergraphOutputNormalByRow(name="softmax_reverse")
        self.softmax_tcp = HypergraphOutputSoftmax(name="softmax_tcp", prob_vec=True)

    def call(self, graph):
        graph = HypergraphsTuple(*graph)

        # make it symmetric
        edges = graph.edges                                        # [N, edge_dim]
        edge_dim = edges.shape[-1]

        # repeat edge features
        order = graph.hyperedges.shape[-1]
        dim = edge_dim // (2 * order)
        edges = tf.reshape(edges, [-1, 2, order, dim])             # [N, 2, order, dim]
        indices = list(itertools.permutations(range(order)))       # [P, order], where P=order!
        edges = tf.gather(edges, indices, axis=2)                  # [N, 2, n_perm, order, dim]
        edges = tf.transpose(edges, perm=[0, 2, 1, 3, 4])          # [N, n_perm, 2, order, dim]
        edges = tf.reshape(edges, [-1, edge_dim])                  # [N*n_perm, edge_dim]
        n_perm = len(indices)

        permuted_graph = graph.replace(edges=edges)

        latent = self.encoder(permuted_graph)

        edges = latent.edges

        # average edge features
        edge_dim_encoder = edges.shape[-1]
        edges = tf.reshape(edges, [-1, n_perm, edge_dim_encoder])
        edges = tf.reduce_mean(edges, axis=1)                      # [N, edge_dim_encoder]

        # graph convolution
        latent = latent.replace(edges=edges)

        latent0 = latent
        output_graphs = []
        output_confs = []
        output_preds = []

        for step in range(self.num_processing_steps):                  # GCN
            core_input = concat_graphs([latent0, latent])
            latent = self.core(core_input)
            decoded_latent = self.decoder(latent)
            
            tcp_conf = self.tcp_confidence_layer(decoded_latent) # [N, 1]

            
            tcp_logit = self.tcp_classfier_layer(decoded_latent) 
            output_graph = utils.normalize_rows_for_graph(tcp_logit, method='softmax')
            pred_tcp = self.softmax_reverse(output_graph)
            # pred_tcp = self.softmax_tcp(pred_tcp)
            
            output_graphs.append(output_graph)
            output_confs.append(tcp_conf)
            output_preds.append(pred_tcp)

        return output_graphs, output_confs, output_preds
