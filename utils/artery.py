import tensorflow as tf
import numpy as np
import os
import networkx as nx
import pickle
import cv2
import itertools


from glob import glob
from tqdm import tqdm
from methods.HNN_HM.hnn_hm.hypergraph import HypergraphsTuple
from methods.HNN_HM.hnn_hm import hypergraph
from methods.HNN_HM.hnn_hm.hypergraph_utils_tf import set_zero_feature, merge_graphs, concat_attributes

ARTERY_CATEGORY = ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]
SEMANTIC_MAPPING = {"OTHER": [255, 0, 0], "LAD": [255, 255, 0], "LCX": [102, 255, 102], "LMA": [0, 102, 255], "D": [255, 0, 255], "OM": [102, 255, 255]}
SUB_BRANCH_CATEGORY = ["LMA", "LAD1", "LAD2", "LAD3", "LCX1", "LCX2", "LCX3", "D1", "D2", "OM1", "OM2", "OM3"]
MAIN_BRANCH_CATEGORY = ["LMA", "LAD", "LCX", "D", "OM"]
POSSIBLE_NEIGHBORS = {"LMA": ["LAD1", "LCX1"], 
                     "LAD1": ["LMA", "LCX1", "D1", "LAD2"], 
                     "LAD2": ["LAD1", "D1", "D2", "LAD3"],
                     "LAD3": ["LAD2", "D2"],
                     "LCX1": ["LMA", "LAD1", "LCX2", "OM1"],
                     "LCX2": ["LCX1", "OM1", "OM2", "LCX3"],
                     "LCX3": ["LCX2", "OM2"],
                     "OM1": ["LCX1", "LCX2"], "OM2": ["LCX2", "LCX3"],
                     "D1": ["LAD1", "LAD2"], "D2": ["LAD2", "LAD3"]}


def _get_sample_list(data_file_path, category):
    pkl_file_paths = glob(f"{data_file_path}/*binary_image.png")
    samples = []
    for pkl_file_path in pkl_file_paths:
        sample_name = pkl_file_path[pkl_file_path.rfind("/")+1: pkl_file_path.rfind("_binary_image.png")]
        if category == "":
            samples.append(sample_name)
        else:
            if sample_name.rfind(category) !=-1:
                samples.append(sample_name)
    return samples


def _load_graph(data_file_path, sample_id):
    image = cv2.imread(os.path.join(data_file_path, f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
    binary_image = cv2.imread(os.path.join(data_file_path, f"{sample_id}_binary_image.png"), cv2.IMREAD_GRAYSCALE)
    pkl_file_path = os.path.join(data_file_path, f"{sample_id}.pkl")
    g = pickle.load(open(pkl_file_path, 'rb'))
    return image, binary_image, g


def _load_graph_in_mem(data_file_path, sample_id="", dataset=None):
    if dataset is None:
        # load all samples from hd
        sample_list = _get_sample_list(data_file_path, "")
        data = {}
        # this will load all data into memory, for 300 subject, costs about 10GB RAM
        print(f"Artery._load_graph_in_mem, loading all data, len(sample_list) = {len(sample_list)}")
        for sample_name in tqdm(sample_list):
            image, bin, g = _load_graph(data_file_path, sample_name)
            data[sample_name] = {"image": image, "binary_image": bin, "g": g}
        return data, sample_list
    else:
        return dataset[sample_id]['image'], dataset[sample_id]['binary_image'], dataset[sample_id]['g']
    

def _gen_random_graph_in_mem(rand, category_id, dataset, cache=True, cache_path="./cache"):
    """
    similary to utils.make_matching_problem_synthetic._generate_point_sets
    return:
        P1: features for each artery branch (node) in G1
        P2: features for each artery branch (node) in G2
        gX: assignemtn matrix with dimension of n1*n2
    """

    def __switch__(sample_idx):
        sample_name0, sample_name1 = sample_list[sample_idx[0]], sample_list[sample_idx[1]]
        _, _, g0 = dataset[sample_name0]['image'], dataset[sample_name0]['binary_image'], dataset[sample_name0]['g']
        _, _, g1 = dataset[sample_name1]['image'], dataset[sample_name1]['binary_image'], dataset[sample_name1]['g']

        n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()

        if n0 >= n1:
            return [sample_idx[1], sample_idx[0]]
        else:
            return sample_idx

    all_sample_list = list(dataset.keys())
    sample_list = []
    for sample_name in all_sample_list:
        if category_id == -1:
            sample_list.append(sample_name)
        else:
            if sample_name.rfind(ARTERY_CATEGORY[category_id]) != -1:
                sample_list.append(sample_name)

    sample_idx = rand.randint(0, len(sample_list), size=2)
    sample_idx = __switch__(sample_idx)

    g1 = dataset[sample_list[sample_idx[0]]]['g']
    g2 = dataset[sample_list[sample_idx[1]]]['g']

    if cache:
        if os.path.isfile(f"{cache_path}/{sample_list[sample_idx[0]]}_{sample_list[sample_idx[1]]}.pkl"):
            ahg = pickle.load(open(f"{cache_path}/{sample_list[sample_idx[0]]}_{sample_list[sample_idx[1]]}.pkl", "rb"))
            return ahg["P1"], ahg["P1"], ahg["assignmentMatrix"], g1, g2, [sample_list[sample_idx[0]], sample_list[sample_idx[1]]], ahg
    
    n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
    gX = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            if g1.nodes()[i]['data'].vessel_class == g2.nodes()[j]['data'].vessel_class:
                gX[i, j] = 1.0

    p1, p2 = [], []
    for i in range(n1):
        p1.append(g1.nodes()[i]['data'].features)

    for i in range(n2):
        p2.append(g2.nodes()[i]['data'].features)

    p1 = np.array(p1)
    p2 = np.array(p2)
    return p1, p2, gX, g1, g2, [sample_list[sample_idx[0]], sample_list[sample_idx[1]]], None


def _find_all_subgraph(g, n_node=3):
    # all_connected_subgraphs = []
    all_connected_subgraphs_node_list = []

    # here we ask for all connected subgraphs that have at least 2 nodes AND have less nodes than the input graph
    for nb_nodes in range(2, g.number_of_nodes()):
        for SG in (g.subgraph(selected_nodes) for selected_nodes in itertools.combinations(g, nb_nodes)):
            if nx.is_connected(SG) and len(SG.nodes)==n_node:
                # print(SG.nodes)
                # all_connected_subgraphs.append(SG)
                all_connected_subgraphs_node_list.append(list(SG.nodes))

    return all_connected_subgraphs_node_list


def create_association_hypergraph(rand, category_id, dataset, cache=True, cache_path=".cache"):
    p1, p2, assignmentMatrix, g1, g2, samples, ahg = _gen_random_graph_in_mem(rand, category_id=category_id, dataset=dataset, cache=cache, cache_path=cache_path)
    if ahg:
        return ahg
    else:
        ahg = _create_hyperedges(g1, g2)
        ahg["P1"] = p1
        ahg["P2"] = p2
        ahg["assignmentMatrix"] = assignmentMatrix
        ahg["samples"] = samples

        if cache:
            pickle.dump(ahg, open(f"{cache_path}/{samples[0]}_{samples[1]}.pkl", "wb"))
        return ahg


def create_association_hypergraph_test(sample_name1, sample_name2, dataset1, dataset2, cache=True, cache_path=".cache"):
    """
    create association hypergraph for model testing
    sample_name1 and dataset1 are from testset
    sample_name2 and dataset2 are from template set
    """

    if cache:
        if os.path.isfile(f"{cache_path}/{sample_name1}_{sample_name2}.pkl"):
            ahg = pickle.load(open(f"{cache_path}/{sample_name1}_{sample_name2}.pkl", "rb"))
            return ahg
        
    _, _, g1 = _load_graph_in_mem("", sample_name1, dataset1)
    _, _, g2 = _load_graph_in_mem("", sample_name2, dataset2)

    n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
    gX = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            if g1.nodes()[i]['data'].vessel_class == g2.nodes()[j]['data'].vessel_class:
                gX[i, j] = 1.0

    p1, p2 = [], []
    for i in range(n1):
        p1.append(g1.nodes()[i]['data'].features)

    for i in range(n2):
        p2.append(g2.nodes()[i]['data'].features)

    p1 = np.array(p1)
    p2 = np.array(p2)

    problem = _create_hyperedges(g1, g2)
    problem["P1"] = p1
    problem["P2"] = p2
    problem["assignmentMatrix"] = gX
    problem["samples"] = [sample_name1, sample_name2]

    return problem


def _create_hyperedges(g0: nx.Graph, g1: nx.Graph, cache=True, cache_path=".cache"):
    hyperedges = []

    num_nodes0 = g0.number_of_nodes()
    num_nodes1 = g1.number_of_nodes()
    num_matches = num_nodes0*num_nodes1
    feature_per_node = len(g0.nodes()[0]['data'].features)

    # the index of the node index in the vertex in association graph
    gidx0 = np.zeros(num_matches, np.int) 
    gidx1 = np.zeros(num_matches, np.int)
    for i in range(num_matches):
        gidx0[i] = i / num_nodes1
        gidx1[i] = i % num_nodes1

    hyper_edges_g0, hyper_edges_g1 = _find_all_subgraph(g0, 3), _find_all_subgraph(g1, 3)
    vertex_features = np.zeros((num_matches, feature_per_node*2), np.float)
    hyperedge_features = np.zeros((len(hyper_edges_g0)*len(hyper_edges_g1), feature_per_node*6), np.float)
    vertex_labels = []

    # assign graph vertex features
    for i in range(num_matches):
        feat_node0 = np.array(g0.nodes()[gidx0[i]]['data'].features)
        feat_node1 = np.array(g1.nodes()[gidx1[i]]['data'].features)
        vertex_features[i] = np.hstack((feat_node0, feat_node1))
        vertex_labels.append((g0.nodes()[gidx0[i]]['data'].vessel_class, g1.nodes()[gidx1[i]]['data'].vessel_class))

    idx = 0
    for i in range(len(hyper_edges_g0)):
        for j in range(len(hyper_edges_g1)):
            idx_g0_node0, idx_g0_node1, idx_g0_node2 = hyper_edges_g0[i][0], hyper_edges_g0[i][1], hyper_edges_g0[i][2]
            idx_g1_node0, idx_g1_node1, idx_g1_node2 = hyper_edges_g1[j][0], hyper_edges_g1[j][1], hyper_edges_g1[j][2]
            gh_node0 = idx_g0_node0*num_nodes1 + idx_g1_node0
            gh_node1 = idx_g0_node1*num_nodes1 + idx_g1_node1
            gh_node2 = idx_g0_node2*num_nodes1 + idx_g1_node2

            feat_g0_node0 = np.array(g0.nodes()[idx_g0_node0]['data'].features)
            feat_g0_node1 = np.array(g0.nodes()[idx_g0_node1]['data'].features)
            feat_g0_node2 = np.array(g0.nodes()[idx_g0_node2]['data'].features)
            feat_g1_node0 = np.array(g1.nodes()[idx_g1_node0]['data'].features)
            feat_g1_node1 = np.array(g1.nodes()[idx_g1_node1]['data'].features)
            feat_g1_node2 = np.array(g1.nodes()[idx_g1_node2]['data'].features)
            hyperedges.append([gh_node0, gh_node1, gh_node2])
            hyperedge_features[idx] = np.hstack((feat_g0_node0, feat_g0_node1, feat_g0_node2, feat_g1_node0, feat_g1_node1, feat_g1_node2))
            idx += 1

    hyperedges = np.array(hyperedges, dtype=np.int32)
    hyperedge_features = np.array(hyperedge_features, dtype=np.float32)
    vertex_features = np.array(vertex_features, dtype=np.float32)

    problem = {"cid": None, "nP1": num_nodes0, "nP2": num_nodes1, "feature_per_node": feature_per_node,
               "hyperedges": hyperedges, "hyperedge_features": hyperedge_features, "vertex_labels": vertex_labels, "vertex_features": vertex_features}
    
    return problem


class ArteryGraphCreator:

    def __init__(self, cfg):
        print("ArteryGraphCreator.__init__")
        self.input_dtype = cfg.INPUT_DTYPE
        self.input_shape = cfg.INPUT_SHAPE
        self.input_signature = cfg.INPUT_SIGNATURE

        self.target_dtype = cfg.TARGET_DTYPE
        self.target_shape = cfg.TARGET_SHAPE
        self.target_signature = cfg.TARGET_SIGNATURE

        # # TODO: Is it right to define `repeat_hyperedge` here?
        # self.default_repeat_hyperedge = cfg.REPEAT_HYPEREDGE

        # self.default_normalize_point_set = cfg.NORMALIZE_POINT_SET

    def get_input_graph(self, problem):
        hyperedges = tf.convert_to_tensor(problem["hyperedges"], dtype=tf.int32) #vertex index of hyperedges, stored in tuple
        hyperedge_features = tf.convert_to_tensor(problem['hyperedge_features'], dtype=tf.float32)
        vertex_features = tf.convert_to_tensor(problem["vertex_features"], dtype=tf.float32)
        
        n_vertex = tf.convert_to_tensor([problem['nP1']*problem['nP2']], dtype=tf.int32)
        n_edge = tf.convert_to_tensor([hyperedges.shape[0]], dtype=tf.int32)
        n_row = tf.convert_to_tensor([problem['nP1']], dtype=tf.int32)
        n_col = tf.convert_to_tensor([problem['nP2']], dtype=tf.int32)
        nrow = n_row[0]
        ncol = n_col[0]
        row_id = tf.repeat(tf.range(nrow), repeats=ncol, axis=0)
        col_id = tf.tile(tf.range(ncol), multiples=[nrow])

        n_global = tf.convert_to_tensor([1], dtype=tf.int32)

        graph = HypergraphsTuple(nodes=vertex_features, n_node=n_vertex,
                                 edges=hyperedge_features, n_edge=n_edge, hyperedges=hyperedges,
                                 rows=None, n_row=n_row, row_id=row_id,
                                 cols=None, n_col=n_col, col_id=col_id,
                                 globals=None, n_global=n_global)
        size_dict = {}
        for k in hypergraph.HYPERGRAPH_FEATURE_FIELDS:
            if getattr(graph, k) is None:
                size_dict[k] = getattr(HypergraphsTuple(*self.input_shape), k)[1:]
        graph = set_zero_feature(graph, size_dict)

        return graph
    
    def get_target_graph(self, input_graph, assignment_matrix):
        assignment_matrix = tf.cast(assignment_matrix, tf.float32)  # TODO: user-specific dtype
        true_node = tf.reshape(assignment_matrix, [-1, 1])
        return input_graph.replace(nodes=true_node)

    def create_data(self, problem_list):
        input_graph_list = []
        target_graph_list = []
        for problem in problem_list:
            input_graph = self.get_input_graph(problem)
            target_graph = self.get_target_graph(input_graph, problem["assignmentMatrix"])
            input_graph_list.append(input_graph)
            target_graph_list.append(target_graph)
        inputs = merge_graphs(input_graph_list)

        key = hypergraph.NODES
        target_nodes = concat_attributes(target_graph_list, keys=[key], axis=0)[key]
        targets = inputs.replace(nodes=target_nodes)

        return inputs, targets
    

class ArteryProblemGeneratorTest:

    def __init__(self, rng, dataset_test, dataset_template, cache=True, cache_path=""):
        self.rng = rng
        self.dataset_test = dataset_test
        self.dataset_template = dataset_template
        self.cache = cache
        self.cache_path = cache_path
    
    def generate_problem(self, test_subject, template_subject):
        problem = create_association_hypergraph_test(test_subject, template_subject, self.dataset_test, self.dataset_template, 
                                                     cache=self.cache, cache_path=self.cache_path)
        return problem


class DataLoader:
    def __init__(self, problem_generator, graph_creator):
        self.input_dtype = graph_creator.input_dtype
        self.input_shape = graph_creator.input_shape

        self.target_dtype = graph_creator.target_dtype
        self.target_shape = graph_creator.target_shape

        self.problem_generator = problem_generator
        self.graph_creator = graph_creator

    def generate_data(self, batch_size, category_id):
        problem_list = self.problem_generator.generate_problems(batch_size, category_id)
        return self.graph_creator.create_data(problem_list)
    
    def generate_data_test(self, sample_name1, sample_name2):
        problem_list = [self.problem_generator.generate_problem(sample_name1, sample_name2)]
        inputs, targets = self.graph_creator.create_data(problem_list)
        return inputs, targets, problem_list[0]
    