import numpy as np
import networkx as nx
import os
import cv2


from .hungarian import hungarian
from .artery import SUB_BRANCH_CATEGORY, MAIN_BRANCH_CATEGORY, SEMANTIC_MAPPING


def compute_accuracy_artery_node2(gt, output, dataset, ahg, save_path, plot=True):
    output = hungarian(output)

    total = np.sum(gt)
    matched, unmatched = 0, 0
    mapping = {}

    for i in range(gt.flatten().shape[0]):
        if gt.flatten()[i] == 1 and output.flatten()[i] == 1:
            # MATCHED
            start_vessel_class = ahg['vertex_labels'][i][0]
            target_vessel_class = ahg['vertex_labels'][i][1]
            matched += 1
            mapping[start_vessel_class] = target_vessel_class
        elif gt.flatten()[i] == 0 and output.flatten()[i] == 1:
            # NOT MATCHED
            unmatched += 1
            start_vessel_class = ahg['vertex_labels'][i][0]
            target_vessel_class = ahg['vertex_labels'][i][1]
            mapping[start_vessel_class] = target_vessel_class
    if plot:
        plot_match(dataset, ahg["samples"], output, ahg, SEMANTIC_MAPPING, save_path, thickness=1)

    return mapping, total, matched, unmatched, output



def visualize_semantic_cv2(graph: nx.Graph, original_image, semantic_mapping, save_path):
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    positions = {}
    for node in list(graph.nodes):
        vessel_obj = graph.nodes[node]['data']
        # plot points
        # visualize artery centerlines
        vessel_class = ''.join(
            [i for i in vessel_obj.vessel_class if not i.isdigit()])
        original_image[np.where(vessel_obj.vessel_centerline == 1)[0],
                       np.where(vessel_obj.vessel_centerline == 1)[1], :] = semantic_mapping[vessel_class][::-1]

        label_x = np.where(vessel_obj.vessel_centerline == 1)[1][
            len(np.where(vessel_obj.vessel_centerline == 1)[1]) // 2]
        label_y = np.where(vessel_obj.vessel_centerline == 1)[0][
            len(np.where(vessel_obj.vessel_centerline == 1)[0]) // 2]
        # print(f"{label_x},{label_y}")
        original_image = cv2.putText(original_image, vessel_obj.vessel_class, (label_x, label_y),
                                     fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                                     color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        positions[vessel_obj.vessel_class] = (label_x, label_y)

    cv2.imwrite(save_path, original_image)
    return original_image, positions


def plot_match(dataset, sample_names, output, ahg, semantic_mapping, save_path, thickness=2):
    gt = ahg['assignmentMatrix']
    output = np.squeeze(output)
    image_0, g_0 = dataset[sample_names[0]]['image'], dataset[sample_names[0]]['g']
    image_1, g_1 = dataset[sample_names[1]]['image'], dataset[sample_names[1]]['g']
    g0_save_path = os.path.join(save_path, f"{sample_names[0]}.png")
    g1_save_path = os.path.join(save_path, f"{sample_names[1]}.png")

    # start_positions = visualize_semantic_image_unique(g_0, image_0, semantic_mapping, g0_save_path)
    # end_positions = visualize_semantic_image_unique(g_1, image_1, semantic_mapping, g1_save_path)

    color_im0, start_positions = visualize_semantic_cv2(g_0, image_0, semantic_mapping, g0_save_path)
    color_im1, end_positions = visualize_semantic_cv2(g_1, image_1, semantic_mapping, g1_save_path)

    # im0 = cv2.imread(g0_save_path, cv2.IMREAD_COLOR)
    # im1 = cv2.imread(g1_save_path, cv2.IMREAD_COLOR)
    # assert im0.shape[0] == im1.shape[1]
    im = np.zeros([color_im0.shape[0], color_im0.shape[1]+color_im1.shape[1], 3], dtype=np.uint8)
    im[:, 0:color_im0.shape[1], :] = color_im0
    im[:, color_im1.shape[1]:, :] = color_im1

    match_file_name = f"{save_path}/{sample_names[0]}<->{sample_names[1]}_match.png"


    for i in range(gt.flatten().shape[0]):
        if gt.flatten()[i] == 1 and output.flatten()[i] == 1:
            # MATCHED
            vessel_class = ahg['vertex_labels'][i][0]
            start_pos = start_positions[vessel_class]
            end_pos = end_positions[vessel_class]
            end_pos = (end_pos[0]+color_im1.shape[0], end_pos[1])
            im = cv2.line(im, start_pos, end_pos, (0, 255, 0), thickness) # GREEN

        elif gt.flatten()[i] == 0 and output.flatten()[i] == 1:
            # NOT MATCHED
            vessel_class_left = ahg['vertex_labels'][i][0]
            vessel_class_right = ahg['vertex_labels'][i][1]
            start_pos = start_positions[vessel_class_left]
            end_pos = end_positions[vessel_class_right]
            end_pos = (end_pos[0] + color_im1.shape[0], end_pos[1])
            im = cv2.line(im, start_pos, end_pos, (0, 0, 255), thickness) # RED

    cv2.imwrite(match_file_name, im)
