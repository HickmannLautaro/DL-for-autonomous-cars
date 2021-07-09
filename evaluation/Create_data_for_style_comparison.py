import glob
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm, trange

sys.path.insert(0, './styletransfer/cut/models/base_model')
from styletransfer.predict import prepare, generate, generate_edges_only, prepare_edges_only


import re

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# model_list_single=["road_FastCUT"]

model_list_edges = ["new_trees/edges/edge_detection_BCE",
                    "new_trees/edges/edge_detection_MSE",
                    "new_trees/edges/edge_detection_SSIM"]  # ["dual_random_background_MSE", "dual_random_background_SSIM", "dual_random_background_BCE"]
model_list_edges = []

losses = ["BCE","MSE", "SSIM"]
model_list_dual = ["new_trees/styletransfer_histo",
                   "new_trees/styletransfer_BCE_histo_10_edges_5",
                   "new_trees/styletransfer_BCE_histo_10_edges_10",
                   "new_trees/styletransfer_MSE_histo_10_edges_5",
                   "new_trees/styletransfer_MSE_histo_10_edges_10",
                   "new_trees/styletransfer_SSIM_histo_10_edges_5",
                   "new_trees/styletransfer_SSIM_histo_10_edges_10"]  # ["dual_color_hist_short" ]
model_list_dual = ['dual_color_hist_tree']

# ["dual_simple","dual_all_the_data_30","dual_all_the_data_60","dual_all_the_data_100", "dual_all_the_data_150","dual_all_the_data_200",
#                 "dual_all_the_data_short_25","dual_all_the_data_short_50","dual_all_the_data_short_75","dual_all_the_data_short_100" ]

save_path = "files_for_style_comparisson/"
save_path = "classifier"
save_name = "Grad"#'continuous_trees_test' #'new_trees_train'#f'create_date_{str(datetime.now())}'
# loaded = np.load("runs/2021-06-13 08:11:56.351191/discrete_histories.npz")  # "runs/04-05 test fixed to 40 fps for visualization/40fps_for_GIF.npz")
# frames = loaded["frames"][:, :256, :256, :]
images = glob.glob('styletransfer/cut/datasets/road_tree_continuous_test/trainA/*') #glob.glob('styletransfer/cut/datasets/road_tree_new_test/trainA/*')#glob.glob('styletransfer/cut/datasets/road_tree_new_train/trainA/*') #
images = natural_sort(images) # simulate continuous driving


frames = np.array([np.asarray(Image.open(fname)) for fname in images])
loaded = {"frames": frames}
loop_step = 1
list_of_expes = []
org_frame = []
step = []
print("extracting simulated")
for n in trange(0, frames.shape[0]):
    new_frame = loaded["frames"][n][:256, :256]
    org_frame.append(new_frame)
    step.append(n)
org_frame = np.array(org_frame)
steps = np.array(step)

list_of_expes.append(org_frame)

dup_names = []
if len(model_list_edges) > 0:
    list_of_edges = []
    edges_names = []
    print("Extract only edge data")
    for models, loss in tqdm(zip(model_list_edges, losses)):
        model = prepare_edges_only(name=models, loss=loss)
        converted_frame_class_0 = []
        converted_frame_class_1 = []
        for n in trange(0, frames.shape[0], loop_step):
            new_frame = loaded["frames"][n][:256, :256]
            new_frame_0 = generate_edges_only(model, new_frame, a=1, b=0)
            new_frame_1 = generate_edges_only(model, new_frame, a=0, b=1)
            converted_frame_class_0.append(new_frame_0)
            converted_frame_class_1.append(new_frame_1)
        converted_frame_class_0 = np.array(converted_frame_class_0)
        converted_frame_class_1 = np.array(converted_frame_class_1)
        list_of_edges.append(converted_frame_class_0)
        list_of_edges.append(converted_frame_class_1)
    _ = [[edges_names.append(a + "_0"), edges_names.append(a + "_1")] for a in model_list_edges]

if len(model_list_dual) > 0:
    print("Extract dual styles")
    for models in tqdm(model_list_dual):
        model = prepare(name=models, mode="FastCUT")
        converted_frame_class_0 = []
        converted_frame_class_1 = []
        for n in trange(0, frames.shape[0], loop_step):
            new_frame = loaded["frames"][n][:256, :256]
            new_frame_0 = generate(model, new_frame, a=1, b=0)
            new_frame_1 = generate(model, new_frame, a=0, b=1)
            converted_frame_class_0.append(new_frame_0)
            converted_frame_class_1.append(new_frame_1)
        converted_frame_class_0 = np.array(converted_frame_class_0)
        converted_frame_class_1 = np.array(converted_frame_class_1)
        list_of_expes.append(converted_frame_class_0)
        list_of_expes.append(converted_frame_class_1)
    _ = [[dup_names.append(a + "_0"), dup_names.append(a + "_1")] for a in model_list_dual]

print("Saving")
if len(model_list_edges) > 0:
    np.savez_compressed(f'{save_path}/{save_name}', steps=steps, names=np.array(["org"] + dup_names), frames=list_of_expes, edges=list_of_edges, edge_names=edges_names)
else:
    np.savez_compressed(f'{save_path}/{save_name}', steps=steps, names=np.array(["org"] + dup_names), frames=list_of_expes )
# for models in model_list_single:
#     model = prepare_single_style(name=models, mode="FastCUT")
#
# FAST_CUT_converted_frame = []
# org_frame = []
# step = []
# for n in range(0, 1000, 10):
#     new_frame = loaded["frames"][n][:256, :256]
#     org_frame.append(new_frame)
#     new_frame = generate_single_style(model, new_frame)
#     FAST_CUT_converted_frame.append(new_frame)
#     step.append(n)
# org_frame = np.array(org_frame)
# FAST_CUT_converted_frame = np.array(FAST_CUT_converted_frame)
#
# model = prepare(name="road_CUT", mode="CUT")
# CUT_converted_frame = []
# for n in range(0, 1000, 10):
#     new_frame = loaded["frames"][n][:256, :256]
#     new_frame = generate(model, new_frame)
#     CUT_converted_frame.append(new_frame)
# CUT_converted_frame = np.array(CUT_converted_frame)
