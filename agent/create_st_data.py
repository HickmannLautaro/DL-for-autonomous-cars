from tqdm import trange, tqdm
import classifier.AD_test as AD_test
import numpy as np
import io
from contextlib import redirect_stdout

def main():
    selcted_models = ['out/old_dataset/LeNet_mod_bz_128_lr_0.0001_ep_30/model_ep_3.pth',
                      'out/old_dataset/Nvidia_bar_model_resized_cons_no_bn_bz_256_lr_0.0001_ep_30/model_ep_0.pth',
                      'out/new_dataset/Nvidia_model_bz_128_lr_0.001_ep_30/model_ep_9.pth',
                      'out/new_dataset/Nvidia_bar_model_resized_cons_bz_128_lr_0.0001_ep_30/model_ep_27.pth',
                      'out/new_dataset/style_transfer/Nvidia_model_bz_128_lr_0.001_ep_30/style_green/model_ep_24.pth',
                      'out/new_dataset/style_transfer/Nvidia_model_bz_128_lr_0.001_ep_30/style_brown/model_ep_6.pth',
                      'out/new_dataset/style_transfer/Nvidia_model_bz_128_lr_0.001_ep_30/style_mixed/model_ep_0.pth']





    st_model_vals = []
    for mod in tqdm(selcted_models, desc="OpenAI Models"):
        rew_list = []
        if "bar" in mod:
            dim = 288
        else:
            dim = 256
        for i in trange(10, desc=f"Simulation runs {mod}", leave=False):
            f = io.StringIO()
            with redirect_stdout(f):
                rew, _ = AD_test.main(False, mod, None, None, 256)
            rew_list.append(rew)
        rew_list = np.array(rew_list)
        st_model_vals.append([mod, rew_list])

    style_0 = []
    for mod in tqdm(selcted_models, desc="Style 0 Models"):
        rew_list = []
        if "bar" in mod:
            dim = 288
        else:
            dim = 256
        for i in trange(10, desc=f"Simulation runs {mod}", leave=False):
            f = io.StringIO()
            with redirect_stdout(f):
                rew, _ = AD_test.main(False, mod, None, None, dim, True, 1, 0)
            rew_list.append(rew)
        rew_list = np.array(rew_list)
        style_0.append([mod, rew_list])

    style_1 = []
    for mod in tqdm(selcted_models, desc="Style 1 Models"):
        rew_list = []
        if "bar" in mod:
            dim = 288
        else:
            dim = 256
        for i in trange(10, desc=f"Simulation runs {mod}", leave=False):
            f = io.StringIO()
            with redirect_stdout(f):
                rew, _ = AD_test.main(False, mod, None, None, dim, True, 0, 1)
            rew_list.append(rew)
        rew_list = np.array(rew_list)
        style_1.append([mod, rew_list])

    np.savez("out/new_dataset/style_transfer/st_eval_on_generated", style_0=style_0, style_1=style_1, st_model_vals=st_model_vals)

if __name__ == "__main__":
    main()