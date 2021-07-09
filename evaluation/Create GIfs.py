import numpy as np
import matplotlib.pyplot as plt
from styletransfer.predict import prepare, generate, prepare_single_style,generate_single_style
from matplotlib.animation import FuncAnimation

import os

print(os.getcwd())

loaded = np.load("runs/04-05 test fixed to 40 fps for visualization/40fps_for_GIF.npz")
print(loaded["frames"][0].shape)

model = prepare_single_style(name="road_FastCUT", mode="FastCUT")

FAST_CUT_converted_frame = []
org_frame = []
step = []
for n in range(0, 1000, 10):
    new_frame = loaded["frames"][n][:256, :256]
    org_frame.append(new_frame)
    new_frame = generate_single_style(model, new_frame)
    FAST_CUT_converted_frame.append(new_frame)
    step.append(n)
org_frame = np.array(org_frame)
FAST_CUT_converted_frame = np.array(FAST_CUT_converted_frame)

model = prepare(name="road_CUT", mode="CUT")
CUT_converted_frame = []
for n in range(0, 1000, 10):
    new_frame = loaded["frames"][n][:256, :256]
    new_frame = generate(model, new_frame)
    CUT_converted_frame.append(new_frame)
CUT_converted_frame = np.array(CUT_converted_frame)

# fig, axarr = plt.subplots(1,3,figsize=(15,5))

fig, axarr = plt.subplots(3, 1, figsize=(5, 15))
fig.set_tight_layout(True)

n = 0
axarr[0].imshow(org_frame[n])
axarr[0].axis("off")
axarr[0].set_title("OpenAI original")

axarr[1].imshow(FAST_CUT_converted_frame[n])
axarr[1].axis("off")
axarr[1].set_title("Style transfer: FastCUT")

axarr[2].imshow(CUT_converted_frame[n])
axarr[2].axis("off")
axarr[2].set_title("Style transfer: CUT")

fig.suptitle(f"Style transfer for Frame {n}, (256x256)")


# for n in range(org_frame.shape[0]):
def update(n):
    # fig, axarr = plt.subplots(1,2)
    # fig.set_tight_layout(True)

    axarr[0].imshow(org_frame[n])
    axarr[0].axis("off")
    axarr[0].set_title("OpenAI original")

    axarr[1].imshow(FAST_CUT_converted_frame[n])
    axarr[1].axis("off")
    axarr[1].set_title("Style transfer: FAST_CUT")

    axarr[2].imshow(CUT_converted_frame[n])
    axarr[2].axis("off")
    axarr[2].set_title("Style transfer: CUT")
    fig.suptitle(f"Frame {step[n]}")
    return fig, axarr


anim = FuncAnimation(fig, update, frames=np.arange(0, org_frame.shape[0]), interval=400)
# anim.save('Gif_files/vertical_FAST_CUT_and_CUT_slow.gif', dpi=80, writer='imagemagick')
plt.show()
