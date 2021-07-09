import sys
import traceback

import numpy as np
from predict import prepare, generate_no_conv
from tqdm import tqdm, trange
import os


def get_train_data(name, dim=256):
    print("[INFO] Start training phase...")
    # define the train and val splits
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 1 - TRAIN_SPLIT

    # load the KMNIST dataset
    print("[INFO] loading the dataset...")

    data = np.load(f"./runs/New_runs_with_car_corrected/{name}.npz", mmap_mode='r')
    train_labels = data['input_history_train']
    train_frames = data['train_frames'][:, :, :dim, :]

    print("equalized_frames Org", train_frames.shape, "min", train_frames.min(), "max", train_frames.max())

    if not os.path.exists("classifier/out/tmp/"):
        os.makedirs("classifier/out/tmp/")

    train_frames_mem_0 = np.memmap("classifier/out/tmp/train_data_0", dtype=float, mode='w+', shape=(train_frames.shape[0], 3, 256, 256))
    train_frames_mem_1 = np.memmap("classifier/out/tmp/train_data_1", dtype=float, mode='w+', shape=(train_frames.shape[0], 3, 256, 256))
    train_frames_mem_0[:, :, :, :] = train_frames[:, :, :, :]
    train_frames_mem_0.flush()
    train_frames_mem_1.flush()

    return train_frames_mem_0, train_frames_mem_1, train_labels, train_frames.shape[0]


def get_test_data(name, dim=256):
    print("[INFO] loading the dataset...")
    data = np.load(f"./runs/New_runs_with_car_corrected/{name}.npz", mmap_mode='r')

    test_labels = data['input_history_test']

    test_frames = data['test_frames'][:,:, :dim, :]

    print("equalized_frames Org test", test_frames.shape, "min", test_frames.min(), "max", test_frames.max())

    print("[INFO] converting to pytorch dataset...")

    test_frames_mem_0 = np.memmap("classifier/out/tmp/test_data_0", dtype=float, mode='w+', shape=(test_frames.shape[0], 3, 256, 256))
    test_frames_mem_1 = np.memmap("classifier/out/tmp/test_data_1", dtype=float, mode='w+', shape=(test_frames.shape[0], 3, 256, 256))

    test_frames_mem_0[:, :, :,:] = test_frames[:, :, :,:]
    test_frames_mem_0.flush()

    test_frames_mem_1.flush()

    return test_frames_mem_0, test_frames_mem_1, test_labels, test_frames.shape[0]


def predict(load_fun):
    converted_frame_class_0, converted_frame_class_1, labels, samples = load_fun

    print("Extract dual styles")
    model = prepare(name="dual_color_hist_tree", mode="FastCUT")

    for n in trange(0, samples):
        new_frame = converted_frame_class_0[n, :, :256, :256]
        converted_frame_class_0[n] = generate_no_conv(model, new_frame, a=1, b=0)
        converted_frame_class_1[n] = generate_no_conv(model, new_frame, a=0, b=1)
    converted_frame_class_0.flush()
    converted_frame_class_1.flush()
    return converted_frame_class_0, converted_frame_class_1, labels


def mix_styles(converted_frame_class_0, converted_frame_class_1, name, labels):
    mixed_frames_mem = np.memmap(f"classifier/out/tmp/{name}_mixed", dtype=float, mode='w+', shape=(converted_frame_class_0.shape[0] * 2, converted_frame_class_0.shape[1], converted_frame_class_0.shape[2], converted_frame_class_0.shape[3]))
    # for n in trange(0, converted_frame_class_0.shape[0]):
    #     rnd = np.random.randint(2, size=1)[0]
    #     if rnd == 0:
    #         mixed_frames_mem[n] = converted_frame_class_0
    #     elif rnd == 1:
    #         mixed_frames_mem[n] = converted_frame_class_1
    #     else:
    #         raise ValueError('Random broke')
    mixed_frames_mem[:converted_frame_class_0.shape[0]] = converted_frame_class_0
    mixed_frames_mem[converted_frame_class_0.shape[0]:] = converted_frame_class_1
    mixed_frames_mem.flush()
    return mixed_frames_mem, np.hstack((labels, labels))


def main():
    dataset = 'Dataset_equalized_Small_50'
    train_converted_frame_class_0, train_converted_frame_class_1, train_labels = predict(get_train_data(dataset))
    print("equalized_frames style0", train_converted_frame_class_0.shape, "min", train_converted_frame_class_0.min(), "max", train_converted_frame_class_0.max())
    print("equalized_frames style1", train_converted_frame_class_1.shape, "min", train_converted_frame_class_1.min(), "max", train_converted_frame_class_1.max())

    test_converted_frame_class_0, test_converted_frame_class_1, test_labels = predict(get_test_data(dataset))

    print("test equalized_frames style0", test_converted_frame_class_0.shape, "min", test_converted_frame_class_0.min(), "max", test_converted_frame_class_0.max())
    print("test equalized_frames style1", test_converted_frame_class_1.shape, "min", test_converted_frame_class_1.min(), "max", test_converted_frame_class_1.max())

    np.savez_compressed(f"runs/New_runs_with_car_corrected/Dataset_equalized_green_style", input_history_train=train_labels, train_frames=train_converted_frame_class_0, input_history_test=test_labels, test_frames=test_converted_frame_class_0)
    np.savez_compressed(f"runs/New_runs_with_car_corrected/Dataset_equalized_brown_style", input_history_train=train_labels, train_frames=train_converted_frame_class_1, input_history_test=test_labels, test_frames=test_converted_frame_class_1)

    train_mixed_frames_mem, train_mixed_labels = mix_styles(train_converted_frame_class_0, train_converted_frame_class_1, 'train', train_labels)
    test_mixed_frames_mem, test_mixed_labels = mix_styles(test_converted_frame_class_0, test_converted_frame_class_1, 'test', test_labels)
    print("mixed equalized_frames ", train_mixed_frames_mem.shape, "min", train_mixed_frames_mem.min(), "max", train_mixed_frames_mem.max())

    print("mixed test equalized_frames", test_mixed_frames_mem.shape, "min", test_mixed_frames_mem.min(), "max", test_mixed_frames_mem.max())
    np.savez_compressed(f"runs/New_runs_with_car_corrected/Dataset_equalized_mixed_style", input_history_train=train_mixed_labels, train_frames=train_mixed_frames_mem, input_history_test=test_mixed_labels, test_frames=test_mixed_frames_mem)


if __name__ == "__main__":
    try:
        main()
    except:
        if os.path.exists("classifier/out/tmp/test_data_0"):
            os.remove("classifier/out/tmp/test_data_0")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists("classifier/out/tmp/test_data_1"):
            os.remove("classifier/out/tmp/test_data_1")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists("classifier/out/tmp/test_mixed"):
            os.remove("classifier/out/tmp/test_mixed")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists("classifier/out/tmp/train_data_0"):
            os.remove("classifier/out/tmp/train_data_0")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists("classifier/out/tmp/train_data_1"):
            os.remove("classifier/out/tmp/train_data_1")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists("classifier/out/tmp/train_mixed"):
            os.remove("classifier/out/tmp/train_mixed")
            print("File deleted")
        else:
            print("The file does not exist")
        print(traceback.print_exc())
        sys.exit(0)

    if os.path.exists("classifier/out/tmp/test_data_0"):
        os.remove("classifier/out/tmp/test_data_0")
        print("File deleted")
    else:
        print("The file does not exist")

    if os.path.exists("classifier/out/tmp/test_data_1"):
        os.remove("classifier/out/tmp/test_data_1")
        print("File deleted")
    else:
        print("The file does not exist")

    if os.path.exists("classifier/out/tmp/test_mixed"):
        os.remove("classifier/out/tmp/test_mixed")
        print("File deleted")
    else:
        print("The file does not exist")

    if os.path.exists("classifier/out/tmp/train_data_0"):
        os.remove("classifier/out/tmp/train_data_0")
        print("File deleted")
    else:
        print("The file does not exist")

    if os.path.exists("classifier/out/tmp/train_data_1"):
        os.remove("classifier/out/tmp/train_data_1")
        print("File deleted")
    else:
        print("The file does not exist")

    if os.path.exists("classifier/out/tmp/train_mixed"):
        os.remove("classifier/out/tmp/train_mixed")
        print("File deleted")
    else:
        print("The file does not exist")
