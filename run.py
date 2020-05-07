import os
import numpy as np

import torch
from torchvision import transforms

import scipy.io
import skimage.io
import argparse
from PIL import Image
from dataset import cvt1to3channels
import time

def normalize_image(image):
    return 255*((image - np.min(image)) / (np.max(image) - np.min(image)))

def main(args):
    # Prepare and instantiate the model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32,
                           pretrained=False)
    model.load_state_dict(torch.load(args.weights))
    model.cuda()

    mat = scipy.io.loadmat(args.image)
    if 'N' in mat:
        input_mat = mat["N"]

    if args.normalize:
        input_mat = normalize_image(input_mat)
    input_mat = np.uint8(input_mat)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    for image_idx in range(input_mat.shape[2]):
        input_image_original = input_mat[:,:,image_idx]
        input_image = cvt1to3channels(input_image_original)
        input_image = Image.fromarray(np.uint8(input_image))

        trans = transforms.Compose([transforms.Resize((225,225)),transforms.CenterCrop(256), transforms.ToTensor()])
        input_image = trans(input_image).unsqueeze(0)

        with torch.no_grad():
            if torch.cuda.is_available():
                input_image = input_image.cuda()
            prediction = model(input_image)

        prediction_np = prediction.cpu().numpy()
        prediction_img = prediction_np[0,0,:,:]
        prediction_mask = (prediction_img > 0.5).astype(np.uint8)

        input_image_np = input_image.cpu().numpy()
        input_image_np = input_image_np[0,1,:,:]
        input_image_np = (255*(input_image_np / np.max(input_image_np))).astype(np.uint8)

        result_img = prediction_mask * input_image_np

        if not os.path.exists(args.result):
            os.makedirs(args.result)

        output_filename = os.path.join(args.result, "res_"+str(image_idx)+"_"+timestr+".png")
        skimage.io.imsave(output_filename + ".png", result_img)

        print("Saving channel", image_idx)

    print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )

    parser.add_argument(
        "--weights", type=str, default="./paper_weights/skull-stripper-paper.pth", help="checkpoint with weights"
    )
    parser.add_argument(
        "--image", type=str, default="./source_images/N_04_1.mat", help="image as .mat"
    )
    parser.add_argument(
        "--result", type=str, default="./run_result", help="folder for output resulting images"
    )

    parser.add_argument(
        '--normalize', dest='normalize', action='store_true', help="normalize input"
    )
    parser.set_defaults(normalize=False)
    args = parser.parse_args()
    main(args)
