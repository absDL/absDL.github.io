from main import *


def infer_and_save_single_tif(img_path, model_path, output_img_path):
    """
    Makes a prediction using the current model on a single input image
    :param img_path: str, path for the input image
    :param model_path: str, path for the model weights
    :param output_img_path: str, path for the output image
    """
    
    parser = get_parser()
    args = parser.parse_args()

    outL = 2 * args.maskR  # Output size
    mask = generate_mask(args.inL, args.maskR)
    model = unet_model(args.inL, outL)

    model.load_weights(model_path)
    _, y_prime, _ = infer_single_img(args.inL, outL, mask, model, img_path)
    save_tif(output_img_path, y_prime)

    return


if __name__ == '__main__':
    modelFile = 'model example/epoch_1133.h5'
    img_path = 'miniset/A_no_atoms/000001.tif'
    output_img_path = 'miniset/A_no_atoms/000001_prediction.tif'

    infer_and_save_single_tif(img_path, modelFile, output_img_path)
