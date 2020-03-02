from main import *

def infer_and_save_single_tif(img_path, model_path, output_img_path):
    parser = get_parser()
    args = parser.parse_args()

    outL = 2 * args.maskR  # Output size
    mask = generate_mask(args.inL, args.maskR)
    model = unet_model(args.inL, outL)

    model.load_weights(model_path)
    _, y_prime, _ = infer_single_img(args.inL, outL, mask, model, img_path)
    save_tif(output_img_path, y_prime)

    return

if __name__=='__main__':
    modelFile = 'models/epoch_0003.h5'
    img_path = 'DEMO_ds/A_no_atoms_cropped476px/00007.tif'
    output_img_path = 'bla.tif'

    infer_and_save_single_tif(img_path, modelFile, output_img_path)



