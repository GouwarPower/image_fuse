import argparse
import cv2
import numpy as np

def scale_final_image(final_image, resize_factor):
    '''
    Rescales the final image by a particular factor

    Parameters
      final_image: an HxWx3 numpy array representing the final fused image
      resize_factor: a float factor by which the final image will be scaled up
                      or down by

    Returns
      new_final_image: a (H*resize_factor) x (W * resize_factor) x 3 numpy array
                        that is a scaled up or down version of the original
                        final_image

    Preconditions:
      resize_factor is greater than 0

    Postconditions:
      If the image is being upscaled, a cubic interpolation is used. If the
       image is being downscaled an area interpolation is used.
    '''
    if resize_factor > 1:
        inter = cv2.INTER_CUBIC
    else:
        inter = cv2.INTER_AREA

    new_final_image = np.copy(final_image)
    new_final_image = cv2.resize(final_image, (0,0), new_final_image,
                         fx=resize_factor, fy=resize_factor,
                         interpolation=inter)
    return new_final_image


def make_images_same_shape(reference_img, moving_img):
    '''
    Reshapes moving_img so that it is the same shape as reference_img so that
    they can be accurately fused

    Parameters
      reference_img: the image that is serving as the sizing template
      moving_img: the image to be resized so that it is the same shape as
                   reference_img
    Returns

    Preconditions
      Both images are HxWx3 numpy arrays

    Postconditions
    '''
    # Convert Numpy HxW -> CV2 WxH
    shp = (reference_img.shape[1], reference_img.shape[0])
    new_img = np.copy(reference_img)
    new_img = cv2.resize(moving_img, shp, new_img)
    return new_img


def parse_command_line_args():
    '''
    Parses command line arguments using argparse

    Positional arguments
      img1_path: a valid path to the first image to be fused
      img2_path: a valid path to the second image to be fused
      output_name: the path/name of the final output image (extension included)

    Optional flags
      --w1: the fusion weight of the first image (float)
      --w2: the fusion weight of the second image (float)
      --g: a constant amount to be added to each pixel in the fused image (int)
      --fs: a factor by which the final image will be scaled (float)
      --s: flag indicating to show the image after fusion

    Returns
      results: a Namespace object with the values of the appropriate flags
    '''
    parser = argparse.ArgumentParser(description="Wraps OpenCV's addWeighted()")
    parser.add_argument("img1_path", action="store",
                        type=str,
                        help="Path to first image")
    parser.add_argument("img2_path", action="store",
                        type=str,
                        help="Path to second image")
    parser.add_argument("output_name", action="store",
                        type=str,
                        help="Path/name of output image")
    parser.add_argument("--w1", action="store", dest="weight1",
                        type=float, default=0.5,
                        help="Weight for first image")
    parser.add_argument("--w2", action="store", dest="weight2",
                        type=float, default=0.5,
                        help="Weight for second image")
    parser.add_argument("--g", action="store", dest="gamma",
                        type=int, default=0,
                        help="Amount added to every pixel in final image")
    parser.add_argument("--fs", action="store", dest="scale_factor",
                        type=float, default=1.0,
                        help="Factor to scale final image")
    parser.add_argument("--s", action="store_true", dest="show_img",
                        default=False,
                        help="Flag for whether or not to show the image")

    results = parser.parse_args()
    return results


def main():
    '''
    Parse the command line args, run cv2.addWeighted(), and write the output
    '''
    # Parse command line arguments and read in images
    results = parse_command_line_args()
    img1 = cv2.imread(results.img1_path)
    img2 = cv2.imread(results.img2_path)

    # Process images to make same size
    if img1.shape != img2.shape:
        img2 = make_images_same_shape(img1, img2)

    # Create the new fused image
    new_img = np.zeros(img1.shape, dtype=np.uint8)
    new_img = cv2.addWeighted(img1, results.weight1, img2, results.weight2,
                              results.gamma, new_img)

    # Scale the image if necessary
    if results.final_scale_factor != 1:
        new_img = scale_final_image(new_img, results.scale_factor)

    # Write the final image and report its shape
    cv2.imwrite(results.output_name, new_img)
    print("Final image shape: {}".format(new_img.shape))

    # Show the image if necessary
    if results.show_img:
        cv2.imshow(results.output_name, new_img)
        cv2.waitKey()

if __name__ == "__main__":
    main()
