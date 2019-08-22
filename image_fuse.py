import argparse
import cv2
import numpy as np

def scale_final_image(final_image, resize_factor):
    '''
    TODO: Document
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


def make_images_same_shape(img1, img2):
    '''
    TODO: Document
    '''
    # Convert Numpy HxW -> CV2 WxH
    shp = (img1.shape[1], img1.shape[0])
    new_img2 = np.copy(img1)
    new_img2 = cv2.resize(img2, shp, new_img2)
    return new_img2


def parse_command_line_args():
    '''
    TODO: Document
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
    parser.add_argument("--fs", action="store", dest="final_scale_factor",
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
    print(results)
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
        new_img = scale_final_image(new_img, results.final_scale_factor)

    # Write the final image and report its shape
    cv2.imwrite(results.output_name, new_img)
    print("Final image shape: {}".format(new_img.shape))

    # Show the image if necessary
    if results.show_img:
        cv2.imshow(results.output_name, new_img)
        cv2.waitKey()

if __name__ == "__main__":
    main()
