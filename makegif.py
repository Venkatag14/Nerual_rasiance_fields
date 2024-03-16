import cv2
from pathlib import Path
import imageio as iio


def make_gif_from_dir(src_dir, dst_path, fps=30.):
    """
    Saves a file in dst_path when given a directory
    with images (*.png) at a given frames per sec

    Args:
        src_dir: directory containing .png images with end of filename 
                    indicating order (image_1.png, image_2.png, ...)
        dst_path: filepath to save GIF (e.g., ./NeRF.gif)
        fps: frames per second

    Return:
        None
    """
    # Load images from dir
    files = Path(src_dir).glob('*.png')
    images = []
    for file in sorted(files, key=lambda x: int(x.stem.rsplit("_")[-1])):
        img = cv2.imread(str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        
    # Save GIF at the specified path
    iio.mimsave(dst_path, images, "GIF", loop=0, duration=1000.*1/fps)
    print(f"Saved GIF at {dst_path}")

if __name__ == "__main__":
    images_dir = Path.cwd().joinpath("output_imgs_lego_withoutpos")
    savegifs_dir = Path.cwd().joinpath("Data", "lego", "NeRF_withoutpos.gif")
    make_gif_from_dir(images_dir, savegifs_dir, 30)