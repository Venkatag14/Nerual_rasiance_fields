import torch
import matplotlib.pyplot as plt
# from nerf_wrapper import DataLoader, NeRF, PositionalEncoder, get_rays, raw2outputs, sample_stratified, prepare_chunks, prepare_viewdirs_chunks
from PositionalEncoding import PositionalEncoder
from network import NeRF
from rendering import get_rays, raw2outputs, sample_stratified
from pipeline import prepare_chunks, prepare_viewdirs_chunks, nerf_forward
from dataloader import DataLoader
import argparse
    
def test(d_input, n_freqs, log_space, n_freqs_views, d_filter, n_layers, skip, n_samples_hierarchical, chunksize, kwargs_sample_stratified, kwargs_sample_hierarchical, modelpath):
    
    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,
                                        log_space=log_space)
    encode = lambda x: encoder(x)
    encode_viewdirs = lambda x: encoder_viewdirs(x)
    d_viewdirs = encoder_viewdirs.d_output
    
    model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
              d_viewdirs=d_viewdirs)
    model.to(device)
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint)
    model.eval()
    
    output_images = []
    
    with torch.no_grad():
        for target_img_idx in range(poses.shape[0]):
            target_pose = poses[target_img_idx].to(device)
            target_img = images[target_img_idx].to(device)
            height, width = target_img.shape[:2]
            rays_o, rays_d = get_rays(height, width, focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            
            outputs = nerf_forward(rays_o, rays_d, t_near, t_far, encode, model, kwargs_sample_stratified, n_samples_hierarchical, kwargs_sample_hierarchical, model, encode_viewdirs, chunksize=chunksize)
            
            rgb_predicted = outputs['rgb_map'].reshape([height, width, 3]).detach().cpu().numpy()
            # print(rgb_predicted.shape)
            output_images.append(rgb_predicted)
        
        

    return output_images

if __name__ == "__main__":
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--model', default='checkpoints/nerf7000.pt', help='select coarse model')
    Args = Parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    
    file = "lego"

    data = DataLoader(file)
    print("Data loaded")

    images = torch.from_numpy(data.test.images).to(device)
    poses = torch.from_numpy(data.test.transforms).to(device)
    focal = torch.as_tensor(138.8889).to(device)
    print("Images and poses loaded")
    print(poses)

    N_stratified_samples = 8
    t_near, t_far = 2., 6.
    h, w = data.test.images.shape[1:3]

    # Encoders
    d_input = 3           # Number of input dimensions
    n_freqs = 10          # Number of encoding functions for samples
    log_space = True      # If set, frequencies scale in log space
    use_viewdirs = True   # If set, use view direction as input
    n_freqs_views = 4     # Number of encoding functions for views

    # Stratified sampling
    n_samples = 64         # Number of spatial samples per ray
    perturb = True         # If set, applies noise to sample positions
    inverse_depth = False  # If set, samples points linearly in inverse depth

    # Model
    d_filter = 128          # Dimensions of linear layer filters
    n_layers = 2            # Number of layers in network bottleneck
    skip = []               # Layers at which to apply input residual
    use_fine_model = True   # If set, creates a fine model
    d_filter_fine = 128     # Dimensions of linear layer filters of fine network
    n_layers_fine = 6       # Number of layers in fine network bottleneck

    # Hierarchical sampling
    n_samples_hierarchical = 64   # Number of samples per ray
    perturb_hierarchical = False  # If set, applies noise to sample positions

    # Optimizer
    lr = 5e-4  # Learning rate

    # Training
    n_iters = 10000
    batch_size = 2**14          # Number of rays per gradient step (power of 2)
    one_image_per_step = True   # One image per gradient step (disables batching)
    chunksize = 2**14           # Modify as needed to fit in GPU memory
    center_crop = True          # Crop the center of image (one_image_per_)
    center_crop_iters = 50      # Stop cropping center after this many epochs
    display_rate = 25          # Display test output every X epochs

    # Early Stopping
    warmup_iters = 100          # Number of iterations during warmup phase
    warmup_min_fitness = 10.0   # Min val PSNR to continue training at warmup_iters
    n_restarts = 10             # Number of times to restart if training stalls

    # We bundle the kwargs for various functions to pass all at once.
    kwargs_sample_stratified = {
        'n_samples': n_samples,
        'perturb': perturb,
        'inverse_depth': inverse_depth
    }
    kwargs_sample_hierarchical = {
        'perturb': perturb
    }
    print("Testing...")
    output_images = test(d_input, n_freqs, log_space, n_freqs_views, d_filter, n_layers, skip, n_samples_hierarchical, chunksize, kwargs_sample_stratified, kwargs_sample_hierarchical, Args.model)
    for i in range(len(output_images)):
        plt.imsave(f"nerftrain2/output_imgs{file}/{i}.png", output_images[i])
        plt.clf()







