import argparse
import numpy as np
import math
import sys
import os

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (CEPAM root)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add dlg directory to path (relative to project root)
dlg_path = os.path.join(PROJECT_ROOT, 'dlg')
if os.path.exists(dlg_path):
    sys.path.insert(0, dlg_path)

import iDLG
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
print(torch.__version__, torchvision.__version__)
import os
os.system('')
from utils import label_to_onehot, cross_entropy_for_onehot
import random
from torch.distributions.laplace import Laplace
from vision import LeNet, CNN, weights_init
import copy
from dlg2_0 import dlg_cls, add_uveqFed, run_dlg
import sys
from datetime import datetime
import pickle


# Note: Noise strength (b/sigma) and privacy budget (epsilon) are controlled independently
# - b/sigma: Directly controls noise strength (higher = more noise)
# - epsilon: Controls privacy protection strength (higher = weaker privacy, theoretical privacy budget)


# Custom transform to convert grayscale to RGB
class GrayscaleToRGB:
    """Convert grayscale image (1 channel) to RGB (3 channels)"""
    def __call__(self, img):
        if img.shape[0] == 1:  # if grayscale
            return img.repeat(3, 1, 1)
        return img


# Add code2 directory to path (relative to project root)
code2_path = os.path.join(PROJECT_ROOT, 'code2')
if os.path.exists(code2_path):
    sys.path.append(code2_path)

# CEPAM
try:
    from federated_utils import LRSUQ as CEPAMClass
    from quantization import LRSUQuantization
    CEPAM_AVAILABLE = True
    print("✓ CEPAM components loaded successfully")
except ImportError as e:
    print(f"✗ Warning: Could not import CEPAM components: {e}")
    print("  Make sure code is in the path.")
    CEPAMClass = None
    CEPAM_AVAILABLE = False
parser = argparse.ArgumentParser(
    description='Main 2.0 - DLG Attack on CEPAM Protection',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)


parser.add_argument('--index', type=int, default=25,
                    help='Image index for testing (CIFAR/MNIST)')
parser.add_argument('--image', type=str, default="",
                    help='Path to custom image (leave empty to use dataset)')
parser.add_argument('--dataset', type=str, default="CIFAR10",
                    choices=['CIFAR10', 'CIFAR100', 'MNIST'],
                    help='Dataset to use')
parser.add_argument('--device', type=str, default='cuda:0',
                    choices=['cuda:0', 'cuda:1', 'cpu'],
                    help='Device to use (GPU or CPU)')

parser.add_argument('--attack', type=str, default='cepam',
                    choices=['cepam', 'vanilla', 'noise_only', 'quantization'],
                    help='Attack type: cepam=CEPAM protection, vanilla=no protection')

parser.add_argument('--lattice_dim', type=int, default=1,
                    choices=[1, 2, 3],
                    help='CEPAM lattice dimension (higher = more complex)')
parser.add_argument('--privacy_type', type=str, default='laplace',
                    choices=['gaussian', 'laplace'],
                    help='CEPAM privacy noise type')
parser.add_argument('--lattice_scale', type=float, default=1e-05,
                    help='CEPAM lattice scaling factor (affects quantization)')
parser.add_argument('--clip_threshold', type=float, default=1.0,
                    help='CEPAM gradient clipping threshold')
parser.add_argument('--max_iterations', type=int, default=10000,
                    help='CEPAM max iterations for LRSUQ quantization')
parser.add_argument('--seed', type=int, default=923,
                    help='Random seed for reproducibility')

parser.add_argument('--R', type=int, default=16,
                    help='[Legacy] Compression rate (not used with CEPAM)')

# Noise strength parameters (directly control noise magnitude)
parser.add_argument('--b', type=float, default=0.005,
                    help='Laplace noise scale parameter (directly controls noise strength: higher b = more noise)')
parser.add_argument('--sigma', type=float, default=0.1,
                    help='Gaussian noise standard deviation (directly controls noise strength: higher sigma = more noise)')
parser.add_argument('--b_values', type=str, default=None,
                    help='Comma-separated list of b values for Laplace noise (e.g., "0.001,0.005,0.01"). Higher b = more noise. If not provided, uses --b value.')
parser.add_argument('--sigmas', type=str, default=None,
                    help='Comma-separated list of sigma values for Gaussian noise (e.g., "0.01,0.1,1.0"). Higher sigma = more noise. If not provided, uses --sigma value.')

# Privacy budget parameter (theoretical privacy protection strength, independent from noise)
parser.add_argument('--epsilon', type=float, default=500,
                    help='Privacy budget epsilon (theoretical privacy protection strength: higher epsilon = weaker privacy protection)')
parser.add_argument('--epsilons', type=str, default=None,
                    help='Comma-separated list of epsilon values (e.g., "10,100,1000"). Higher epsilon = weaker privacy protection. Note: epsilon and noise strength (b/sigma) are controlled independently.')

parser.add_argument('--privacy_noise', type=str, default='laplace',
                    help='[Legacy] Privacy noise type (CEPAM uses --privacy_type instead)')
parser.add_argument('--epochs', type=int, default=None,
                    help='Number of training epochs to test (0 to N-1, inclusive). Default: 31 (epochs 0-30)')
parser.add_argument('--num_images', type=int, default=3,
                    help='Number of images to test (default: 3)')

args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and args.device.startswith('cuda'):
    print(f"Warning: CUDA not available, falling back to CPU")
    device = "cpu"
    args.device = "cpu"

print("\n" + "=" * 70)
print("Main 2.0 - DLG Attack on CEPAM Protection - Configuration")
print("=" * 70)
print(f"Device: {device}")
print(f"Dataset: {args.dataset}")
print(f"Attack type: {args.attack}")
if args.attack == 'cepam':
    print(f"\nCEPAM Configuration:")
    print(f"  - Lattice dimension: {args.lattice_dim}")
    print(f"  - Privacy type: {args.privacy_type}")
    if args.privacy_type == 'gaussian':
        print(f"  - Sigma (Gaussian noise std, controls noise strength): {args.sigma}")
    elif args.privacy_type == 'laplace':
        print(f"  - b (Laplace noise scale, controls noise strength): {args.b}")
    print(f"  - Epsilon (privacy budget, theoretical privacy protection): {args.epsilon}")
    print(f"  - Lattice scale: {args.lattice_scale}")
    print(f"  - Clip threshold: {args.clip_threshold}")
print("=" * 70 + "\n")

img_index = args.index




import iDLG

def produce_image_pentas(image_number_list, iteration_list, noise_param_list, lattice_dim_list, privacy_type='laplace'):
    """
    Comprehensive CEPAM testing: Generate images comparing different protection levels
    
    This function tests DLG attack effectiveness against CEPAM with various configurations:
    1. Vanilla DLG (no protection) - baseline
    2. Quantization only - simple protection  
    3. Noise only - privacy noise without quantization
    4. CEPAM - full protection (quantization + privacy noise)
    
    Args:
        image_number_list: List of image indices to test
        iteration_list: List of training iterations (epochs)
        noise_param_list: List of noise parameter values
            - For Laplace: list of b values (e.g., [0.001, 0.005, 0.01])
            - For Gaussian: list of sigma values (e.g., [0.1, 0.5, 1.0])
            - Higher value = more noise = stronger protection against DLG attack
        lattice_dim_list: List of lattice dimensions (CEPAM complexity: 1, 2, or 3)
        privacy_type: 'laplace' or 'gaussian' - determines which noise mechanism to use
        
    Outputs:
        - Saves reconstructed images for visual comparison
        - Saves metrics matrices (Loss, MSE, SSIM) as .npy files
        - Creates organized output directory structure
    """
    plt.xscale("log")
    loss_matrix = np.zeros([4, len(iteration_list), len(image_number_list), len(lattice_dim_list), len(noise_param_list)])
    MSE_matrix = np.zeros([4, len(iteration_list), len(image_number_list), len(lattice_dim_list), len(noise_param_list)])
    SSIM_matrix = np.zeros([4, len(iteration_list), len(image_number_list), len(lattice_dim_list), len(noise_param_list)])
    # grads_norm_mat = np.zeros([len(iteration_list), len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)


    now = datetime.now()


    dt_string = now.strftime("%Y%m%d%H%M")
    parent_path = "output/image_penta_run-"+str(dt_string)

    os.mkdir(parent_path)
    dlg = dlg_cls(
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        noise_func=add_uveqFed)
    dlg.config_model()
    for i, iter in enumerate(iteration_list):
        print("iteration number {0}".format(i))
        if i > 0:
            dlg.train_model(1)
        for j, n in enumerate(image_number_list):
            step_size = 1/len(iteration_list)
            print("testing image number{0} finished {1}%".format(j,round(100*(i*step_size+step_size*n/len(image_number_list)))))
            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            orig_img = dlg.load_image(n)
            dlg.compute_gradients()

            ## Vanilla DLG (no protection) ##
            args.attack = 'vanilla'
            dlg.apply_noise(0, 1, args=args)
            loss_matrix[0, i, j, 0, 0], MSE_matrix[0, i, j, 0, 0], SSIM_matrix[0, i, j, 0, 0] = dlg.dlg()
            dlg_only_img = dlg.final_image

            for k, lattice_dim in enumerate(lattice_dim_list):
                args.lattice_dim = lattice_dim

                ## QUANTIZATION ONLY ##
                args.attack = 'quantization'
                dlg.apply_noise(0, 1, args=args)
                loss_matrix[1, i, j, k, 0], MSE_matrix[1, i, j, k, 0], SSIM_matrix[1, i, j, k, 0] = dlg.dlg()
                quant_only_img = dlg.final_image

                for l, noise_param in enumerate(noise_param_list):
                    # Set noise parameter directly (b for Laplace, sigma for Gaussian)
                    if privacy_type == 'laplace':
                        args.b = noise_param
                        param_str = f"b_{noise_param}"
                    else:  # gaussian
                        args.sigma = noise_param
                        param_str = f"sigma_{noise_param}"
                    
                    ## NOISE ONLY ##
                    args.attack = 'noise_only'
                    args.privacy_type = privacy_type
                    args.privacy_noise = privacy_type
                    dlg.apply_noise(0, 1, args=args)
                    loss_matrix[2, i, j, k, l], MSE_matrix[2, i, j, k, l], SSIM_matrix[2, i, j, k, l] = dlg.dlg()
                    noise_only_img = dlg.final_image

                    ## CEPAM ##
                    args.attack = 'cepam'
                    args.privacy_type = privacy_type
                    # Noise parameters are set from epsilon above
                    dlg.apply_noise(0, 1, args=args)
                    loss_matrix[3, i, j, k, l], MSE_matrix[3, i, j, k, l], SSIM_matrix[3, i, j, k, l] = dlg.dlg()
                    cepam_only_img = dlg.final_image
                    dir_path = parent_path+"/{0}_{1}_{2}_{3}".format(iter, lattice_dim, param_str, n)

                    # save the images
                    os.mkdir(dir_path)
                    orig_img.save(dir_path+"/orig-SSIM(1)-MSE(0).png")
                    dlg_only_img.save(dir_path + "/vanilla_dlg-SSIM("+str(SSIM_matrix[0, i, j, 0, 0])+")-MSE("+str(MSE_matrix[0, i, j, 0, 0])+").png")
                    quant_only_img.save(dir_path + "/quantization_only-SSIM("+str(SSIM_matrix[1, i, j, k, 0])+")-MSE("+str(MSE_matrix[1, i, j, k, 0])+").png")
                    cepam_only_img.save(dir_path + "/CEPAM-SSIM("+str(SSIM_matrix[3, i, j, k, l])+")-MSE("+str(MSE_matrix[3, i, j, k, l])+").png")
                    noise_only_img.save(dir_path + "/noise_only-SSIM("+str(SSIM_matrix[2, i, j, k, l])+")-MSE("+str(MSE_matrix[2, i, j, k, l])+").png")
        
        # Save results after each iteration
        print("\n" + "=" * 70)
        print(f"Iteration {i} completed! Saving results...")
        print("=" * 70)
        with open(parent_path+'/loss_mat.npy', 'wb') as f:
            pickle.dump(loss_matrix, f)
        with open(parent_path + '/MSE_mat.npy', 'wb') as f:
            pickle.dump(MSE_matrix, f)
        with open(parent_path + '/SSIM_mat.npy', 'wb') as f:
            pickle.dump(SSIM_matrix, f)
        print(f"✓ Results saved to: {parent_path}")
        print(f"✓ Completed iterations: {i+1}/{len(iteration_list)}")
        print("=" * 70 + "\n")
    
    # Final save (redundant but ensures everything is saved)
    print("\n" + "=" * 70)
    print("All iterations completed! Final save...")
    print("=" * 70)
    with open(parent_path+'/loss_mat.npy', 'wb') as f:
        pickle.dump(loss_matrix, f)
    with open(parent_path + '/MSE_mat.npy', 'wb') as f:
        pickle.dump(MSE_matrix, f)
    with open(parent_path + '/SSIM_mat.npy', 'wb') as f:
        pickle.dump(SSIM_matrix, f)
    print(f"✓ Final results saved to: {parent_path}")
    print("=" * 70 + "\n")


def produce_image_pentas_idlg(image_number_list, iteration_list, noise_param_list, lattice_dim_list, privacy_type='laplace'):
    """
    Comprehensive CEPAM testing with iDLG attack: Generate images comparing different protection levels
    
    This function tests iDLG attack effectiveness against CEPAM with various configurations:
    1. Vanilla iDLG (no protection) - baseline
    2. Quantization only - simple protection  
    3. Noise only - privacy noise without quantization
    4. CEPAM - full protection (quantization + privacy noise)
    
    Args:
        image_number_list: List of image indices to test
        iteration_list: List of training iterations (epochs)
        noise_param_list: List of noise parameter values
            - For Laplace: list of b values (e.g., [0.001, 0.005, 0.01])
            - For Gaussian: list of sigma values (e.g., [0.1, 0.5, 1.0])
            - Higher value = more noise = stronger protection against DLG attack
        lattice_dim_list: List of lattice dimensions (CEPAM complexity: 1, 2, or 3)
        privacy_type: 'laplace' or 'gaussian' - determines which noise mechanism to use
        
    Outputs:
        - Saves reconstructed images for visual comparison
        - Saves metrics matrices (Loss, MSE, SSIM) as .npy files
        - Creates organized output directory structure
    """
    plt.xscale("log")
    loss_matrix = np.zeros([4, len(iteration_list), len(image_number_list), len(lattice_dim_list), len(noise_param_list)])
    MSE_matrix = np.zeros([4, len(iteration_list), len(image_number_list), len(lattice_dim_list), len(noise_param_list)])
    SSIM_matrix = np.zeros([4, len(iteration_list), len(image_number_list), len(lattice_dim_list), len(noise_param_list)])
    
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M")
    parent_path = "output/image_penta_idlg_run-"+str(dt_string)
    os.mkdir(parent_path)
    
    # Initialize model for training
    dlg = dlg_cls(
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        noise_func=add_uveqFed)
    dlg.config_model()
    
    for i, iter in enumerate(iteration_list):
        print("iteration number {0}".format(i))
        if i > 0:
            dlg.train_model(1)
        for j, n in enumerate(image_number_list):
            step_size = 1/len(iteration_list)
            print("testing image number{0} finished {1}%".format(j,round(100*(i*step_size+step_size*n/len(image_number_list)))))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            
            # Run iDLG attack for different protection methods
            # Note: iDLG.run_idlg will compute gradients internally and apply protection via noise_func
            
            ## Vanilla iDLG (no protection) ##
            args.attack = 'vanilla'
            loss_vanilla = iDLG.run_idlg(
                idx=n,
                train_loader=train_loader,
                test_loader=test_loader,
                noise_func=add_uveqFed,
                epsilon=0,
                bit_rate=1,
                args=args
            )
            loss_matrix[0, i, j, 0, 0] = loss_vanilla
            
            for k, lattice_dim in enumerate(lattice_dim_list):
                args.lattice_dim = lattice_dim

                ## QUANTIZATION ONLY ##
                args.attack = 'quantization'
                loss_quant = iDLG.run_idlg(
                    idx=n,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    noise_func=add_uveqFed,
                    epsilon=0,
                    bit_rate=1,
                    args=args
                )
                loss_matrix[1, i, j, k, 0] = loss_quant

                for l, noise_param in enumerate(noise_param_list):
                    # Set noise parameter directly (b for Laplace, sigma for Gaussian)
                    if privacy_type == 'laplace':
                        args.b = noise_param
                    else:  # gaussian
                        args.sigma = noise_param
                    
                    ## NOISE ONLY ##
                    args.attack = 'noise_only'
                    args.privacy_type = privacy_type
                    args.privacy_noise = privacy_type
                    loss_noise = iDLG.run_idlg(
                        idx=n,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        noise_func=add_uveqFed,
                        epsilon=0,
                        bit_rate=1,
                        args=args
                    )
                    loss_matrix[2, i, j, k, l] = loss_noise

                    ## CEPAM ##
                    args.attack = 'cepam'
                    args.privacy_type = privacy_type
                    loss_cepam = iDLG.run_idlg(
                        idx=n,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        noise_func=add_uveqFed,
                        epsilon=0,
                        bit_rate=1,
                        args=args
                    )
                    loss_matrix[3, i, j, k, l] = loss_cepam
                    
                    # Note: iDLG.run_idlg only returns loss
                    # MSE and SSIM would need to be computed from saved images in iDLG results directory
                    # For now, we store loss values only
        
        # Save results after each iteration
        print("\n" + "=" * 70)
        print(f"Iteration {i} completed! Saving results...")
        print("=" * 70)
        with open(parent_path+'/loss_mat.npy', 'wb') as f:
            pickle.dump(loss_matrix, f)
        with open(parent_path + '/MSE_mat.npy', 'wb') as f:
            pickle.dump(MSE_matrix, f)
        with open(parent_path + '/SSIM_mat.npy', 'wb') as f:
            pickle.dump(SSIM_matrix, f)
        print(f"✓ Results saved to: {parent_path}")
        print(f"✓ Completed iterations: {i+1}/{len(iteration_list)}")
        print("=" * 70 + "\n")
    
    # Final save
    print("\n" + "=" * 70)
    print("All iterations completed! Final save...")
    print("=" * 70)
    with open(parent_path+'/loss_mat.npy', 'wb') as f:
        pickle.dump(loss_matrix, f)
    with open(parent_path + '/MSE_mat.npy', 'wb') as f:
        pickle.dump(MSE_matrix, f)
    with open(parent_path + '/SSIM_mat.npy', 'wb') as f:
        pickle.dump(SSIM_matrix, f)
    print(f"✓ Final results saved to: {parent_path}")
    print("=" * 70 + "\n")


def train_model(image_number_list,iteration_list, algo='DLG'):
    plt.xscale("log")
    acc_vec = np.zeros([len(iteration_list)])
    grads_norm_mat = np.zeros([len(iteration_list), len(image_number_list)])

    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    dlg = dlg_cls(
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        noise_func=add_uveqFed)
    dlg.config_model()
    for i, iter in enumerate(iteration_list):
        print("iteration number {0}".format(i))
        if i > 0:
            acc_vec[i] = dlg.train_model(1)
        for j, n in enumerate(image_number_list):
            dlg.load_image(n)
            gradients = dlg.compute_gradients()
            grads_norm_mat[i, j] = sum([x.norm(p=2) ** 2 for x in gradients]) ** (0.5)
    with open('output/TRAINING_TEST_GRADS'+algo+'.npy', 'wb') as f:
        pickle.dump(grads_norm_mat, f)
    with open('output/TRAINING_TEST_ACC' + algo + '.npy', 'wb') as f:
        pickle.dump(acc_vec, f)

    plt.figure()
    font = {
        'weight': 'bold',
        'size': 16}
    plt.rc('font', **font)
    plt.plot(iteration_list, acc_vec, linewidth=3)
    plt.title("Lenet acc after training the model")
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("accuracy[%]")
    plt.figure()
    plt.plot(iteration_list, np.mean(grads_norm_mat, axis=1), linewidth=3)
    plt.title("gradients L2 Norm after training the model")
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("mean L2 Norm")
    plt.show()



def run_iteration_dlg_idlg_tests(image_number_list, iteration_list, algo='DLG', noise_param_list=None, bit_rate=8):
    """
    Run DLG attack tests across different training iterations
    
    Compares vanilla DLG (no protection) vs CEPAM-protected gradients across
    multiple training epochs to see how model training affects attack success.
    
    Args:
        image_number_list: List of image indices to test
        iteration_list: List of training iterations/epochs
        algo: Algorithm type ('DLG' or 'iDLG')
        noise_param_list: List of noise parameter values
            - For Laplace: list of b values (e.g., [0.001, 0.005, 0.01])
            - For Gaussian: list of sigma values (e.g., [0.1, 0.5, 1.0])
        bit_rate: [Legacy] Bit rate (not used with CEPAM)
        
    Returns:
        Saves results to 'output/' directory with metrics for both vanilla and CEPAM
        Saves reconstructed images to 'output/image_penta_run-*/' directory
    """
    
    plt.xscale("log")
    
    # Initialize result matrices for Vanilla DLG (baseline) - Only SSIM
    dlg_ssim_per_iter_matrix = np.zeros([len(iteration_list), len(image_number_list)])

    # Initialize result matrices for CEPAM protection - Only SSIM
    cepam_ssim_per_iter_matrix = np.zeros([len(iteration_list), len(image_number_list)])

    # Track gradient norms
    grads_norm_mat = np.zeros([len(iteration_list), len(image_number_list)])
    
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True,transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # Create output directory for images
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M")
    parent_path = "output/image_penta_run-"+str(dt_string)
    os.makedirs(parent_path, exist_ok=True)

    # run all the tests:
    dlg = dlg_cls(
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        noise_func=add_uveqFed)
    dlg.config_model()

    # Use noise parameter list (b for Laplace, sigma for Gaussian)
    if noise_param_list is None:
        # Use default noise parameter based on privacy_type
        if args.privacy_type == 'laplace':
            noise_param_list = [args.b]
            param_name = 'b'
        else:  # gaussian
            noise_param_list = [args.sigma]
            param_name = 'sigma'
        print(f"Warning: noise_param_list not provided. Using default {param_name}={noise_param_list[0]}")
    
    # Determine parameter name based on privacy_type
    param_name = 'b' if args.privacy_type == 'laplace' else 'sigma'
    
    for param_value in noise_param_list:
        print("\n" + "=" * 70)
        print(f"Testing with {param_name} = {param_value}")
        print("=" * 70)
        
        # Set noise parameter directly
        if args.privacy_type == 'laplace':
            args.b = param_value
            print(f"CEPAM Configuration: b={args.b:.6f}, lattice_dim={args.lattice_dim}, epsilon={args.epsilon}")
        else:  # gaussian
            args.sigma = param_value
            print(f"CEPAM Configuration: sigma={args.sigma:.6f}, lattice_dim={args.lattice_dim}, epsilon={args.epsilon}")
        
        for i, iter in enumerate(iteration_list):
            print("iteration number {0}".format(i))

            if i > 0:
                dlg.train_model(1)

            for j, n in enumerate(image_number_list):
                orig_img = dlg.load_image(n)
                gradients = dlg.compute_gradients()
                grads_norm_mat[i,j] = sum([x.norm(p=2) ** 2 for x in gradients]) ** (0.5)

                # Run Vanilla DLG (no protection)
                args.attack = 'vanilla'
                dlg.apply_noise(0, bit_rate, args=args)
                _, _, ssim = dlg.dlg()  # Only use SSIM, ignore loss and mse

                dlg_ssim_per_iter_matrix[i, j] = ssim
                dlg_only_img = dlg.final_image

                # Run DLG with CEPAM protected gradients
                args.attack = 'cepam'
                dlg.apply_noise(0, bit_rate, args=args)

                _, _, ssim = dlg.dlg()  # Only use SSIM, ignore loss and mse
                cepam_ssim_per_iter_matrix[i, j] = ssim
                cepam_only_img = dlg.final_image
                
                # Save images
                dir_path = parent_path+"/{0}_{1}_{2}_{3}".format(iter, args.lattice_dim, param_value, n)
                os.makedirs(dir_path, exist_ok=True)
                orig_img.save(dir_path+"/orig-SSIM(1).png")
                dlg_only_img.save(dir_path + "/vanilla_dlg-SSIM("+str(dlg_ssim_per_iter_matrix[i, j])+").png")
                cepam_only_img.save(dir_path + "/CEPAM-SSIM("+str(cepam_ssim_per_iter_matrix[i, j])+").png")

        # save the SSIM matrix only
        param_str = '{0}_{1}'.format(param_name, param_value)

        with open('output/ITER_MAT_SSIM_'+algo+'_VANILLA_'+param_str+'.npy', 'wb') as f:
            pickle.dump(dlg_ssim_per_iter_matrix, f)

        with open('output/ITER_MAT_SSIM_'+algo+'_CEPAM_'+param_str+'.npy', 'wb') as f:
            pickle.dump(cepam_ssim_per_iter_matrix, f)

        with open('output/ITER_GRAD_MAT_NORM_'+algo+'_new_'+param_str+'.npy', 'wb') as f:
            pickle.dump(grads_norm_mat, f)

        # plot SSIM comparison only
        font = {'weight': 'bold', 'size': 16}
        
        # Plot SSIM comparison
        plt.figure(figsize=(10, 6))
        plt.rc('font', **font)
        plt.plot(iteration_list, np.mean(dlg_ssim_per_iter_matrix, axis=1), 
                linewidth=3, label='Vanilla DLG', marker='o')
        plt.plot(iteration_list, np.mean(cepam_ssim_per_iter_matrix, axis=1), 
                linewidth=3, label='CEPAM', marker='s')
        plt.title(f"Structural Similarity (SSIM): Vanilla vs CEPAM ({param_name}={param_value})")
        plt.legend()
        plt.grid(visible=True, axis="y")
        plt.grid(visible=True, which='minor')
        plt.xlabel("Training Epochs")
        plt.ylabel("SSIM - Lower is Better Defense")

        # Plot gradient norms
        plt.figure(figsize=(10, 6))
        plt.rc('font', **font)
        plt.plot(iteration_list, np.mean(grads_norm_mat, axis=1), linewidth=3, marker='o')
        plt.title("Gradient L2 Norm vs Training Epochs")
        plt.grid(visible=True, axis="y")
        plt.grid(visible=True, which='minor')
        plt.xlabel("Training Epochs")
        plt.ylabel("Mean L2 Norm")

        plt.show()
        
        print(f"\n✓ Results saved to output/ directory with prefix: {param_name}_{param_value}")
        print(f"✓ Images saved to: {parent_path}")


def run_epsilon_dlg_idlg_tests(image_number_list,epsilon_list,bit_rate_lst, algo='DLG'):
    """

    Args:
        image_number_list:
        epsilon_list:
        algo:

    Returns:

    """
    plt.xscale("log")
    loss_per_epsilon_matrix = np.zeros([len(bit_rate_lst), len(epsilon_list),len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True,transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # run all the tests:
    for k, bit_rate in enumerate(bit_rate_lst):
        for i, epsilon in enumerate(epsilon_list):
            for j,n in enumerate(image_number_list):

                # extract_img = run_dlg if algo == 'DLG' else iDLG.run_idlg
                dlg = dlg_cls(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    args=args,
                    noise_func=add_uveqFed)
                loss_per_epsilon_matrix[k, i, j] = dlg(
                    img_index=n,
                    learning_epoches=0,
                    read_grads=-1,
                    epsilon=epsilon,
                    bit_rate=bit_rate)
                # loss_per_epsilon_matrix[k,i, j] = k+i+j
                print("#### image {0} epsilon {1} bitRate {2} loss {3}####".format(j, epsilon, bit_rate,loss_per_epsilon_matrix[k,i,j]))
            print("bit_rate: {0} epsilon:{1} average loss: {2} loss values:{3}".format(bit_rate, epsilon,np.mean(loss_per_epsilon_matrix[k][i]),loss_per_epsilon_matrix[k][i]))

    # # save the loss into a matrix

    #     np.save(f, loss_per_epsilon_matrix[0,:,:])
    # np.savetxt('output/epsilon_mat'+algo+'.txt', loss_per_epsilon_matrix[0,:,:], fmt='%1.4e')

    with open('output/TOTAL_MAT'+algo+'.npy', 'wb') as f:
        pickle.dump(loss_per_epsilon_matrix, f)

    # # plot the accuracy
    # plt.figure()
    # font = {'weight': 'bold','size': 16}
    #
    # plt.rc('font', **font)
    # plt.plot(epsilon_list,np.mean(loss_per_epsilon_matrix,axis=1),linewidth=3)
    # plt.title("{0} loss attack type {1} for various levels of noise levels".format(algo, args.attack))
    # plt.grid(visible=True,axis="y")
    # plt.grid(visible=True,which='minor')
    # plt.xlabel("2/epsilon")
    # plt.ylabel("loss")
import pickle


def run_dlg_idlg_tests(image_number_list,check_point_list,model_number, algo='DLG'):
    plt.xscale("log")
    loss_per_iter_matrix = np.zeros([len(check_point_list),len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True,transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), GrayscaleToRGB()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # run all the tests:
    for i,iter in enumerate(check_point_list):
        for j,n in enumerate(image_number_list):
            extract_img = run_dlg if algo == 'DLG' else iDLG.run_idlg
            loss_per_iter_matrix[i, j] = extract_img(n,
                                                        train_loader=train_loader,
                                                        test_loader=test_loader,
                                                        learning_epoches=0,
                                                        epsilon=0,
                                                        noise_func=add_uveqFed,
                                                        read_grads=iter,
                                                        model_number=model_number)
        #loss_per_epsilon_matrix[i, j] = i+j
        print("iter:{0} average loss: {1} loss values:{2}".format(iter,np.mean(loss_per_iter_matrix[i]),loss_per_iter_matrix[i]))

    # # save the loss into a matrix
    # with open('../output/loss_mat'+algo+'.npy', 'wb') as f:
    #     np.save(f, loss_per_iter_matrix)
    # np.savetxt('../output/loss_mat'+algo+'.txt', loss_per_iter_matrix, fmt='%1.4e')

    # plot the accuracy
    plt.figure()
    font = {
        'weight': 'bold',
        'size': 16}

    plt.rc('font', **font)
    plt.plot(check_point_list,np.mean(loss_per_iter_matrix,axis=1),linewidth=3)
    plt.title("{0} loss attack type {1}".format(algo, args.attack))
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("iter")
    plt.ylabel("loss")

import cProfile,pstats



def plot_graphs(algo, iteration_list, param_str=''):
    """
    Load and plot saved results comparing Vanilla DLG vs CEPAM
    
    This function loads previously saved test results and generates
    comparison plots showing how CEPAM protects against DLG attacks.
    
    Args:
        algo: Algorithm name ('DLG' or 'iDLG')
        iteration_list: List of training iterations used in tests
        param_str: Parameter string for file naming (e.g., 'sigma_0.001')
        
    Expected file format in output/ directory:
        - ITER_MAT_LOSS_{algo}_VANILLA_{param_str}.npy
        - ITER_MAT_LOSS_{algo}_CEPAM_{param_str}.npy
        - etc.
    """
    with open('output/ITER_MAT_LOSS_' + algo + '_VANILLA_' + param_str + '.npy', 'rb') as f:
        dlg_loss_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_MSE_' + algo + '_VANILLA_' + param_str + '.npy', 'rb') as f:
        dlg_mse_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_SSIM_' + algo + '_VANILLA_' + param_str + '.npy', 'rb') as f:
        dlg_ssim_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_LOSS_' + algo + '_CEPAM_' + param_str + '.npy', 'rb') as f:
        cepam_loss_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_MSE_' + algo + '_CEPAM_' + param_str + '.npy', 'rb') as f:
        cepam_mse_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_SSIM_' + algo + '_CEPAM_' + param_str + '.npy', 'rb') as f:
        cepam_ssim_per_iter_matrix = pickle.load(f)
    with open('output/ITER_GRAD_MAT_NORM_' + algo + '_new_' + param_str + '.npy', 'rb') as f:
        grads_norm_mat = pickle.load(f)

    font = {
        'weight': 'bold',
        'size': 16}
    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list, np.mean(np.log(dlg_loss_per_iter_matrix), axis=1),
             linewidth=3, label='Vanilla DLG')
    plt.plot(iteration_list,
             np.mean(np.log(cepam_loss_per_iter_matrix), axis=1), linewidth=3, label='CEPAM')
    plt.title("DLG Vanilla vs CEPAM - Gradient Loss Metric")
    plt.legend()
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epochs")
    plt.ylabel("log(loss)")

    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list, np.mean(np.log(dlg_mse_per_iter_matrix), axis=1),
             linewidth=3, label='Vanilla DLG')
    plt.plot(iteration_list, np.mean(np.log(cepam_mse_per_iter_matrix), axis=1),
             linewidth=3, label='CEPAM')
    plt.title("DLG Vanilla MSE vs CEPAM MSE")
    plt.legend()
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epochs")
    plt.ylabel("log(MSE)")

    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list, np.mean(dlg_ssim_per_iter_matrix, axis=1),
             linewidth=3, label='Vanilla DLG')
    plt.plot(iteration_list,
             np.mean(cepam_ssim_per_iter_matrix, axis=1), linewidth=3, label='CEPAM')
    plt.title("DLG Vanilla vs CEPAM SSIM")
    plt.legend()
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epochs")
    plt.ylabel("SSIM")

    plt.figure()
    plt.plot(iteration_list, np.mean(grads_norm_mat, axis=1), linewidth=3)

    plt.show()


# import cProfile,pstats
def main():
    """
    Main 2.0 function for comprehensive DLG attack testing against CEPAM
    
    This is an alternative implementation to main.py with similar functionality
    but focuses on image reconstruction tests. It provides:
    - Visual comparison of different protection levels
    - Comprehensive metrics collection
    
    Test Pipeline:
    1. Comprehensive image tests - Visual comparison of protection levels
    2. Optional: Iteration tests (uncomment to enable)
    
    CEPAM Parameters Tested:
    - Sigma values: Controls noise strength (privacy level)
    - Lattice dimensions: Controls quantization complexity (1, 2, 3)
    - Training iterations: How model training affects attack
    
    Output:
    - Numerical metrics (Loss, MSE, SSIM) saved as .npy files
    - Reconstructed images for visual comparison
    - Comparison plots
    """
    
    # Check if CEPAM is available
    if not CEPAM_AVAILABLE:
        print("\n" + "!" * 70)
        print("ERROR: CEPAM components not available!")
        print("Please ensure code2 folder is in the path.")
        print("!" * 70)
        return
    
    # ============================================================
    # Test Configuration
    # ============================================================
    
    # Parse noise parameter values from command line (b for Laplace, sigma for Gaussian)
    if args.privacy_type == 'laplace':
        if args.b_values:
            # Parse comma-separated b values
            noise_param_lst = [float(b.strip()) for b in args.b_values.split(',')]
            print(f"Using Laplace b values from command line: {noise_param_lst}")
            print(f"  Note: Higher b = more noise = stronger protection")
        else:
            # Use single b value from --b argument
            noise_param_lst = [args.b]
            print(f"Using single Laplace b value from --b: {noise_param_lst}")
            print(f"  Note: Higher b = more noise = stronger protection")
    else:  # gaussian
        if args.sigmas:
            # Parse comma-separated sigma values
            noise_param_lst = [float(s.strip()) for s in args.sigmas.split(',')]
            print(f"Using Gaussian sigma values from command line: {noise_param_lst}")
            print(f"  Note: Higher sigma = more noise = stronger protection")
        else:
            # Use single sigma value from --sigma argument
            noise_param_lst = [args.sigma]
            print(f"Using single Gaussian sigma value from --sigma: {noise_param_lst}")
            print(f"  Note: Higher sigma = more noise = stronger protection")
    
    # Parse epsilon values (privacy budget, independent from noise strength)
    if args.epsilons:
        # Parse comma-separated epsilon values
        epsilon_lst = [float(e.strip()) for e in args.epsilons.split(',')]
        print(f"Using epsilon values (privacy budget) from command line: {epsilon_lst}")
        print(f"  Note: Higher epsilon = weaker privacy protection (theoretical)")
    else:
        # Use single epsilon value from --epsilon argument
        epsilon_lst = [args.epsilon]
        print(f"Using single epsilon value (privacy budget) from --epsilon: {epsilon_lst}")
        print(f"  Note: Higher epsilon = weaker privacy protection (theoretical)")
    
    # Parse number of epochs from command line or use default
    if args.epochs is not None:
        num_epochs = args.epochs
        iteration_lst = list(range(num_epochs))  # 0, 1, 2, ..., N-1
        print(f"Using {num_epochs} epochs (0 to {num_epochs-1}) from command line")
    else:
        num_epochs = 31
        iteration_lst = list(range(31))  # 0, 1, 2, ..., 30 (default)
        print(f"Using default {num_epochs} epochs (0 to {num_epochs-1})")

    # Test parameters
    lattice_dim_lst = [1, 2]
    img_lst = list(range(0, args.num_images))
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    print("\n" + "=" * 70)
    print("Main 2.0 - DLG Attack on CEPAM Protection Mechanism")
    print("=" * 70)
    print(f"Test Configuration:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Images to test: {len(img_lst)} images")
    print(f"  - Training iterations: {len(iteration_lst)} epochs")
    if args.privacy_type == 'laplace':
        print(f"  - Laplace b values (noise strength): {noise_param_lst}")
        print(f"    (Higher b = more noise = stronger protection)")
    else:
        print(f"  - Gaussian sigma values (noise strength): {noise_param_lst}")
        print(f"    (Higher sigma = more noise = stronger protection)")
    print(f"  - Epsilon values (privacy budget, theoretical): {epsilon_lst}")
    print(f"    (Higher epsilon = weaker privacy protection, independent from noise strength)")
    print(f"  - Lattice dimensions: {lattice_dim_lst}")
    print(f"  - Privacy type: {args.privacy_type}")
    print(f"  - Lattice scale: {args.lattice_scale}")
    print(f"  - Clip threshold: {args.clip_threshold}")
    print("=" * 70)
    print(f"\nNote: Using {'QUICK TEST' if len(img_lst) <= 5 else 'FULL TEST'} configuration")
    print("=" * 70)

    # ============================================================
    # Test 1: Iteration Tests (SSIM vs Epoch)
    # ============================================================
    print("\n" + ">" * 70)
    print("[Test 1] Running Iteration Tests (SSIM vs Epoch)")
    print("  Purpose: Evaluate how model training affects DLG attack success")
    print("  Comparing: Vanilla DLG vs CEPAM protection")
    print("  Output: SSIM score vs epoch number plots")
    print(">" * 70)
    
    run_iteration_dlg_idlg_tests(
        img_lst, 
        iteration_lst, 
        algo='DLG', 
        noise_param_list=noise_param_lst
    )

    # ============================================================
    # Test 2: Comprehensive Image Tests (Visual Comparison)
    # ============================================================
    # print("\n" + ">" * 70)
    # print("[Test 2] Running Comprehensive Image Reconstruction Tests")
    # print("  Purpose: Generate visual comparisons of attack effectiveness")
    # print("  Testing: Vanilla, Quantization, Noise, and CEPAM")
    # print(">" * 70)
    # 
    # produce_image_pentas(
    #     img_lst, 
    #     iteration_lst, 
    #     noise_param_lst, 
    #     lattice_dim_lst,
    #     privacy_type=args.privacy_type
    # )
    

    print("\n" + "=" * 70)
    print("✓ All Tests Completed Successfully!")
    print("=" * 70)
    print(f"Results Location: ./output/")
    print(f"\nGenerated Files:")
    print(f"  - Metrics: loss_mat.npy, MSE_mat.npy, SSIM_mat.npy")
    print(f"  - Images: image_penta_run-*/")
    print(f"\nNext Steps:")
    print(f"  1. Check ./output/ directory for results")
    print(f"  2. View reconstructed images for visual comparison")
    print(f"  3. Analyze metrics to quantify CEPAM effectiveness")
    print(f"\nTo enable image reconstruction test:")
    print(f"  - Uncomment the 'Test 2' section in main() function")
    print("=" * 70)
    
    # plt.show()  # Commented out to avoid blocking terminal
    # Uncomment above line if you want to see plots interactively


if __name__ == "__main__":
    main()
