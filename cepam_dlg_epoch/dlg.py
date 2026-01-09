import torch
from torch import optim
from torchvision import datasets, transforms
from vision import LeNet, CNN, weights_init
from PIL import Image
from utils import label_to_onehot, cross_entropy_for_onehot
import torch.nn.functional as F
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

cepam_path = os.path.join(PROJECT_ROOT, 'cepam')
if os.path.exists(cepam_path):
    sys.path.append(cepam_path)

#import cepam
try:
    from federated_utils import LRSUQ as CEPAMClass
    from quantization import LRSUQuantization
    import copy
    CEPAM_AVAILABLE = True
except ImportError:
    print("Warning: Could not import CEPAM components. Make sure cepam is in the path.")
    CEPAMClass = None
    CEPAM_AVAILABLE = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


def add_uveqFed(original_dy_dx, epsilon, bit_rate, args):
    noised_dy_dx = []
    args.epsilon = epsilon
    args.R = bit_rate
    
    if args.attack == 'cepam' or args.attack == 'CEPAM':
        # CEPAM mechanism（Gaussian and Laplace noise）
        if not CEPAM_AVAILABLE or CEPAMClass is None:
            raise ImportError("CEPAM not available. Ensure cepam/federated_utils.py is accessible.")
        
        # Set privacy_type
        if not hasattr(args, 'privacy_type') or args.privacy_type is None:
            args.privacy_type = 'laplace'
        
        # set all required parameters
        if not hasattr(args, 'device') or args.device is None:
            args.device = device
        if not hasattr(args, 'lattice_dim'):
            args.lattice_dim = 1
        if not hasattr(args, 'lattice_scale'):
            args.lattice_scale = 1e-05
        if not hasattr(args, 'clip_threshold'):
            args.clip_threshold = 1.0
        
        # Set noise parameters based on privacy_type
        if args.privacy_type == 'laplace':
            # For Laplace: use b parameter (should be set from epsilon in main.py)
            if not hasattr(args, 'b') or args.b is None:
                args.b = 0.005  # Default Laplace scale parameter
        elif args.privacy_type == 'gaussian':
            # For Gaussian: use sigma parameter (should be set from epsilon in main.py)
            if not hasattr(args, 'sigma') or args.sigma is None:
                args.sigma = 0.1  # Default Gaussian standard deviation
        else:
            raise ValueError(f"Unsupported privacy_type: {args.privacy_type}. Must be 'gaussian' or 'laplace'")
        
        if not hasattr(args, 'max_iterations'):
            args.max_iterations = 10000
        if not hasattr(args, 'seed'):
            args.seed = 923
        
        # Initialize CEPAM mechanism with selected noise type
        noiser = CEPAMClass(args)
        for g in original_dy_dx:
            output = noiser(g)
            noised_dy_dx.append(output)
    elif args.attack == "quantization":
        # quantization only
        if not CEPAM_AVAILABLE or CEPAMClass is None:
            raise ImportError("CEPAM components needed for quantization.")
        noiser = CEPAMClass(args)
        for g in original_dy_dx:
            output = noiser.apply_quantization(g) if hasattr(noiser, 'apply_quantization') else noiser(g)
            noised_dy_dx.append(output)
    else:  # privacy noise only(gaussian and laplace)
        privacy_type = getattr(args, 'privacy_type', 'laplace')
        privacy_noise = getattr(args, 'privacy_noise', privacy_type) 
        
        for g in original_dy_dx:
            if privacy_noise == 'gaussian' or privacy_type == 'gaussian':
                # Use Gaussian noise
                sigma = getattr(args, 'sigma', 0.1)
                noise = torch.normal(mean=0.0, std=sigma, size=g.shape, device=g.device)
            else:
                # Use Laplace noise with b
                b = getattr(args, 'b', 0.005) 
                noise = torch.distributions.Laplace(0, b).sample(g.shape).to(g.device)
            noised_dy_dx.append(g + noise)

    return noised_dy_dx


def mse(imageA, imageB):
    # MSE: the sum of the squared difference between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

class dlg_cls():
    def __init__(self, model=None, train_loader=None, test_loader=None, args=None, noise_func=lambda x, y, z, l: x):
        self.dst = getattr(datasets, args.dataset)("~/.torch", download=True)
        self.preprocessing = transforms.Compose([
            transforms.Resize((32, 32))
        ])
        self.tp = transforms.ToTensor()
        self.tt = transforms.ToPILImage()
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.noise_func = noise_func

    def apply_noise(self, epsilon, bit_rate, noise_func = None, args = None):
        if noise_func != None:
            self.noise_func = noise_func
        if args != None:
            self.args = args
       
        if (epsilon > 0 or self.args.attack == "quantization" or self.args.attack == "cepam" or self.args.attack == "CEPAM"):
            self.original_dy_dx = self.noise_func(list((_.detach().clone() for _ in self.dy_dx)), epsilon, bit_rate, self.args)
        else:
            # Vanilla DLG: use original gradients without any protection
            self.original_dy_dx = self.dy_dx

    #image_index:被攻击图片在dataset中的索引，learning_epoches:训练轮数，read_grads:是否读取梯度，epsilon:噪声大小，bit_rate:量化比特率，num_of_iterations:迭代次数
    def __call__(self, img_index, seed=1234,learning_epoches=0,read_grads= -1, epsilon=0, bit_rate=1,num_of_iterations=200):
        self.load_image(img_index)
        self.config_model(None,seed)
        self.train_model(learning_epoches)
        if (read_grads == -1):
            self.compute_gradients()
        else:
            self.load_model_and_gradients(read_grads)
        self.apply_noise(epsilon,bit_rate)
        return self.dlg(num_of_iterations=num_of_iterations)

    def load_image(self, img_index):
        self.img_index = img_index
        img = self.dst[img_index][0]
        img = self.preprocessing(img)
        self.gt_data = self.tp(img).to(device)
        if len(self.args.image) > 1:
            self.gt_data = Image.open(self.args.image)
            self.gt_data = self.tp(self.gt_data).to(device)
        self.gt_data = self.gt_data.view(1, *self.gt_data.size())
        # Convert grayscale (1 channel) to RGB (3 channels) for MNIST compatibility
        if self.gt_data.shape[1] == 1:
            self.gt_data = self.gt_data.repeat(1, 3, 1, 1)
        self.gt_label = torch.Tensor([self.dst[img_index][1]]).long().to(device)
        self.gt_label = self.gt_label.view(1, )
        self.gt_onehot_label = label_to_onehot(self.gt_label)
        return self.dst[self.img_index][0]

    def config_model(self,model=None,seed=1234):
        if model == None:
            self.model = LeNet().to(device)
        else:
            self.model = model
        torch.manual_seed(seed)
        self.model.apply(weights_init)
        self.model.to(device)
        self.criterion = cross_entropy_for_onehot
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self,learning_epoches=0):
        if (learning_epoches > 0):
            self.model.train_nn(
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                epoch_num=learning_epoches,
                test_loader=self.test_loader)

            return self.model.test_nn(self.test_loader, self.criterion)
    def compute_gradients(self):
        self.pred = self.model(self.gt_data)
        y = self.criterion(self.pred, self.gt_onehot_label)
        self.dy_dx = torch.autograd.grad(y, self.model.parameters())
        self.original_dy_dx = self.dy_dx
        return self.dy_dx

    def load_model_and_gradients(self,read_grads):
        grad_checkpoint_address = "./fed-ler_checkpoints/grad/checkpoint{0}_{1}.pk".format(model_number, read_grads)
        global_checkpoint_address = "./fed-ler_checkpoints/global/checkpoint{0}_{1}.pk".format(model_number, read_grads)
        fed_ler_grad_state_dict = torch.load(grad_checkpoint_address)

        global_model = torch.load(global_checkpoint_address)
        self.model = global_model
        self.dy_dx = tuple([fed_ler_grad_state_dict[key] for key in fed_ler_grad_state_dict.keys()])
        return self.dy_dx
    
    #原代码num_of_iterations = 2000
    def dlg(self,num_of_iterations = 200):
        dummy_data = torch.randn(self.gt_data.size()).to(device).requires_grad_(True)
        dummy_label = torch.randn(self.gt_onehot_label.size()).to(device).requires_grad_(True)
        # plt.figure()
        # plt.imshow(tt(dummy_data[0].cpu()))

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

        # history = []
        current_loss = torch.Tensor([1])
        iters = 0
        MSE=0
        SSIM=0
        # stop early if loss is small enough
        while (current_loss.item() > 0.00001 and iters < num_of_iterations):

            def closure():
                optimizer.zero_grad()

                dummy_pred = self.model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, self.original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()

                return grad_diff

            optimizer.step(closure)
            if iters % 10 == 0:
                current_loss = closure()
                reconstructedIm = np.asarray(self.tt(dummy_data[0].cpu()))
                RecImShape = reconstructedIm.shape
                
                # vision.py: 32x32 RGB，CIFAR：32x32x3，MNIST：28x28（resized to 32x32, converted to RGB）
                groundTruthIm = np.asarray(self.tt(self.gt_data[0].cpu()))
                MSE = mse(reconstructedIm,groundTruthIm)
                SSIM = ssim(reconstructedIm,groundTruthIm,channel_axis=2, multichannel=True)

                print(iters, "%.4f" % current_loss.item()," MSE {0:.4f}, SSIM {1:.4f}".format(MSE,SSIM))
                # history.append(self.tt(dummy_data[0].cpu()))
            iters = iters + 1

        self.final_image = self.tt(dummy_data[0].cpu())
        return current_loss.item(), MSE, SSIM


def run_dlg(img_index, model=None, train_loader=None, test_loader=None, noise_func = lambda x, y, z: x, learning_epoches = 0, epsilon=0.1, bit_rate=1,read_grads=-1,model_number=0):
    gt_data = tp(dst[img_index][0]).to(device)
    if len(args.image) > 1:
        gt_data = Image.open(args.image)
        gt_data = tp(gt_data).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)


    model = LeNet().to(device)

    torch.manual_seed(1234)
    model.apply(weights_init)
    criterion = cross_entropy_for_onehot
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if (read_grads == -1):
        #################### Train & Test ####################
        if (learning_epoches >0):
            model.train_nn(train_loader=train_loader, optimizer=optimizer, criterion=criterion,  epoch_num=learning_epoches,test_loader=test_loader)
            model.test_nn(test_loader,criterion)
        ######################################################
        # compute original gradient
        pred = model(gt_data)
        y = criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, model.parameters())
    else: 
        grad_checkpoint_address = "./fed-ler_checkpoints/grad/checkpoint{0}_{1}.pk".format(model_number,read_grads)
        global_checkpoint_address = "./fed-ler_checkpoints/global/checkpoint{0}_{1}.pk".format(model_number,read_grads)
        fed_ler_grad_state_dict = torch.load(grad_checkpoint_address)


        global_model = torch.load(global_checkpoint_address)
        model =global_model
        dy_dx = tuple([fed_ler_grad_state_dict[key] for key in fed_ler_grad_state_dict.keys()])
    #################### adding noise ####################
    if (epsilon > 0):
        original_dy_dx = noise_func(list((_.detach().clone() for _ in dy_dx)), epsilon, bit_rate)
    else:
        original_dy_dx = dy_dx

    #### adding noise!! ####
    #original_dy_dx = [w_layer + torch.normal(mean = 0, std= 0.01,size = w_layer.shape) for w_layer in original_dy_dx]
    #original_dy_dx = [w_layer+np.random.laplace(0,epsilon,w_layer.shape) for w_layer in original_dy_dx]


    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    plt.figure()
    plt.imshow(tt(dummy_data[0].cpu()))
    # plt.imshow(tt(dummy_data[0].cpu()))

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


    history = []
    current_loss = torch.Tensor([1])
    iters = 0
    #for iters in range(num_of_iterations):
    #while (iters < num_of_iterations):
    while (current_loss.item()>0.00001 and iters < num_of_iterations):

        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))
        iters = iters + 1

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(tt(dummy_data[0].cpu()))
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(dst[img_index][0])
    # plt.axis('off')

    # plt.figure(figsize=(12, 8))
    # for i in range(round(iters / 10)):
    #     plt.subplot(int(np.ceil(iters / 100)), 10, i + 1)
    #     plt.imshow(history[i])
    #     plt.title("iter=%d" % (i * 10))
    #     plt.axis('off')
    return current_loss.item()

# l = []
# for i in range(10):
#     l.append(test_image(img_index,learning_iterations=500+50*i))
# print(l)
#plt.hist([7 if (x>5) else x for x in l])
# plt.plot(l)
