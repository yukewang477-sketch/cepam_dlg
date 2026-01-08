import torch
import torch.optim as optim
import copy
import math
from quantization import LRSUQuantization

def federated_setup(global_model, train_data, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    indexes = torch.randperm(len(train_data))
    user_data_len = math.floor(len(train_data) / args.num_users)
    local_models = {}
    for user_idx in range(args.num_users):
        user = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        if args.lr_scheduler:
            user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1)#, verbose=True)
        local_models[user_idx] = user
    return local_models


def distribute_model(local_models, global_model):
    for user_idx in range(len(local_models)):
        local_models[user_idx]['model'].load_state_dict(copy.deepcopy(global_model.state_dict()))

# def aggregate_models(local_models, global_model, mechanism):  # FeaAvg
def aggregate_models(local_models, global_model, mechanism, args):  # FeaAvg
    mean = lambda x: sum(x) / len(x)
    state_dict = copy.deepcopy(global_model.state_dict())
    SNR_layers = []
    server_lr = getattr(args, 'server_lr', args.lr)
    for key in state_dict.keys():
        accumulator = torch.zeros_like(state_dict[key])
        SNR_users = []
        use_gradient = key in local_models[0].get('gradients', {})
        for user_idx in range(0, len(local_models)):
            if use_gradient:
                local_payload_orig = local_models[user_idx]['gradients'][key].to(state_dict[key].device).to(state_dict[key].dtype)
            else:
                local_payload_orig = local_models[user_idx]['model'].state_dict()[key] - state_dict[key]
            if mechanism is None:
                local_payload = local_payload_orig
            else:
                # set different seed for each user
                if isinstance(mechanism, LRSUQ):
                    mechanism.quantizer.update_seed(user_idx)
                local_payload = mechanism(local_payload_orig)
            if mechanism is None:
                SNR_users.append(torch.tensor(1.0, device=state_dict[key].device, dtype=state_dict[key].dtype))  # SNR is 1 for original FL
            else:
                denom = torch.var(local_payload_orig - local_payload)
                if denom.item() == 0:
                    snr_value = torch.tensor(float('inf'), device=state_dict[key].device, dtype=state_dict[key].dtype)
                else:
                    snr_value = torch.var(local_payload_orig) / denom
                SNR_users.append(snr_value)
            accumulator += local_payload
        SNR_layers.append(mean(SNR_users))
        update_value = (accumulator / len(local_models)).to(state_dict[key].dtype)
        if use_gradient:
            state_dict[key] -= server_lr * update_value
        else:
            state_dict[key] += update_value
    global_model.load_state_dict(copy.deepcopy(state_dict))
    return mean(SNR_layers)


class BaseMechanism:
    """Base mechanism class, containing common operations"""
    def __init__(self, args):
        self.device = args.device
        self.dim = args.lattice_dim
        self.C = args.clip_threshold if hasattr(args, 'clip_threshold') else 0.01
        
    def divide_into_blocks(self, input):
        """Divide input into blocks"""
        modulo = len(input) % self.dim
        if modulo:
            pad_with = self.dim - modulo
            input_vec = torch.cat((input, torch.zeros(pad_with).to(input.dtype).to(input.device)))
        else:
            pad_with = 0
            input_vec = input
        input_vec = input_vec.view(self.dim, -1)
        return input_vec, pad_with
    
    def clipping(self, input):
        """Perform clipping on the input"""
        norm = torch.norm(input, p=2)
        scaling_factor = max(1, norm / self.C)
        clipped_input = input / scaling_factor
        return clipped_input
    
    def scaled_lattice_quantize(self, input, lattice_scale):
        # Step 1: Scale the input
        scaled_input = input / lattice_scale
        # Step 2: Custom rounding (round X.5 to X+1)
        rounded_scaled_input = torch.floor(scaled_input + 0.5)
        
        # Step 3: Rescale the quantized values
        quantized_output = rounded_scaled_input * lattice_scale
        
        return quantized_output
        
    def process(self, input):
        """Core processing logic to be implemented by subclasses"""
        raise NotImplementedError
        
    def __call__(self, input):
        original_shape = input.shape
        input = input.view(-1)
        input, pad_with = self.divide_into_blocks(input)
        
        # clipping
        input = self.clipping(input)
        
        # core processing logic
        input = self.process(input)
        
        if pad_with:
            input = input.view(-1)[:-pad_with]
        return input.reshape(original_shape)

class PrivacyMechanism(BaseMechanism):
    def __init__(self, args):
        super().__init__(args)
        self.lattice_scale = args.lattice_scale
        self.privacy_type = args.privacy_type
        if self.privacy_type == 'gaussian':
            self.sigma = args.sigma
        elif self.privacy_type == 'laplace':
            self.b = args.b
        
        
        if self.privacy_type == 'gaussian':
            self.noise_dist = torch.distributions.normal.Normal(
                loc=0.0, 
                scale=self.sigma
            )
        elif self.privacy_type == 'laplace':
            self.noise_dist = torch.distributions.laplace.Laplace(
                loc=0.0, 
                scale=self.b  
            )
    
    def clipping(self, input):
        norm = torch.norm(input, p=2)
        scaling_factor = max(1, norm / self.C)
        clipped_input = input / scaling_factor
        return clipped_input
        
    def process(self, input):
        input = self.clipping(input)
        noise = self.noise_dist.sample(input.shape).to(self.device)
        input = input + noise
        input =  input = self.scaled_lattice_quantize(input, self.lattice_scale)
        return input
    
class SDQ(BaseMechanism):
    def __init__(self, args):
        super().__init__(args)
        self.lattice_scale = args.lattice_scale
        
    def clipping(self, input):
        norm = torch.norm(input, p=2)
        scaling_factor = max(1, norm / self.C)
        clipped_input = input / scaling_factor
        return clipped_input
        
    def process(self, input):
        input = self.clipping(input)        # clipping
        # SDQ quantization
        dither = torch.zeros_like(input).uniform_(-0.5, 0.5)
        input = input + dither
        input = self.scaled_lattice_quantize(input, self.lattice_scale)
        # print(f"Processed (quantized) vector:\n{input}")
        return input - dither
    """
    def process(self, input):
        dither = torch.zeros_like(input).uniform_(-0.5, 0.5)
        input = input + dither
        return torch.round(input) - dither
    """

class Privacy_SDQ(BaseMechanism):
    def __init__(self, args):
        super().__init__(args)
        self.privacy_type = args.privacy_type
        self.lattice_scale = args.lattice_scale
        if self.privacy_type == 'gaussian':
            self.sigma = args.sigma
        elif self.privacy_type == 'laplace':
            self.b = args.b
        
        if self.privacy_type == 'gaussian':
            self.noise_dist = torch.distributions.normal.Normal(
                loc=0.0, 
                scale=self.sigma
            )
        elif self.privacy_type == 'laplace':
            self.noise_dist = torch.distributions.laplace.Laplace(
                loc=0.0, 
                scale=self.b  
            )
            
    def clipping(self, input):
        norm = torch.norm(input, p=2)
        scaling_factor = max(1, norm / self.C)
        clipped_input = input / scaling_factor
        return clipped_input
        
    def process(self, input):
        input = self.clipping(input)        # clipping
        # add noise
        noise = self.noise_dist.sample(input.shape).to(self.device)
        input = input + noise
        # SDQ quantization
        dither = torch.zeros_like(input).uniform_(-0.5, 0.5)
        input = input + dither
        input = self.scaled_lattice_quantize(input, self.lattice_scale)
        # print(f"Processed (quantized) vector:\n{input}")
        return input - dither

class LRSUQ(BaseMechanism):
    def __init__(self, args):
        super().__init__(args)
        self.quantizer = LRSUQuantization(args)
        self.privacy_type = args.privacy_type
        if self.privacy_type == 'gaussian':
            self.sigma = args.sigma
        elif self.privacy_type == 'laplace':
            self.b = args.b

                
    def clipping(self, input):
        norm = torch.norm(input, p=2)
        scaling_factor = max(1, norm / self.C)
        clipped_input = input / scaling_factor
        return clipped_input
        
    def process(self, input):
        input = self.clipping(input)
        input = self.quantizer(input)
        return input

def setup_mechanism(args):
    """set up different baseline mechanisms"""
    if args.baseline == 'fl':
        return None
    elif args.baseline == 'fl_sdq':
        return SDQ(args)
    elif args.baseline == 'fl_privacy':
        return PrivacyMechanism(args)
    elif args.baseline == 'fl_privacy_sdq':
        return Privacy_SDQ(args)
    elif args.baseline == 'cepam':
        return LRSUQ(args)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")