import torch
import numpy as np


class LRSUQuantization:
    def __init__(self, args):
        self.device = args.device
        self.d = args.lattice_dim  # dimension
        self.sigma = args.sigma if hasattr(args, 'sigma') else 0.1  # default sigma=0.1
        self.b = args.b if hasattr(args, 'b') else 0.1  # default b=0.1
        self.max_iterations = args.max_iterations if hasattr(args, 'max_iterations') else 10000  # default max_iterations=10000
        self.privacy_type = args.privacy_type
        self.lattice_scale = args.lattice_scale
        self.C = args.clip_threshold

        # set random seed
        self.base_seed = args.seed if hasattr(args, 'seed') else 42
        self.current_seed = self.base_seed
        # save initial random state
        self.initial_rng_state = torch.get_rng_state()
        self.update_generator()
    
    def update_seed(self, user_idx):
        """set different seed for each user"""
        self.current_seed = self.base_seed + user_idx
        self.update_generator()
        
    def update_generator(self):
        """update random number generator"""
        self.generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.manual_seed(self.current_seed)
        # save current random state
        current_rng_state = torch.get_rng_state()
        # temporarily set new random seed
        torch.manual_seed(self.current_seed)
        if self.privacy_type == 'gaussian':
            # create random number generator for chi-square distribution
            self.chi2_dist = torch.distributions.chi2.Chi2(
                df=torch.tensor(self.d+2.0).to(self.device)
            )   
        elif self.privacy_type == 'laplace':
            # create random number generator for Gamma(2,1) distribution
            self.gamma_dist = torch.distributions.gamma.Gamma( 
                concentration=torch.tensor(2.0).to(self.device),  
                rate=torch.tensor(1.0).to(self.device)           
            )
        # restore previous random state
        torch.set_rng_state(current_rng_state)

    def scaled_lattice_quantize(self, input, lattice_scale):
        # Step 1: Scale the input
        scaled_input = input / lattice_scale
        # Step 2: Custom rounding (round X.5 to X+1)
        rounded_scaled_input = torch.floor(scaled_input + 0.5)
        
        # Step 3: Rescale the quantized values
        quantized_output = rounded_scaled_input * lattice_scale
        
        return quantized_output
    
    def quantize_vector(self, x):
        """quantize a single vector using LRSUQ"""
        #print(f"Input vector L2 norm = {torch.norm(x, p=2):.6f}")
        assert len(x) == self.d, f"Input vector dimension must be {self.d}"
        if self.privacy_type == 'gaussian': 
            # generate chi-square random number
            radius = (self.chi2_dist.sample().sqrt()) * self.sigma
        elif self.privacy_type == 'laplace':
            radius = self.gamma_dist.sample() * self.b
        lamda = torch.zeros(self.d, device=x.device)

        # for d=1 case, return result directly
        if self.d == 1:
            v = (torch.rand(self.d, device=x.device, generator=self.generator) - 0.5)
            # lamda = torch.round(x/(radius*self.lattice_scale) - v)
            lamda = torch.floor((((x/radius - v)) / self.lattice_scale) + 0.5)
            # lamda = torch.floor((x/(radius*self.lattice_scale) - v) + 0.5)
            return lamda, 1

        # for d>1 case, check condition
        for i in range(self.max_iterations):
            # generate uniform random vector
            v = (torch.rand(self.d, device=x.device, generator=self.generator) - 0.5)
            
            # quantization
            # lamda = torch.round(x/(radius*self.lattice_scale) - v)
            lamda = torch.floor((x/(radius*self.lattice_scale) - v) + 0.5)
            result = (radius*self.lattice_scale) * lamda
            
            # check condition
            error = result - x
            if torch.norm(error, p=2) <= radius:
                return lamda, i
    
        # if not found suitable quantization value after max iterations, return original value
        return lamda, self.max_iterations
    
    def vector_decoder(self, x, iteration):
        """
        decode quantized vector
        """
        assert len(x) == self.d, f"Input vector dimension must be {self.d}"
        if self.privacy_type == 'gaussian': 
            # generate chi-square random number
            radius = (self.chi2_dist.sample().sqrt()) * self.sigma
        elif self.privacy_type == 'laplace':
            radius = self.gamma_dist.sample() * self.b
        
        # for d=1 case, generate one random vector
        
        if self.d == 1:
            v = (torch.rand(self.d, device=x.device, generator=self.generator) - 0.5)
            return (radius*self.lattice_scale) * (x + v) # beta(U)(M+V)

        # for d>1 case, generate random vector specified times
        v = torch.zeros(self.d, device=x.device)
        # generate random vector specified times
        for i in range(iteration):
            # generate uniform random vector
            v = (torch.rand(self.d, device=x.device, generator=self.generator) - 0.5)
            
        result = (radius*self.lattice_scale) * (x + v)    
        return result
    
    def __call__(self, input_tensor):
        """quantize input tensor
        Args:
            input_tensor: shape is [d, -1], each column is a d-dimensional vector
        Returns:
            output_tensor: quantized and decoded tensor, shape is the same as input_tensor
        """
        assert input_tensor.shape[0] == self.d, f"Input tensor first dimension must be {self.d}"
        
        # initialize output
        output_tensor = torch.zeros_like(input_tensor, dtype=input_tensor.dtype)
        
        # quantize each column
        for i in range(input_tensor.shape[1]):
            # server
            quantized_vector, iterations = self.quantize_vector(input_tensor[:, i])
            # center
            decoded_vector = self.vector_decoder(quantized_vector, iterations)
            output_tensor[:, i] = decoded_vector
        
        return output_tensor
    
