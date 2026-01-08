import argparse
import numpy as np


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='exp',
                        help="the name of the current experiment")
    parser.add_argument('--eval', action='store_true',
                        help="weather to perform inference of training")

    # data arguments
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help="dataset to use (mnist or cifar)")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--train_batch_size', type=int, default=1,
                        help="trainset batch size")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size")

    # federated arguments
    parser.add_argument('--model', type=str, default='linear',
                        choices=['cnn2', 'cnn3', 'mlp', 'linear'],
                        help="model to use (cnn, mlp)")
    parser.add_argument('--num_users', type=int, default=30, # Gaussian 30
                        help="number of users participating in the federated learning")
    parser.add_argument('--local_epochs', type=int, default=1,
                        help="number of local epochs")
    parser.add_argument('--local_iterations', type=int, default=15, # Gaussian 15
                        help="number of local iterations instead of local epoch")
    parser.add_argument('--global_epochs', type=int, default=80,
                        help="number of global epochs")

    
    
    # learning arguments
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate is 0.01 for MNIST")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="momentum")
    parser.add_argument('--lr_scheduler', action='store_false',
                        help="reduce the learning rat when val_acc has stopped improving (increasing)")
    parser.add_argument('--device', type=str, default='cuda:0',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help="device to use (gpu or cpu)")
    parser.add_argument('--seed', type=int, default=1234,
                        help="manual seed for reproducibility")

    # LRSUQ arguments
    parser.add_argument('--privacy_type', type=str, default='gaussian',
                        choices=['gaussian', 'laplace'],
                        help='privacy noise type')
    parser.add_argument('--lattice_dim', type=int, default=1,
                        choices=[1, 2, 3],
                        help="the dimension of the lattice")
    parser.add_argument('--sigma', type=float, default=0.001,
                        help='sigma parameter of gaussian distribution for LRSUQ')
    parser.add_argument('--b', type=float, default=0.01,
                        help='b parameter of laplace distribution for LRSUQ ')
    parser.add_argument('--max_iterations', type=int, default=10000,
                        help='maximum iterations for LRSUQ')
    
    parser.add_argument('--baseline', type=str, default='cepam',
                        choices=['fl', 'fl_sdq', 'fl_privacy', 'fl_privacy_sdq', 'cepam'],
                        help="baseline type to run")
    parser.add_argument('--clip_threshold', type=float, default=1.0,
                        help='clipping parameter for input')

    parser.add_argument('--lattice_scale', type=float, default=0.001, # Gaussian 0.00001, Laplace 0.00001
                        help='scaling of the integer lattice')
    args = parser.parse_args()
    return args

 