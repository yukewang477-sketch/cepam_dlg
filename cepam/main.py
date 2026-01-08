import gc
import sys
from statistics import mean
import time
import torch
from configurations import args_parser
from tqdm import tqdm
import utils
import models
import federated_utils
from torchinfo import summary
import numpy as np
import os


if __name__ == '__main__':
    torch.cuda.empty_cache()
    start_time = time.time()
    args = args_parser()
    boardio, textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))

    # Create results directory if it doesn't exist
    os.makedirs('results_testing_gaussian', exist_ok=True)
    os.makedirs('results_testing_laplace', exist_ok=True)
    os.makedirs('results_testing_laplace_0.01', exist_ok=True)
    # Define results file name based on model, baseline, seed (and dimension if cepam is used)
    if args.baseline == 'cepam':
        if args.privacy_type == 'gaussian':
            results_file = f'results_testing_gaussian/{args.model}_{args.baseline}_{args.privacy_type}_{args.seed}_{args.lattice_dim}_{args.sigma}.txt'
        else:
            results_file = f'results_testing_laplace_0.01/{args.model}_{args.baseline}_{args.privacy_type}_{args.seed}_{args.lattice_dim}_{args.b}.txt'
    else:
        if args.privacy_type == 'gaussian':
            results_file = f'results_testing_gaussian/{args.model}_{args.baseline}_{args.privacy_type}_{args.seed}_{args.sigma}.txt'
        else:
            results_file = f'results_testing_laplace_0.01/{args.model}_{args.baseline}_{args.privacy_type}_{args.seed}_{args.b}.txt'

    # data
    train_data, test_loader = utils.data(args)
    input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

    # model
    if args.model == 'mlp':
        global_model = models.FC2Layer(input, output)
    elif args.model == 'cnn2':
        global_model = models.CNN2Layer(input, output, args.data)
    elif args.model == 'cnn3':
        global_model = models.CNN3Layer()
    else:
        global_model = models.Linear(input, output)
    textio.cprint(str(summary(global_model)))
    global_model.to(args.device)

    # Objective function
    train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

    # learning curve
    train_loss_list = []
    val_acc_list = []

    #  inference
    if args.eval:
        global_model.load_state_dict(torch.load(path_best_model))
        test_acc = utils.test(test_loader, global_model,test_creterion, args.device)
        textio.cprint(f'eval test_acc: {test_acc:.2f}%')
        gc.collect()
        sys.exit()

    # training loops
    local_models = federated_utils.federated_setup(global_model, train_data, args)
    # mechanism = federated_utils.JoPEQ(args)
    mechanism = federated_utils.setup_mechanism(args)
    SNR_list = []

    for global_epoch in tqdm(range(0, args.global_epochs)):
        federated_utils.distribute_model(local_models, global_model)
        users_loss = []

        for user_idx in range(args.num_users):
            user = local_models[user_idx]
            grad_accumulator = None
            user_loss = []

            for local_epoch in range(0, args.local_epochs):
                mean_loss, gradients = utils.compute_gradients(
                    user['data'], user['model'], train_creterion,
                    args.device, args.local_iterations
                )
                user_loss.append(mean_loss)

                if grad_accumulator is None:
                    grad_accumulator = {name: torch.zeros_like(param) for name, param in gradients.items()}

                for name in grad_accumulator:
                    grad_accumulator[name] += gradients[name]

            averaged_gradients = {name: grad_accumulator[name] / args.local_epochs for name in grad_accumulator}
            grad_norm = torch.sqrt(sum((g.flatten() ** 2).sum() for g in averaged_gradients.values())).item()
            # textio.cprint(f'user {user_idx} | global_epoch {global_epoch} | grad_norm {grad_norm:.6f}')
            user['gradients'] = averaged_gradients
            users_loss.append(mean(user_loss))

        train_loss = mean(users_loss)
        # SNR = federated_utils.aggregate_models(local_models, global_model, mechanism) 
        SNR = federated_utils.aggregate_models(local_models, global_model, mechanism, args)  # FeaAvg
        SNR_list.append(SNR)

        val_acc = utils.test(val_loader, global_model, test_creterion, args.device)

        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)


        with open(results_file, 'a') as f:
            f.write(f'epoch {global_epoch + 1} accuracy: {val_acc:.2f}%\n')

        boardio.add_scalar('train', train_loss, global_epoch)
        boardio.add_scalar('validation', val_acc, global_epoch)
        gc.collect()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model.state_dict(), path_best_model)

        textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | '
                      f'val_acc: {val_acc:.0f}% | '
                      f'SNR: {20 * torch.log10(SNR):.3f} | '
                      f'avg SNR: {20 * torch.log10(sum(SNR_list) / len(SNR_list)):.3f}')
        
        # write the average SNR at the end into the file
        if global_epoch == args.global_epochs - 1:  # Last epoch
            avg_snr = 20 * torch.log10(sum(SNR_list) / len(SNR_list))
            with open(results_file, 'a') as f:
                f.write(f'Avg. SNR: {avg_snr:.3f}\n')

    np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', val_acc_list)
    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')

    # write the accuracy thing into the file
    # Load best model and test
    global_model.load_state_dict(torch.load(path_best_model))
    test_acc = utils.test(test_loader, global_model, test_creterion, args.device)
    textio.cprint(f'test_acc: {test_acc:.2f}%')
    # Append test accuracy to results file
    with open(results_file, 'a') as f:
        f.write(f'Accuracy: {test_acc:.2f}\n')