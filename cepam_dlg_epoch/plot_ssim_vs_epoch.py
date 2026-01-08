import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
import argparse
from pathlib import Path

def plot_ssim_vs_epoch(result_dir=None, lattice_dim_idx=0, b_idx=0, 
                        b_lst=None, lattice_dim_lst=None):
   
    if result_dir is None:
        output_path = Path('output')
        if not output_path.exists():
            print("Error: output directory not found!")
            return
        
        run_dirs = sorted([d for d in output_path.iterdir() 
                          if d.is_dir() and d.name.startswith('image_penta_run-')],
                         key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not run_dirs:
            print("Error: No result directories found in output/")
            return
        
        result_dir = run_dirs[0]
    
    result_path = Path(result_dir)
    
    output_path = Path('output')
    
    vanilla_path = None
    cepam_path = None
    found_param_str = None
    
    for f in sorted(output_path.glob('ITER_MAT_SSIM_DLG_VANILLA_*.npy')):
        param_str = f.name.replace('ITER_MAT_SSIM_DLG_VANILLA_', '').replace('.npy', '')
        vanilla_test1 = output_path / f'ITER_MAT_SSIM_DLG_VANILLA_{param_str}.npy'
        cepam_test1 = output_path / f'ITER_MAT_SSIM_DLG_CEPAM_{param_str}.npy'
        if cepam_test1.exists():
            vanilla_path = vanilla_test1
            cepam_path = cepam_test1
            found_param_str = param_str
            break
    
    if not vanilla_path:
        if b_lst and len(b_lst) > b_idx:
            param_val = b_lst[b_idx]
            param_strs_to_try = [f'b_{param_val}', f'b_{param_val:.6f}']
        else:
            param_strs_to_try = ['b_0.005']
        
        for param_str in param_strs_to_try:
            vanilla_test1 = output_path / f'ITER_MAT_SSIM_DLG_VANILLA_{param_str}.npy'
            cepam_test1 = output_path / f'ITER_MAT_SSIM_DLG_CEPAM_{param_str}.npy'
            if vanilla_test1.exists() and cepam_test1.exists():
                vanilla_path = vanilla_test1
                cepam_path = cepam_test1
                found_param_str = param_str
                break
    
    if vanilla_path and cepam_path:
        with open(vanilla_path, 'rb') as f:
            vanilla_ssim = pickle.load(f)
        
        with open(cepam_path, 'rb') as f:
            cepam_ssim = pickle.load(f)
        
        if len(vanilla_ssim.shape) != 2 or len(cepam_ssim.shape) != 2:
            print(f"Error: Unexpected shape for Test 1 format. Expected [epochs, images]")
            return
        
        num_epochs = vanilla_ssim.shape[0]
        epochs = list(range(num_epochs))
        
    else:
        ssim_mat_path = result_path / 'SSIM_mat.npy'
        
        if not ssim_mat_path.exists():
            print(f"Error: SSIM_mat.npy not found in {result_dir}")
            print("\nAvailable Test 1 format files in output/:")
            if output_path.exists():
                test1_files = list(output_path.glob('ITER_MAT_SSIM_*.npy'))
                if test1_files:
                    for f in sorted(test1_files):
                        print(f"  - {f.name}")
                else:
                    print("  (No Test 1 format files found)")
            print("\nPlease ensure Test 1 has been run, or specify correct parameters.")
            return
        
        with open(ssim_mat_path, 'rb') as f:
            ssim_mat = pickle.load(f)
        
        num_epochs = ssim_mat.shape[1]
        epochs = list(range(num_epochs))
        
        vanilla_ssim = ssim_mat[0, :, :, lattice_dim_idx, b_idx]
        cepam_ssim = ssim_mat[3, :, :, lattice_dim_idx, b_idx]
    
    vanilla_ssim_avg = np.mean(vanilla_ssim, axis=1)
    cepam_ssim_avg = np.mean(cepam_ssim, axis=1)
    
    vanilla_ssim_std = np.std(vanilla_ssim, axis=1)
    cepam_ssim_std = np.std(cepam_ssim, axis=1)
    
    plt.figure(figsize=(10, 6))
    font = {'weight': 'bold', 'size': 14}
    plt.rc('font', **font)
    
    plt.plot(epochs, vanilla_ssim_avg, '-o', linewidth=2.5, markersize=8, 
             label='Vanilla DLG', color='red', alpha=0.8)
    plt.plot(epochs, cepam_ssim_avg, '-s', linewidth=2.5, markersize=8, 
             label='CEPAM', color='blue', alpha=0.8)
    
    lattice_label = f"n={lattice_dim_lst[lattice_dim_idx]}" if lattice_dim_lst else f"n={lattice_dim_idx+1}"
    
    def format_b_value(val):
        """Format b value to remove trailing zeros while preserving significant digits"""
        s = f"{val:.10f}".rstrip('0').rstrip('.')
        return s
    
    if found_param_str and found_param_str.startswith('b_'):
        b_val = float(found_param_str.replace('b_', ''))
        param_label = f"b={format_b_value(b_val)}"
    elif b_lst and len(b_lst) > b_idx:
        b_val = b_lst[b_idx]
        param_label = f"b={format_b_value(b_val)}"
    else:
        b_val = 0.005
        param_label = f"b={format_b_value(b_val)}"
    
    plt.xlabel('Epoch Number', fontsize=14, fontweight='bold')
    plt.ylabel('SSIM Score', fontsize=14, fontweight='bold')
    plt.title(f'SSIM Score of Vanilla DLG and CEPAM vs Epoch Number\n({lattice_label}, {param_label})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(visible=True, axis='y', alpha=0.3)
    plt.grid(visible=True, which='minor', alpha=0.15)
    plt.tight_layout()
    
    lattice_val = lattice_dim_lst[lattice_dim_idx] if lattice_dim_lst else lattice_dim_idx+1
    
    if found_param_str and found_param_str.startswith('b_'):
        param_val = float(found_param_str.replace('b_', ''))
    elif b_lst and len(b_lst) > b_idx:
        param_val = b_lst[b_idx]
    else:
        param_val = 0.005
    
    param_val_str = str(param_val).replace('.', '_')
    
    current_dir = Path.cwd()
    output_filename = current_dir / f'ssim_vs_epoch_lattice{lattice_val}_b{param_val_str}.png'
    pdf_filename = current_dir / f'ssim_vs_epoch_lattice{lattice_val}_b{param_val_str}.pdf'
    
    try:
        plt.savefig(str(output_filename), dpi=300, bbox_inches='tight')
        print(f"[OK] PNG saved: {output_filename.name}")
    except Exception as e:
        print(f"[ERROR] Failed to save PNG: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        plt.savefig(str(pdf_filename), format='pdf', bbox_inches='tight', dpi=300)
        print(f"[OK] PDF saved: {pdf_filename.name}")
    except Exception as e:
        print(f"[ERROR] Failed to save PDF: {e}")
        import traceback
        traceback.print_exc()
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot SSIM vs Epoch for CEPAM DLG experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Result directory name (e.g., "image_penta_run-202511242045"). '
                             'Can be just the folder name or full path. If not provided, uses latest.')
    parser.add_argument('--lattice_dim_idx', type=int, default=0,
                        help='Index of lattice dimension to plot (0=dim1, 1=dim2, 2=dim3)')
    parser.add_argument('--b_idx', type=int, default=0,
                        help='Index of b value to plot (0=first b, 1=second, etc.)')
    parser.add_argument('--b_values', type=str, default=None,
                        help='Comma-separated list of b values for Laplace noise (e.g., "0.001,0.005"). '
                             'If not provided, will try to infer from data.')
    parser.add_argument('--b_value', type=float, default=None,
                        help='Single b parameter value to use. If provided, overrides --b_values.')
    parser.add_argument('--lattice_dims', type=str, default="1,2",
                        help='Comma-separated list of lattice dimensions used in experiment (e.g., "1,2")')
    
    args = parser.parse_args()
    
    if args.b_value is not None:
        b_lst = [args.b_value]
    elif args.b_values:
        b_lst = [float(b.strip()) for b in args.b_values.split(',')]
    else:
        b_lst = [0.005]
    
    lattice_dim_lst = [int(d.strip()) for d in args.lattice_dims.split(',')]
    
    if args.result_dir:
        result_dir_input = args.result_dir
        
        if os.path.isabs(result_dir_input) or os.path.exists(result_dir_input):
            result_dir = result_dir_input
        elif os.path.exists(f"output/{result_dir_input}"):
            result_dir = f"output/{result_dir_input}"
        else:
            print(f"Error: Directory not found: {result_dir_input}")
            print("Available directories in output/:")
            output_path = Path('output')
            if output_path.exists():
                run_dirs = [d.name for d in output_path.iterdir() 
                           if d.is_dir() and d.name.startswith('image_penta_run-')]
                for d in sorted(run_dirs):
                    print(f"  - {d}")
            sys.exit(1)
    else:
        output_path = Path('output')
        if not output_path.exists():
            print("Error: output directory not found!")
            sys.exit(1)
        
        run_dirs = sorted([d for d in output_path.iterdir() 
                          if d.is_dir() and d.name.startswith('image_penta_run-')],
                         key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not run_dirs:
            print("Error: No result directories found in output/")
            sys.exit(1)
        
        result_dir = str(run_dirs[0])
    
    plot_ssim_vs_epoch(result_dir=result_dir,
                       lattice_dim_idx=args.lattice_dim_idx, 
                       b_idx=args.b_idx, 
                       b_lst=b_lst, 
                       lattice_dim_lst=lattice_dim_lst)

