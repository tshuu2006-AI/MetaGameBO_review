import argparse
import torch
from benchmarks.forrester import Forrester
from MetaGameBO import MetaGameBO
from plotting import plot_features, plot_update

def parse_arguments():
    parser = argparse.ArgumentParser(description='MALIBO with DRO and k-DPP for Forrester benchmark')
    
    # DRO related parameters
    parser.add_argument('--use_dro', action='store_true', help='Use distributionally robust optimization for training')
    parser.add_argument('--alpha', type=float, default=0.8, help='Confidence level for DRO (0.8 means training on the worst performing 20% tasks)')

    # k-DPP related parameters
    parser.add_argument('--use_kdpp', action='store_true', help='Use k-DPP adaptive sampling')
    parser.add_argument('--kdpp_alpha', type=float, default=0.4, 
                        help='Alpha parameter for k-DPP, controls subset size k=alpha*N, default is 0.4')
    parser.add_argument('--kdpp_lr', type=float, default=0.1, 
                        help='Learning rate for k-DPP, controls weight update speed')
    
    # Original parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--seed', type=int, default=44, help='Random seed')
    parser.add_argument('--num_tasks', type=int, default=200, help='Number of tasks')
    parser.add_argument('--samples_per_task', type=int, default=32, help='Number of samples per task')
    
    # GPU related parameters
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    
    return parser.parse_args()

def get_device(args):
    """Determine the device to use"""
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}' if args.gpu_id < torch.cuda.device_count() else 'cuda:0'
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return device

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set device
    device = get_device(args)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)
    
    classifier_config = {
        "num_layers": 5,
        "num_features": 50,
        "num_hidden_units": 64,
        "device": device,  # Update to selected device
        "seed": args.seed
    }
    
    # Remove device parameter to avoid duplicate passing
    train_config = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "use_dro": args.use_dro,
        "alpha": args.alpha,
        "seed": args.seed
    }
    
    # Add k-DPP related parameters
    if args.use_kdpp:
        classifier_config["use_kdpp"] = True
        classifier_config["alpha"] = args.kdpp_alpha
        classifier_config["kdpp_lr"] = args.kdpp_lr
        
        train_config["use_kdpp"] = True
        train_config["kdpp_alpha"] = args.kdpp_alpha
        train_config["kdpp_lr"] = args.kdpp_lr

    benchmark = Forrester(n_data_per_task=[args.samples_per_task] * args.num_tasks, seed=args.seed)
    meta_data = benchmark.get_meta_data()

    optimizer = MALIBO(benchmark.search_space, **classifier_config)
    optimizer.meta_fit(meta_data, **train_config)
    plot_features(benchmark, optimizer)

    # Add result output
    methods = []
    if args.use_dro:
        methods.append("DRO")
    if args.use_kdpp:
        methods.append("kDPP")
    if not methods:
        methods.append("Standard")
    
    run_type = "+".join(methods)
    device_type = "GPU" if device.startswith("cuda") else "CPU"
    
    print(f"\n{run_type} training completed on {device_type}")
    if args.use_dro:
        print(f"DRO alpha={args.alpha}")
    if args.use_kdpp:
        print(f"k-DPP alpha={args.kdpp_alpha}, learning rate={args.kdpp_lr}")
    
    # Result filename
    result_filename = f"results/{run_type.lower()}"
    if args.use_dro:
        result_filename += f"_dro{args.alpha}"
    if args.use_kdpp:
        result_filename += f"_kdpp{args.kdpp_alpha}_lr{args.kdpp_lr}"
    result_filename += f"_{device_type.lower()}.txt"
    
    print(f"Results saved to {result_filename}")

    num_steps = 16
    for i in range(num_steps):
        eval_spec = optimizer.generate_evaluation_specification()
        evaluation = benchmark(eval_spec)
        optimizer.report(evaluation)
        plot_update(
            benchmark,
            optimizer,
            optimizer.seed,
            title=f"Iteration {i+1}"
        )