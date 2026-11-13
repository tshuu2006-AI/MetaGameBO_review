from copy import deepcopy

import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import gaussian_kde
import os

from .loss import MetaLoss

# Add KDE function for estimating VaRα
def estimate_var_alpha(losses, alpha=0.8):
    """
    Calculate Value at Risk VaRα using kernel density estimation
    
    Parameters:
        losses: task loss array
        alpha: confidence level, default 0.8 means selecting the worst 20% tasks
        
    Returns:
        VaR_alpha: α-quantile Value at Risk
    """
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy()
    
    # Estimate distribution using KDE
    kde = gaussian_kde(losses)
    
    # Calculate VaRα (α-quantile)
    x = np.linspace(np.min(losses), np.max(losses), 1000)
    cdf = np.zeros_like(x)
    for i, xi in enumerate(x):
        cdf[i] = kde.integrate_box_1d(-np.inf, xi)
    
    # Find the point closest to α
    idx = np.argmin(np.abs(cdf - alpha))
    var_alpha = x[idx]
    
    return var_alpha

# Add function to calculate CVaR
def calculate_cvar(losses, alpha=0.8):
    """
    Calculate CVaRα metric
    
    Parameters:
        losses: task loss array
        alpha: confidence level
        
    Returns:
        cvar: conditional Value at Risk
    """
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy()
    
    # Calculate VaRα
    var_alpha = estimate_var_alpha(losses, alpha)
    
    # Calculate CVaRα: average of all losses exceeding VaRα
    tail_losses = losses[losses >= var_alpha]
    
    if len(tail_losses) == 0:
        return var_alpha
    
    return np.mean(tail_losses)

class EarlyStopping:
    def __init__(self, count_down=20) -> None:
        self.reset_count_down = count_down
        self.count_down = count_down
        self.current_best = np.inf
        self.update_best = False

    def _reset(self):
        self.count_down = self.reset_count_down

    def early_stop(self, current):
        if self.count_down > 0:
            if current < self.current_best:
                self.current_best = current
                self.update_best = True
                self._reset()
            else:
                self.update_best = False
                self.count_down -= 1
            return False
        else:
            return True


class Recorder:
    def __init__(self) -> None:
        self.reset()

    @property
    def loss(self):
        return np.mean(self.loss_history)

    def update_loss(self, loss, **kwargs):
        self.loss_history.append(loss)

    def reset(self):
        self.loss_history = []


def train(
    model,
    data,
    num_epochs,
    batch_size,
    shuffle=True,
    dtype=torch.float64,
    device=torch.device("cpu"),
    train_validation_split=.8,
    use_dro=False,  # Add DRO switch
    alpha=0.8,      # Add alpha parameter, default selects worst performing 20% tasks
    seed=44,        # Add seed parameter
    use_kdpp=False, # Add k-DPP switch
    kdpp_alpha=0.01,# Add k-DPP alpha parameter
    kdpp_lr=0.1,    # Add k-DPP learning rate
    *args,
    **kwargs
):
    X, task_ids, y, w = data
    X = torch.tensor(X, dtype=dtype, device=device)
    y = torch.tensor(y, dtype=dtype, device=device)
    w = torch.tensor(w, dtype=dtype, device=device)
    task_ids = torch.tensor(task_ids, dtype=torch.long, device=device)
    tensors = [X, task_ids, y, w]
    dataset = torch.utils.data.TensorDataset(*tensors)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_validation_split, 1 - train_validation_split]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=len(validation_dataset),
    )
  
    model.evaluation_data = {"train": train_dataset, "validation": validation_dataset}
    model.initialize(task_ids.max(), device=device, dtype=dtype)
    training_loop(
        model,
        train_dataloader,
        validation_dataloader,
        num_epochs,
        use_dro=use_dro,  # Pass DRO parameter
        alpha=alpha,       # Pass alpha parameter
        seed=seed,         # Pass seed parameter
        use_kdpp=use_kdpp,     # Pass k-DPP switch
        kdpp_alpha=kdpp_alpha, # Pass k-DPP alpha
        kdpp_lr=kdpp_lr        # Pass k-DPP learning rate
    )


def test_loop(model, dataloader, loss_fn, return_task_losses=False): 
    model.eval()
    total_loss = 0.0
    task_losses = []
    
    with torch.no_grad():
        for _, batch_data in enumerate(dataloader):
            # inputs with x and task_id
            inputs = batch_data[:][:2]
            targets = batch_data[:][2]
            weights = batch_data[:][3]                

            outputs = model(*inputs)
            batch_loss = loss_fn(
                outputs, targets, weights, model.task_embedding.weight
            )
            
            # If needed, calculate loss for each task
            if return_task_losses:
                _, predictions = outputs
                task_ids = inputs[1]
                unique_task_ids = torch.unique(task_ids)
                
                for task_id in unique_task_ids:
                    mask = (task_ids == task_id)
                    task_targets = targets[mask]
                    task_preds = predictions[mask]
                    task_weights = weights[mask]
                    
                    # Calculate loss for individual task
                    task_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        task_preds, task_targets, weight=task_weights
                    )
                    task_losses.append(task_loss.item())
            
            total_loss = batch_loss.detach().cpu().numpy()

    model.train()
    if return_task_losses:
        return total_loss, np.array(task_losses)
    return total_loss


def training_loop(
    model,
    train_dataloader,
    validation_dataloader,
    num_epochs,
    use_dro=False,
    alpha=0.8,
    seed=44,  # Default value consistent with run_forrester.py
    use_kdpp=False,  # Add k-DPP switch
    kdpp_alpha=0.01,  # Add k-DPP alpha parameter
    kdpp_lr=0.1,      # Add k-DPP learning rate
):  
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.999
    )
    loss_fn = MetaLoss(
        batch_size=train_dataloader.batch_size,
        latent_dim=model.task_embedding_size
    )
    
    model.metric_loss_fn = loss_fn
    model.training_diagnostics = []

    # Get batch_size for filename
    batch_size = train_dataloader.batch_size
    
    # Create results folder
    import os
    os.makedirs('results', exist_ok=True)
    
    # Result filename - add k-DPP information
    method_parts = []
    if use_dro:
        method_parts.append(f"dro{alpha}")
    if use_kdpp:
        method_parts.append(f"kdpp{kdpp_alpha}_lr{kdpp_lr}")
    if not method_parts:
        method_parts.append("standard")
    
    method_name = "_".join(method_parts)
    result_file = f'results/{method_name}_batch{batch_size}_seed{seed}.txt'
    
    with open(result_file, 'w') as f:
        f.write(f"Training Configuration:\n")
        f.write(f"DRO: {use_dro}\n")
        f.write(f"DRO Alpha: {alpha}\n")
        f.write(f"k-DPP: {use_kdpp}\n")
        if use_kdpp:
            f.write(f"k-DPP Alpha: {kdpp_alpha}\n")
            f.write(f"k-DPP Learning Rate: {kdpp_lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Random Seed: {seed}\n\n")
        f.write("Epoch\tTrain Loss\tTrain CVaR\tVal Loss\tVal CVaR\n")
    
    train_recorder = Recorder()
    validation_recorder = Recorder()

    # DRO mode: preprocess and group data by task
    task_data_dict = {}
    if use_dro:
        print("DRO mode: extracting and grouping data by task ID...")
        
        for batch_data in train_dataloader:
            inputs = batch_data[:][:2]
            targets = batch_data[:][2]
            weights = batch_data[:][3]
            task_ids = inputs[1]
            
            # Decompose batch data into individual task data
            unique_task_ids = torch.unique(task_ids).cpu().numpy()
            for task_id in unique_task_ids:
                mask = (task_ids == task_id)
                if not mask.any():
                    continue
                    
                # Extract data for individual task
                task_inputs = [inputs[0][mask], task_ids[mask]]
                task_targets = targets[mask]
                task_weights = weights[mask]
                
                # Store task data
                if task_id not in task_data_dict:
                    task_data_dict[task_id] = []
                
                task_data_dict[task_id].append((task_inputs, task_targets, task_weights))
        
        print(f"Identified {len(task_data_dict)} different tasks")

    model.train()
    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            train_recorder.reset()
            validation_recorder.reset()
            
            #=================== DRO mode: single task training ===================
            if use_dro:
                # Stage 1: Calculate loss for each task
                task_losses = {}
                
                for task_id, task_data_list in task_data_dict.items():
                    task_loss_sum = 0
                    
                    with torch.no_grad():
                        for task_inputs, task_targets, task_weights in task_data_list:
                            outputs = model(*task_inputs)
                            _, predictions = outputs
                            
                            # Calculate task loss
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                predictions, task_targets, weight=task_weights
                            )
                            task_loss_sum += loss.item()
                    
                    # Calculate average loss for this task
                    task_losses[task_id] = task_loss_sum / len(task_data_list)
                
                # Stage 2: Sort task losses and select the highest (1-alpha) portion
                sorted_tasks = sorted(task_losses.items(), key=lambda x: x[1], reverse=True)
                num_selected = max(1, int(len(sorted_tasks) * (1 - alpha)))
                selected_tasks = [task_id for task_id, _ in sorted_tasks[:num_selected]]
                
                # Stage 3: Train on selected tasks
                train_loss_sum = 0
                train_loss_count = 0
                
                for task_id in selected_tasks:
                    for task_inputs, task_targets, task_weights in task_data_dict[task_id]:
                        optimizer.zero_grad(set_to_none=True)
                        
                        outputs = model(*task_inputs)
                        loss = loss_fn(
                            outputs, task_targets, task_weights, model.task_embedding.weight[1:]
                        )
                        
                        loss.backward()
                        optimizer.step()
                        
                        train_loss_sum += loss.item()
                        train_loss_count += 1
                        
                        train_recorder.update_loss(loss.detach().cpu().numpy())
                
                # Calculate training CVaR - using task-level losses
                train_cvar = np.mean(sorted(list(task_losses.values()), reverse=True)[:num_selected])
            
            #=================== Non-DRO mode: batch training ===================
            else:
                # Standard batch training - using original batch processing
                task_losses = []  # For calculating CVaR
                
                for batch_data in train_dataloader:
                    # Process batch data
                    inputs = batch_data[:][:2]
                    targets = batch_data[:][2]
                    weights = batch_data[:][3]
                    task_ids = inputs[1]
                    
                    # Optimizer step
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Forward propagation
                    outputs = model(*inputs)
                    batch_loss = loss_fn(
                        outputs, targets, weights, model.task_embedding.weight[1:]
                    )
                    
                    # Collect task-level losses for calculating CVaR
                    with torch.no_grad():
                        _, predictions = outputs
                        unique_task_ids = torch.unique(task_ids)
                        for task_id in unique_task_ids:
                            mask = (task_ids == task_id)
                            if not mask.any():
                                continue
                                
                            task_preds = predictions[mask]
                            task_targets = targets[mask]
                            task_weights = weights[mask]
                            
                            task_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                task_preds, task_targets, weight=task_weights
                            )
                            task_losses.append(task_loss.item())
                    
                    # Backpropagation and optimization
                    batch_loss.backward()
                    optimizer.step()
                    
                    train_recorder.update_loss(batch_loss.detach().cpu().numpy())
                
                # Calculate training CVaR - using collected task-level losses
                num_selected = max(1, int(len(task_losses) * (1 - alpha)))
                train_cvar = np.mean(sorted(task_losses, reverse=True)[:num_selected])
            
            #=================== Validation process - same for both modes ===================
            # Perform validation evaluation at task level
            validation_losses = []
            
            for batch_data in validation_dataloader:
                inputs = batch_data[:][:2]
                targets = batch_data[:][2]
                weights = batch_data[:][3]
                task_ids = inputs[1]
                
                with torch.no_grad():
                    outputs = model(*inputs)
                    batch_loss = loss_fn(
                        outputs, targets, weights, model.task_embedding.weight[1:]
                    )
                    validation_recorder.update_loss(batch_loss.detach().cpu().numpy())
                    
                    # Calculate task-level validation losses
                    _, predictions = outputs
                    unique_task_ids = torch.unique(task_ids).cpu().numpy()
                    
                    for task_id in unique_task_ids:
                        mask = (task_ids == task_id)
                        if not mask.any():
                            continue
                        
                        task_targets = targets[mask]
                        task_preds = predictions[mask]
                        task_weights = weights[mask]
                        
                        task_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            task_preds, task_targets, weight=task_weights
                        )
                        validation_losses.append(task_loss.item())
            
            # Calculate validation CVaR
            num_val_tasks = max(1, int(len(validation_losses) * (1 - alpha)))
            val_cvar = np.mean(sorted(validation_losses, reverse=True)[:num_val_tasks])
            
            # Audit complete tasks in eval mode: no embedding noise or batch fragments.
            from models.author_metrics import evaluate_classifier
            audited = {split: evaluate_classifier(model, dataset, loss_fn, alpha)
                       for split, dataset in model.evaluation_data.items()}
            train_cvar = audited['train']['cvar_bce']
            val_cvar = audited['validation']['cvar_bce']
            model.training_diagnostics.append({'epoch': epoch + 1, **audited})
            scheduler.step()

            # Record results every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                pbar.set_description(f"Epoch {epoch + 1}")
                pbar.set_postfix({
                    "train_loss": f"{train_recorder.loss:.4f}",
                    "train_cvar": f"{train_cvar:.4f}",
                    "val_loss": f"{validation_recorder.loss:.4f}",
                    "val_cvar": f"{val_cvar:.4f}"
                })
                
                with open(result_file, 'a') as f:
                    f.write(f"{epoch+1}\t{train_recorder.loss:.4f}\t{train_cvar:.4f}\t{validation_recorder.loss:.4f}\t{val_cvar:.4f}\n")

    # Record final results at end of training
    with open(result_file, 'a') as f:
        f.write(f"\nFinal Results (Completed all {num_epochs} epochs):\n")
        f.write(f"Final Train Loss: {train_recorder.loss:.4f}\n")
        f.write(f"Final Train CVaR: {train_cvar:.4f}\n")
        f.write(f"Final Val Loss: {validation_recorder.loss:.4f}\n")
        f.write(f"Final Val CVaR: {val_cvar:.4f}\n")
    
    print(f"\nTraining completed. Results saved to {result_file}")

    model.eval()
    return model