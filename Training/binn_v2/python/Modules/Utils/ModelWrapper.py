"""Training wrapper for BINN models.

This module provides a lightweight orchestration layer around a PyTorch model:
epoch loops, validation bookkeeping, checkpoint save/load helpers, and
device-selection utilities used by BINN training scripts.
"""

import time, sys, random, os, copy
import numpy as np
import torch
from datetime import timedelta
import subprocess

# ------------------------------ GPU utilities ------------------------------

def get_nvidia_smi_output():
    nvidia_smi = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
    nvidia_smi_output = nvidia_smi.communicate()[0].decode('utf8')
    return nvidia_smi_output.split('\n')

def parse_gpu_usages(nvidia_smi_lines):
    usages = []
    for line in nvidia_smi_lines:
        str_idx = line.find('MiB / ')
        if str_idx != -1:
            usages.append(int(line[str_idx-7:str_idx]))
    return usages

def pick_lowest_usage_gpu(usages, pick_from):
    gpus_sorted = np.argsort(usages)
    for idx in gpus_sorted:
        if idx in pick_from:
            return 'cuda:' + str(idx)
    return 'cpu'

def GetLowestGPU(pick_from=[0, 1, 2, 3], verbose=True, return_usages=False, mps=False, cpu=False):
    """Pick a compute device, preferring the least-used CUDA GPU when available."""

    if cpu:
        if verbose:
            print('Device set to cpu')
        return 'cpu'
    if not torch.cuda.is_available() or not pick_from:
        if mps:
            print('Device set to mps')
            return 'mps'
        if verbose:
            print('Device set to cpu')
        return 'cpu'
    nvidia_smi_lines = get_nvidia_smi_output()
    usages = parse_gpu_usages(nvidia_smi_lines)
    device = pick_lowest_usage_gpu(usages, pick_from)
    if verbose:
        print(" ======================= GPU USAGES ================")
        print('Device set to ' + device)
        print("=====================================================")
    if return_usages:
        return device, usages
    else:
        return device

def synchronize_if_needed(x):
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    elif x.device.type == "mps":
        torch.mps.synchronize()
    # no sync needed for CPU

# ------------------------------ Time helper --------------------------------

def TimeRemaining(current_iter,
                  total_iter,
                  start_time,
                  previous_time=None,
                  ops_per_iter=1.0):
    """Estimate elapsed and remaining wall-clock time for iterative training."""
    current_time = time.time()
    elapsed = current_time - start_time
    remaining = total_iter * elapsed / current_iter - elapsed
    ms_per_op = None
    if previous_time is not None:
        ms_per_op = (current_time - previous_time) / ops_per_iter
    elapsed = str(timedelta(seconds=int(elapsed)))
    remaining = str(timedelta(seconds=int(remaining)))
    return elapsed, remaining, ms_per_op

# ------------------------------ Tensor move helpers ------------------------

def _to_device_obj(obj, device):
    """Return obj moved to device if it's a Tensor/collection containing Tensors."""
    try:
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        elif isinstance(obj, (list, tuple)):
            seq_type = type(obj)
            return seq_type(_to_device_obj(x, device) for x in obj)
        elif isinstance(obj, dict):
            return {k: _to_device_obj(v, device) for k, v in obj.items()}
    except Exception:
        pass
    return obj

def _move_unregistered_tensors_in_module(module, device, _visited=None):
    """Move any Tensor attributes not registered as params/buffers to device."""
    if _visited is None:
        _visited = set()
    if id(module) in _visited:
        return
    _visited.add(id(module))

    registered = set(name for name, _ in module.named_parameters(recurse=False))
    registered.update(name for name, _ in module.named_buffers(recurse=False))

    for name, val in list(vars(module).items()):
        if name in registered:
            continue
        if isinstance(val, torch.nn.Module):
            _move_unregistered_tensors_in_module(val, device, _visited)
            continue
        new_val = _to_device_obj(val, device)
        if new_val is not val:
            try:
                setattr(module, name, new_val)
            except Exception:
                pass

# ------------------------------ Model Wrapper -------------------------------

class ModelWrapper:
    """Utility wrapper for BINN optimization, logging, and checkpointing."""

    # ------------------------------------------------------------------
    # CONSTRUCTOR -------------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 regularizer=None,
                 #augmentation=None,
                 #scheduler=None,
                 save_name=None,
                 save_best_train=False,
                 save_best_val=True,
                 save_opt=False,
                 save_reg=False,
                 seed=0):

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.regularizer = regularizer
        #self.augmentation = augmentation
        #self.scheduler = scheduler
        self.save_name = save_name
        self.save_best_train = bool(save_name and save_best_train)
        self.save_best_val = bool(save_name and save_best_val)
        self.save_opt = bool(save_name and save_opt)
        self.save_reg = bool(save_name and save_reg)
        self.seed = seed

        # Logs ---------------------------------------------------------
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_pde_loss_list = []
        self.val_pde_loss_list = []
        self.train_data_loss_list = []
        self.val_data_loss_list = []
        self.epoch_times = []

        self.diffusion_errors, self.growth_errors = [], []
        self.diffusion_preds, self.growth_preds = [], []

        # Optional spatial snapshots ----------------------------------
        self.train_pde_losses_list_spatial = []
        self.train_data_losses_list_spatial = []
        self.x_train_list = []
        self.loss_count_list = []
        self.save_index = []

        self.train = False
        self.val = False

        if self.seed is not None:
            self.set_seed(self.seed)

    # ------------------------------------------------------------------
    # TRAINING LOOP -----------------------------------------------------
    # ------------------------------------------------------------------

    def fit(self,
            x_tr_input,
            y_tr_input,
            *,
            batch_size=None,
            epochs=1,
            verbose=1,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            early_stopping=None,
            include_val_aug=False,
            include_val_reg=False,
            lr_dec_epoch=None,
            lr_dec_prop=1.0,
            rel_update_thresh=0.01,
            rel_save_thresh=0.01,
            print_freq=100,
            ):
        """Train the wrapped model for a fixed epoch budget.

        This method preserves the original training behavior while making the
        control flow explicit: shuffle, batch updates, optional validation,
        early stopping, and periodic checkpoint logic.
        """

        if self.seed is not None:
            self.set_seed(self.seed)

        self.early_stopping = early_stopping

        if batch_size is None:
            batch_size = len(x_tr_input)
        train_batches_per_epoch = max(1, len(x_tr_input) // batch_size)

        if validation_data is not None:
            x_val, y_val = validation_data
            val_batch_size = batch_size
            val_batches_per_epoch = max(1, len(x_val) // val_batch_size)

        self.best_train_loss = getattr(self, "best_train_loss", float("inf"))
        self.best_val_loss = getattr(self, "best_val_loss", float("inf"))
        self.last_improved = getattr(self, "last_improved", 0)
        self.max_trigger = getattr(self, "max_trigger", 0)
        self.trigger_list = getattr(self, "trigger_list", [])
        global_start_time = time.time()
        self.print_freq = print_freq
        self.avg_epoch_time = None

        for epoch in range(initial_epoch, initial_epoch + epochs):

            current_epochs = getattr(self.model, "epochs", epoch)
            trigger = current_epochs - self.last_improved
            self.trigger_list.append(trigger)

            if trigger > self.max_trigger:
                self.max_trigger = trigger

            if early_stopping is not None and trigger >= early_stopping:
                print("\n\nEarly stopping – no improvement.")
                if self.save_name:
                    self.save(f"{self.save_name}_ES")
                    print(f"Saved model with early stopping at epoch {self.model.epochs}")
                break

            self.train, self.val = True, False
            epoch_start_time = time.time()
            self.model.train()

            torch.manual_seed(self.model.epochs if hasattr(self.model, "epochs") else epoch)

            # Shuffle once per epoch ----------------------------------
            if shuffle:
                perm = torch.randperm(len(x_tr_input))
                x_tr = x_tr_input[perm].data
                y_tr = y_tr_input[perm].data
            else:
                x_tr = x_tr_input.data
                y_tr = y_tr_input.data

            self.device = x_tr.device
            epoch_train_losses = []
            epoch_train_pde = []
            epoch_train_data = []

            # ---------------- batch loop ----------------------------
            for batch_idx in range(train_batches_per_epoch):
                if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                    break

                start = batch_idx * batch_size
                stop = (batch_idx + 1) * batch_size if batch_idx + 1 < train_batches_per_epoch else len(x_tr)

                x_true = x_tr[start:stop]
                y_true = y_tr[start:stop]

                x_true.requires_grad_(True)
                self.optimizer.zero_grad(set_to_none=True)

                y_pred = self.model(x_true)

                losses = self.loss(y_pred, y_true)
                task_loss, data_loss, pde_loss = losses[:3]

                reg_loss = self.regularizer(self.model, x_true, y_true, y_pred) if self.regularizer is not None else 0.0
                total_loss = task_loss + reg_loss
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

                epoch_train_losses.append(total_loss.detach())
                epoch_train_pde.append(pde_loss.detach())
                epoch_train_data.append(data_loss.detach())

                if hasattr(self.model, 'epochs') and self.model.epochs % 10 == 0:
                    self.save_index.append(self.model.loss_count - 1)

            # ---------------- end batch loop -------------------------
            train_loss_epoch = torch.mean(torch.stack(epoch_train_losses)).item()
            train_pde_epoch = torch.mean(torch.stack(epoch_train_pde)).item()
            train_data_epoch = torch.mean(torch.stack(epoch_train_data)).item()

            self.train_loss_list.append(train_loss_epoch)
            self.train_pde_loss_list.append(train_pde_epoch)
            self.train_data_loss_list.append(train_data_epoch)

            # ---------------- Validation step ------------------------
            if validation_data is not None and (epoch % validation_freq == 0):
                self._validate(x_val, y_val,
                               val_batches_per_epoch, val_batch_size,
                               validation_steps,
                               include_val_aug, include_val_reg,
                               self.best_val_loss, rel_save_thresh)

                if self.val_loss_list[-1] < self.best_val_loss * (1 - rel_update_thresh):
                    self.best_val_loss = self.val_loss_list[-1]
                    self.last_improved = self.model.epochs
                    self.best_diffusion_pred = self.model.D_scale * self.model.diffusion(self.model.u_vals_torch).flatten()
                if getattr(self.model, "growth", None) is not None:
                    self.best_growth_pred = self.model.G_scale * self.model.growth(self.model.u_vals_torch).flatten()

            if validation_data is None:
                if self.train_loss_list[-1] < self.best_val_loss * (1 - rel_update_thresh):
                    self.best_val_loss = self.train_loss_list[-1]
                    self.last_improved = self.model.epochs
                    self.best_diffusion_pred = self.model.D_scale * self.model.diffusion(self.model.u_vals_torch).flatten()
                if getattr(self.model, "growth", None) is not None:
                    self.best_growth_pred = self.model.G_scale * self.model.growth(self.model.u_vals_torch).flatten()

                    if self.train_loss_list[-1] < self.best_val_loss * (1 - rel_save_thresh):
                        if self.save_best_val and self.save_name:
                            self.save(f"{self.save_name}_best_val")

            # ---------------- Optional tracking of D/G ----------------
            if hasattr(self.model, "u_vals_torch") and hasattr(self.model, "diffusion") and hasattr(self.model, "D_scale"):
                if self.model.epochs % 10 == 0:
                    diff_pred = self.model.D_scale * self.model.diffusion(self.model.u_vals_torch).flatten()
                    self.diffusion_preds.append(diff_pred)

                    if getattr(self.model, "growth", None) is not None and hasattr(self.model, "G_scale"):
                        growth_pred = self.model.G_scale * self.model.growth(self.model.u_vals_torch).flatten()
                        self.growth_preds.append(growth_pred)

            synchronize_if_needed(x_tr)

            # Epoch-level progress message
            if verbose == 1 and epoch % self.print_freq == 0:

                elapsed, remaining, _ = TimeRemaining(
                    current_iter=self.model.epochs + 1,
                    total_iter=initial_epoch + epochs,
                    start_time=global_start_time,
                    previous_time=epoch_start_time,
                    ops_per_iter=batch_size)
                msg = (f"\rEpoch {self.model.epochs + 1}/{initial_epoch + epochs} | "
                       f"Train loss: {train_loss_epoch:1.4e}")
                if validation_data is not None and self.val_loss_list:
                    msg += f" | Val loss: {self.val_loss_list[-1]:1.4e}"
                msg += f" | Remaining: {remaining}        "
                msg += f" | Trigger = {trigger}"
                msg += f" | Elapsed = {epoch_start_time - global_start_time:.1f} s"
                msg += f" | Max trigger = {self.max_trigger}"

                print(msg, end='\r', flush=True)

            if epoch > 0:
                self.epoch_times.append(time.time() - epoch_start_time)

            if hasattr(self.model, 'epochs'):
                self.model.epochs += 1

        if early_stopping is None or trigger < early_stopping:
            print("\nNumber of epochs to train finished rather than early stopping.")
            if self.save_name:
                self.save(f"{self.save_name}_expired")
                print(f"Saved model at total trained epochs {self.model.epochs}")

        if verbose == 1:
            print("\nTraining finished.")
            print("\nTotal epochs trained =", self.model.epochs)
            print(f"\nBest val loss = {self.best_val_loss:.3e}")

    # ------------------------------------------------------------------
    # VALIDATION --------------------------------------------------------
    # ------------------------------------------------------------------

    def _validate(self,
                  x_val, y_val,
                  val_batches_per_epoch, val_batch_size,
                  validation_steps,
                  include_val_aug, include_val_reg,
                  best_val_loss, rel_save_thresh):
        """Run validation over mini-batches and update validation histories."""
        self.model.eval()

        val_loss_acc = 0.0
        val_pde_acc = 0.0
        val_data_acc = 0.0
        val_reg_acc = 0.0

        for idx in range(val_batches_per_epoch):
            if validation_steps is not None and idx >= validation_steps:
                break

            start = idx * val_batch_size
            stop = (idx + 1) * val_batch_size if idx + 1 < val_batches_per_epoch else len(x_val)
            x_true = x_val[start:stop].clone()
            y_true = y_val[start:stop].clone()

            if include_val_aug and hasattr(self, "augmentation") and self.augmentation is not None:
                x_true, y_true = self.augmentation(x_true, y_true)

            x_true.requires_grad_(True)
            y_pred = self.model(x_true)

            losses = self.loss(y_pred, y_true)
            val_loss_acc += losses[0]
            val_data_acc += losses[1]
            val_pde_acc += losses[2]

            if include_val_reg and self.regularizer is not None:
                val_reg_acc += self.regularizer(self.model, x_true, y_true, y_pred)

        batches = idx + 1
        val_total = (val_loss_acc + val_reg_acc) / batches
        val_pde = (val_pde_acc + val_reg_acc) / batches
        val_data = (val_data_acc + val_reg_acc) / batches

        self.val_loss_list.append(val_total.item())
        self.val_pde_loss_list.append(val_pde.item())
        self.val_data_loss_list.append(val_data.item())

        synchronize_if_needed(x_true)

        if val_total < best_val_loss * (1 - rel_save_thresh):
            if self.save_best_val and self.save_name:
                self.save(f"{self.save_name}_best_val")

    # ------------------------------------------------------------------
    # MISC UTILITIES ---------------------------------------------------
    # ------------------------------------------------------------------

    def set_seed(self, seed):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def predict(self, inputs):
        """
        Runs the model in evaluation mode on the provided inputs.
        """
        self.model.eval()
        return self.model(inputs)

    def save(self, save_name):
        """
        Saves the model, optimizer, and regularizer weights.
        """
        model_name = "{}_model".format(save_name)
        torch.save(self.model.state_dict(), model_name)

        torch.save({'epochs': self.model.epochs},
                   "{}_epochs".format(save_name))

        if self.save_opt and self.optimizer:
            torch.save(self.optimizer.state_dict(),
                       "{}_opt".format(save_name))

        if self.save_reg and self.regularizer:
            torch.save(self.regularizer.state_dict(),
                       "{}_reg".format(save_name))

    def load(self,
             model_weights,
             opt_weights=None,
             reg_weights=None,
             device=None):
        """
        Loads the model, optimizer, and regularizer weights.
        """
        self.model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=False))
        self.model.eval()

        if opt_weights:
            self.optimizer.load_state_dict(torch.load(opt_weights, map_location=device, weights_only=False))

        if reg_weights:
            self.regularizer.load_state_dict(torch.load(reg_weights, map_location=device, weights_only=False))

    def load_best_train(self, device=None, intro=''):
        self._load_best_weights(suffix='best_train', device=device, intro=intro)

    def load_best_val(self, device=None, intro=''):
        self._load_best_weights(suffix='best_val', device=device, intro=intro)

    def load_ES(self, device=None, intro=''):
        self._load_best_weights(suffix='ES', device=device, intro=intro)

    def load_expired(self, device=None, intro=''):
        self._load_best_weights(suffix='expired', device=device, intro=intro)

    def _load_best_weights(self, suffix, device, intro=''):
        """
        Helper for loading model/opt/reg weights with a given suffix.
        """
        model_name = intro + "{}_{}_model".format(self.save_name, suffix)
        self.model.load_state_dict(torch.load(model_name, map_location=device, weights_only=False))
        self.model.eval()

        if self.save_opt and self.optimizer:
            opt_name = "{}_{}_opt".format(self.save_name, suffix)
            self.optimizer.load_state_dict(torch.load(opt_name, map_location=device, weights_only=False))

        if self.save_reg and self.regularizer:
            reg_name = "{}_{}_reg".format(self.save_name, suffix)
            self.regularizer.load_state_dict(torch.load(reg_name, map_location=device, weights_only=False))