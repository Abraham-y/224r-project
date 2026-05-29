"""Ray actor that applies policy-gradient updates for RLOO.

The orchestrator (`rloo.py`) samples responses and computes rewards, then
calls this worker with tokenized sequences to perform gradient updates.

This file is intentionally incomplete. Students are expected to implement
`update(...)` while reusing the data/model/sampling setup provided here.
"""

import os
import warnings
import ray
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")

@ray.remote(num_gpus=1)
class RLOOUpdateWorker:
    """Owns policy/ref models and optimizer state for RLOO updates."""
    def __init__(
        self, 
        model_path, 
        optimizer_path, 
        scheduler_path,
        tokenizer_path=None, 
        ref_model_path=None,
        batch_size=64,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        group_size=16, 
        entropy_coefficient=0.01, 
        kl_divergence_coefficient=0.0, 
        lr_schedule='constant',
        learning_rate=1e-5, 
        weight_decay=0.01, 
        warmup_ratio=0.0,
        num_training_steps=250,
    ):
        self.model_path = model_path
        self.ref_model_path = ref_model_path if ref_model_path is not None else model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.group_size = group_size
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2 for RLOO, got {self.group_size}")
        self.entropy_coefficient = entropy_coefficient
        self.kl_divergence_coefficient = kl_divergence_coefficient
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        if warmup_ratio > 0:
            raise NotImplementedError("Warmup ratio > 0 is not supported for constant learning rate schedule")
        self.num_training_steps = num_training_steps

    def tear_down(self):
        """Release model/optimizer objects and clear GPU memory."""
        import gc
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'ref_model'):
            del self.ref_model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'scheduler'):
            del self.scheduler
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_checkpoint_paths(self, model_path, optimizer_path, scheduler_path, load_checkpoint=False):
        """Update output paths (and optionally reload state immediately)."""
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        if load_checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load policy model, optional reference model, and optimizer/scheduler."""
        self.tear_down()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device="cuda")
        self.model.gradient_checkpointing_enable()

        if self.kl_divergence_coefficient > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.ref_model_path,
                torch_dtype=torch.bfloat16,
            ).to(device="cuda")
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        if self.optimizer_path and self.scheduler_path and os.path.exists(self.optimizer_path) and os.path.exists(self.scheduler_path):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")
            
            self.scheduler.load_state_dict(torch.load(self.scheduler_path))
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")

        self.model.train()

    def save_checkpoint(self):
        """Persist optimizer/scheduler state plus model+tokenizer weights."""
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
        torch.save(self.scheduler.state_dict(), self.scheduler_path)

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)


    def update_gradient_accumulation(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        device='cuda',
    ):
        """Split incoming batch into microbatches and call `update(...)`."""
        update_metrics = None
        if self.gradient_accumulation_steps > 1:
            curr_batch_size = input_ids.shape[0]
            assert curr_batch_size % self.gradient_accumulation_steps == 0, (
                f"Flattened batch size {curr_batch_size} must be divisible by gradient_accumulation_steps "
                f"{self.gradient_accumulation_steps}."
            )
            group_per_gradient_accumulation_step = curr_batch_size // self.gradient_accumulation_steps
            # Ensure each microbatch still contains full RLOO groups so the baseline is meaningful
            assert group_per_gradient_accumulation_step % self.group_size == 0, (
                f"Microbatch size {group_per_gradient_accumulation_step} must be divisible by group_size {self.group_size} "
                f"when using gradient_accumulation_steps={self.gradient_accumulation_steps}."
            )
            all_metrics = []
            for i in range(self.gradient_accumulation_steps):
                curr_input_ids = input_ids[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_attention_mask = attention_mask[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_is_response_token = is_response_token[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_rewards = rewards[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_sample_log_probs = None
                if sample_log_probs is not None:
                    curr_sample_log_probs = sample_log_probs[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                
                is_update_step = (i == self.gradient_accumulation_steps - 1)
                curr_update_metrics = self.update(
                    curr_input_ids,
                    curr_attention_mask,
                    curr_is_response_token,
                    curr_rewards,
                    curr_sample_log_probs,
                    is_update_step,
                    device,
                )
                all_metrics.append(curr_update_metrics)
            update_metrics = {}
            # Some metrics (grad_norm, lr, weight_*) are only defined on the final
            # microbatch and are NaN otherwise, so average while ignoring NaNs.
            for metric_name in all_metrics[0].keys():
                values = [metric[metric_name] for metric in all_metrics]
                if np.all(np.isnan(values)):
                    update_metrics[metric_name] = float('nan')
                else:
                    update_metrics[metric_name] = np.nanmean(values).item()
        else:
            update_metrics = self.update(
                input_ids,
                attention_mask,
                is_response_token,
                rewards,
                sample_log_probs,
                True,
                device,
            )

        return update_metrics

    # `is_update_step` is False on intermediate microbatches so we can
    # accumulate gradients before stepping optimizer/scheduler.
    def update(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        is_update_step: bool = True,
        device='cuda',
    ):
        max_importance_weight=10.0

        input_ids = torch.as_tensor(input_ids, dtype= torch.long, device=device)
        attention_mask = torch.as_tensor(attention_mask, dtype= torch.long, device=device)
        is_response_token = torch.as_tensor(is_response_token, dtype=torch.long, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        response_mask = (is_response_token[:, 1:] * attention_mask[:, 1:]).to(dtype=logits.dtype)

        per_token_log_probs = -F.cross_entropy(
            predictions.transpose(1, 2), targets, reduction='none'
        )
        seq_log_probs = (per_token_log_probs * response_mask).sum(dim=1)

        probs = torch.softmax(predictions, dim=-1)
        per_token_entropy = torch.logsumexp(predictions, dim=-1) - (probs * predictions).sum(dim=-1)
        del probs
        response_token_count = response_mask.sum().clamp(min=1.0)
        entropy = (per_token_entropy * response_mask).sum() / response_token_count

        grouped_rewards = rewards.view(-1, self.group_size)
        group_sum = grouped_rewards.sum(dim=1, keepdim=True)
        baseline = (group_sum - grouped_rewards)/(self.group_size - 1)
        advantages = (grouped_rewards - baseline).reshape(-1)

        if sample_log_probs is not None:
            behavior_log_probs = torch.as_tensor(sample_log_probs, dtype=torch.float32, device=device)
            log_importance_ratio = seq_log_probs.detach() - behavior_log_probs
            log_importance_ratio = log_importance_ratio.clamp(max=float(np.log(max_importance_weight)))
            importance_weights = torch.exp(log_importance_ratio)
        else:
            importance_weights = torch.ones_like(seq_log_probs)

        pg_loss = -(importance_weights * advantages * seq_log_probs).mean()

        loss = pg_loss - self.entropy_coefficient * entropy

        kl = None
        if self.kl_divergence_coefficient > 0:
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits
                ref_per_token_log_probs = -F.cross_entropy(
                    ref_logits[:, :-1, :].transpose(1, 2), targets, reduction='none'
                )
                del ref_logits
            log_ratio_ref = ref_per_token_log_probs - per_token_log_probs
            per_token_kl = log_ratio_ref.exp() - log_ratio_ref - 1.0
            kl = (per_token_kl * response_mask).sum() / response_token_count
            loss = loss + self.kl_divergence_coefficient * kl


        (loss / self.gradient_accumulation_steps).backward()

        grad_norm = float('nan')
        weight_metrics = {
            "weight_mse": float('nan'),
            "weight_max_abs_diff": float('nan'),
            "weight_nonzero_diff_ratio": float('nan'),
            "weight_changed_tensor_ratio": float('nan'),
        }
        if is_update_step:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            old_params = [p.detach().clone() for p in trainable_params]

            if self.gradient_clipping > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping))
            else:
                grads = [p.grad.detach().norm() for p in trainable_params if p.grad is not None]
                grad_norm = float(torch.norm(torch.stack(grads))) if grads else 0.0

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                total_sq_diff = 0.0
                total_numel = 0
                total_nonzero = 0
                max_abs_diff = 0.0
                changed_tensors = 0
                for old, new in zip(old_params, trainable_params):
                    diff = (new.detach() - old).float()
                    numel = diff.numel()
                    total_sq_diff += float((diff * diff).sum())
                    total_numel += numel
                    nonzero = int((diff != 0).sum())
                    total_nonzero += nonzero
                    if numel > 0:
                        max_abs_diff = max(max_abs_diff, float(diff.abs().max()))
                    if nonzero > 0:
                        changed_tensors += 1
            weight_metrics = {
                "weight_mse": total_sq_diff / max(total_numel, 1),
                "weight_max_abs_diff": max_abs_diff,
                "weight_nonzero_diff_ratio": total_nonzero / max(total_numel, 1),
                "weight_changed_tensor_ratio": changed_tensors / max(len(trainable_params), 1),
            }

        with torch.no_grad():
            metrics = {
                "loss": float(loss),
                "pg_loss": float(pg_loss),
                "entropy": float(entropy),
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std(unbiased=False)),
                "rollout_accuracy": float((rewards == 1.0).float().mean()),
                "advantage_mean": float(advantages.mean()),
                "advantage_std": float(advantages.std(unbiased=False)),
                "advantage_abs_mean": float(advantages.abs().mean()),
                "importance_weight_mean": float(importance_weights.mean()),
                "importance_weight_max": float(importance_weights.max()),
                "seq_log_prob_mean": float(seq_log_probs.mean()),
                "response_length_mean": float(response_mask.sum(dim=1).mean()),
                "grad_norm": grad_norm,
                "lr": float(self.scheduler.get_last_lr()[0]),
                "kl_loss": float(kl) if kl is not None else float('nan'),
                **weight_metrics,
            }
        return metrics
