import torch
import random
from torch.optim import Optimizer
from torch import Tensor
from collections import defaultdict
from typing import List, Optional, Dict, Union, Iterable
import time
import math
import warnings
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import numpy as np
# Optional [0, 1, 2]. 
    # 0: no print
    # 1: print the relative time whenever a parameter's grad is ready
    # 2: for debug usage only. Will set all the parameters trainable, print the grad ready time for each parameter. 
    #     In this case, all the grad except the "specified" trainable parameters will be set to None after being calculated.
BACKWARD_VERBOSE = 0

class LisaOptimizer(Optimizer):
    """Wrap the original optimizer to update trainable parameters periodically based on number of activated layers."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        named_parameters_list,
        lisa_activated_layers = 1,
        lisa_interval_steps = 5,
        active_modules: List[str] = [],
        include_embedding=True,
        include_lm_head=True,
        verbose: int = 1,
        log_fn = None,
    ):
        """
        Args:
            base_optimizer (Optimizer): The base optimizer being wrapped by the BlockOptimizer.
            named_parameters_list: A function that generates the named parameters of the model.
            block_prefix_list (List[List[str]]): The list of blocks of parameters to be updated.
            lisa_interval_steps (int, optional): The number of optimization steps before switching to the next block. Defaults to 10.
            start_block (Optional[int], optional): The index of the block to start with. Defaults to None.
            switch_mode (str, optional): The mode for switching between different blocks of parameters. Defaults to "descending".
            active_modules (List[str]): The list of modules that are always active during optimization. Defaults to None.
            verbose (int, optional): The verbosity level for printing information during optimization. Defaults to 1.
            log_fn: A logging function for recording information during optimization. Defaults to None.
        """
        # print(base_optimizer)
        block_prefix_list, other_params = self.infer_param_groups([n for n, _ in named_parameters_list], include_embedding, include_lm_head)

        switch_mode = "random"
        assert isinstance(block_prefix_list, list)
        self.lisa_activated_layers = lisa_activated_layers
        self.lisa_interval_steps = lisa_interval_steps
        self.verbose = verbose
        self.switch_mode = switch_mode
        self.named_parameters_list = named_parameters_list
        self.weight_decay = base_optimizer.param_groups[0]["weight_decay"]
        self.block_prefix_list = block_prefix_list
        self.other_params = other_params
        self.block_num = len(block_prefix_list)
        self.log_fn = log_fn
        self.global_step = 0
        self.base_optimizer = base_optimizer
        self.active_modules = active_modules
        self.defaults = base_optimizer.defaults

        self.param_groups = base_optimizer.param_groups
        self.state_dict = base_optimizer.state_dict # for compatibility of hf Trainer
    

        # lora not supported in Lisa
        self.lora_mode = False
            
        if any(isinstance(p, torch.FloatTensor) for _, p in named_parameters_list):
            warnings.warn("BAdam expect model to be loaded in fp16 precision while detect fp32 weight. \
                This will cause additional memory usage and lose the benefit of mixed precision training.")
            
        super().__init__(self.param_groups, base_optimizer.defaults)
        
        if BACKWARD_VERBOSE:
            self.record_mark = True
            self.ordered_named_params = []
            self.param_num = len(named_parameters_list)
            for n, p in named_parameters_list:
                p.register_post_accumulate_grad_hook(self.test_hook(n))

        self.update_trainable_params()

        if BACKWARD_VERBOSE == 2:
            for name, param in self.named_parameters_list:
                param.requires_grad_(True)
    
    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p
    
    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p

    def infer_param_groups(self, param_names, include_embedding, include_lm_head):
        """automatic inference of the parameter groups based on the parameter names.
        divide groups into:
            * embedding
            * transformer layers
            * lm_head and others
        """
        import re
        
        block_prefix_list = []
        other_params = []
        embed_pattern = r'.*embed[^.]*\.'
        layer_pattern = r'.*layers.[^.]*\.'

        for name in param_names:
            if any(prefix[0] in name for prefix in block_prefix_list):
                continue
            
            if re.findall(layer_pattern, name):
                block_prefix_list.append(re.findall(layer_pattern, name))
            else: 
                other_params.append(name)
            # elif re.findall(embed_pattern, name) and include_embedding:
            #     embed_list.append(re.findall(embed_pattern, name)[0])
            # else:
            #     lm_head_and_other_params.append(name)
        
        # if include_lm_head:
        #     block_prefix_list.append(lm_head_and_other_params)
        # print("checking")
        # for i in range(10) :
        #     print(block_prefix_list)
        print(block_prefix_list)
        print(other_params)
        return block_prefix_list, other_params
                
    def test_hook(self, name):
        """hook used for recording the time of gradient calculation, see comments on BACKWARD_VERBOSE for more details."""
        
        def func(x):
            if self.record_mark:
                self.backward_start_time = time.time()          
                self.record_mark = False
                relative_time = 0.
            else:
                relative_time = time.time() - self.backward_start_time
            if any(p_name in name for p_name in self.active_param_prefixs):
                print(f"param: {name:<50} relative time: {relative_time}")
            
            iterator = self.named_parameters_list
                
            for n, p in iterator:
                
                if p.requires_grad and p.grad is not None:
                    print("parameter name: ", n, "relative time", time.time() - self.backward_start_time)
                    
                    if (not any(p_name in n for p_name in self.active_param_prefixs)) and \
                        BACKWARD_VERBOSE == 2:
                        p.grad = None
                    
                    if len(self.ordered_named_params) < self.param_num:
                        self.ordered_named_params.append((n, p))
                    # break since for each step only one parameter's grad is updated
                    break
            return x
        
        return func

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        return self.base_optimizer.load_state_dict(state_dict)
    
    def _update_lr(self):
        # Make sure the learning rate of the base_optimizer is consistent with the BlockOptimizer
        for group in self.base_optimizer.param_groups:
            group["lr"] = self.param_groups[0]["lr"]

    def step(self, *args, **kwargs) -> None:
        self.record_mark = True

        self._update_lr()
        self._grad_to_hp()
        self.base_optimizer.step(*args, **kwargs)
        self._update_param()
        self._clean_hp_grad()
        self.global_step += 1

        torch.cuda.empty_cache()
        if (self.global_step + 1) % self.lisa_interval_steps == 0:
            self.update_trainable_params()

    def _clean_hp_grad(self) -> None:
        """Clean the gradients of the high precision parameters."""
        for hp_param in self.param_idx2hp.values():
            hp_param.grad = None

    def _update_param(self) -> None:
        """Update the low precision parameters with the values of the high precision parameters."""
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            lp_param.data.copy_(hp_param.to(lp_param.dtype).data)

    def _grad_to_hp(self, clear_lp_grads: bool = True) -> None:
        """
        Convert the gradients of the low precision parameters to high precision and calculate the gradient norm.

        Args:
            clear_lp_grads (bool, optional): Whether to clear the gradients of the low precision parameters. Defaults to True.
        """
        grad_norm = 0.0
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            assert lp_param.grad is not None, "The low precision parameter's gradient is None."
            hp_param.grad = lp_param.grad.float()

            if clear_lp_grads:
                lp_param.grad = None

    def update_trainable_params(self, verbose: Optional[int] = None) -> None:
        """
        Update the trainable parameters based on the current block index and the specified verbosity level.

        Args:
            verbose (Optional[int], optional): The verbosity level for printing information. Defaults to None.
        """
        if verbose is None:
            verbose = self.verbose

        
        
        active_layers_indices = np.random.choice(range(self.block_num), self.lisa_activated_layers, replace=False)
        print(f"Activating layers at indices: {active_layers_indices} for the next steps.", flush=True)
        
        # self.active_param_prefixs = self.block_prefix_list[active_layers_indices]
        self.active_param_prefixs = []
        
        for i in active_layers_indices :
            self.active_param_prefixs.append(self.block_prefix_list[i][0])
        # for i in range(10) :
        #     print(self.active_param_prefixs)
        
        # if verbose >= 1:
        #     print("Parameters with the following prefix will be trainable:", self.active_param_prefixs)

        # Reset parameters to be optimized
        self.param_idx2lp = {}
        self.param_idx2hp = {}
        
        active_param_groups = [
            {
                "params": [],
                "weight_decay": self.param_groups[0]['weight_decay'],
                **self.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.defaults
            },
        ]
        for i, (name, param) in enumerate(self.named_parameters_list):
            if not any (p in name for p in self.other_params) and not any(p in name for p in self.active_param_prefixs):
                param.requires_grad_(False)
                param.grad = None
                # print("NOT activated name: ", name)
            else:
                # print("activated name: ", name)
                if self.lora_mode and "lora" not in name:
                    continue
                param.requires_grad_(True)
                param_hp = param.clone().float().detach().to(param.device)
                param_hp.requires_grad = True
                
                self.param_idx2lp[i] = param
                self.param_idx2hp[i] = param_hp
                
                if "bias" not in name and not isinstance(param, tuple(ALL_LAYERNORM_LAYERS)):
                    active_param_groups[0]['params'].append(param_hp)
                else:
                    active_param_groups[1]['params'].append(param_hp)
                
                if verbose >= 2:
                    print(name)

        self.base_optimizer.param_groups = active_param_groups
        
        import gc
        gc.collect()
        # Clean the optimizer state
        self.base_optimizer.state = defaultdict(lambda: {})


# For torch>=2.1, `_foreach_norm` is used when implementing `clip_grad_norm_`, which doesn't support sparse tensor yet.
# We can temporarily fix this issue by using the older torch version's implementation:
    # self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
def clip_grad_norm_for_sparse_tensor(self, parameters, max_norm, norm_type=2):
    """
    Modification of the accelerator.clip_grad_norm_ to enable gradient clipping for sparse tensor.
    Used for torch version >= 2.1
    """
    from accelerate.utils import DistributedType
    from torch import inf

    if self.distributed_type == DistributedType.FSDP:
        self.unscale_gradients()
        parameters = [p for p in parameters]
        for model in self._models:
            if parameters == [p for p in model.parameters()]:
                return model.clip_grad_norm_(max_norm, norm_type)
    elif self.distributed_type == DistributedType.DEEPSPEED:
        # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed
        # We cannot return the gradient norm because DeepSpeed does it.
        return None
    self.unscale_gradients()
    
    def clip_func_(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]], max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
        r""" torch 1.13 version clip_grad_norm_, works well with sparse tensor.
        Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [p.grad for p in parameters if p.grad is not None]
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if len(grads) == 0:
            return torch.tensor(0.)
        device = grads[0].device
        if norm_type == inf:
            norms = [g.detach().abs().max().to(device) for g in grads]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        return total_norm
    
    return clip_func_(parameters, max_norm, norm_type=norm_type)