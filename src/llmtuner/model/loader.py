from typing import TYPE_CHECKING, Optional, Tuple

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import AutoModelForCausalLMWithValueHead

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, get_current_device, try_download_model_from_ms
from .adapter import init_adapter
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
from .utils import load_valuehead_params, register_autoclass
import torch
import random

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    add_valuehead: Optional[bool] = False,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    try_download_model_from_ms(model_args)

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **config_kwargs,
    )

    patch_tokenizer(tokenizer)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    patch_config(config, tokenizer, model_args, config_kwargs, is_trainable)

    model = None
    if is_trainable and model_args.use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore

        unsloth_kwargs = {
            "model_name": model_args.model_name_or_path,
            "max_seq_length": model_args.model_max_length,
            "dtype": model_args.compute_dtype,
            "load_in_4bit": model_args.quantization_bit == 4,
            "token": model_args.hf_hub_token,
            "device_map": {"": get_current_device()},
            "rope_scaling": getattr(config, "rope_scaling", None),
        }
        try:
            model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
        except NotImplementedError:
            logger.warning("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
            model_args.use_unsloth = False

        if model_args.adapter_name_or_path:
            model_args.adapter_name_or_path = None
            logger.warning("Unsloth does not support loading adapters.")

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=model_args.compute_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs,
        )

    patch_model(model, tokenizer, model_args, is_trainable)
    register_autoclass(config, model, tokenizer)

    if finetuning_args.special_train_layers and finetuning_args.random_layers is not None:
        print("两个参数都有值. Exiting...")
        exit()
    
    if finetuning_args.only_mlp and finetuning_args.only_attention:
        print("不能同时训mlp和attention模块. Exiting...")
        exit()
        
    # 第一阶段
    if finetuning_args.special_train_layers!= None:
    # 只训某些层
        for name, param in model.named_parameters():
            param.requires_grad = False

        if finetuning_args.only_mlp:
            for name, param in model.named_parameters():
                for layer in finetuning_args.special_train_layers:
                    # if "layers.{}.".format(layer) in name:
                    if f"layers.{layer}.mlp.up_proj" in name or f"layers.{layer}.mlp.down_proj" in name:
                        param.requires_grad = True
                        print(f"{layer} 的梯度被设置为False: {param.requires_grad}")
                        
        elif finetuning_args.only_attention:
            for name, param in model.named_parameters():
                for layer in finetuning_args.special_train_layers:
                    # if "layers.{}.".format(layer) in name:
                    if f"layers.{layer}.self_attn.q_proj" in name or f"layers.{layer}.self_attn.k_proj" in name or f"layers.{layer}.self_attn.v_proj" in name or f"layers.{layer}.self_attn.o_proj"in name:
                        param.requires_grad = True
                        print(f"{layer} 的梯度被设置为False: {param.requires_grad}")
            
        else:
            for name, param in model.named_parameters():
                for layer in finetuning_args.special_train_layers:
                    if "layers.{}.".format(layer) in name:
                        param.requires_grad = True
                        print(f"{layer} 的梯度被设置为False: {param.requires_grad}")
                        
    elif finetuning_args.random_layers:

        random_layers = [random.randint(0, 31) for _ in range(finetuning_args.random_layers)]
        # 只训某些层
        for name, param in model.named_parameters():
            param.requires_grad = False

        if finetuning_args.only_mlp:
            for name, param in model.named_parameters():
                for layer in random_layers:
                    # if "layers.{}.".format(layer) in name:
                    if f"layers.{layer}.mlp.up_proj" in name or f"layers.{layer}.mlp.down_proj" in name:
                        param.requires_grad = True
                        print(f"{layer} 的梯度被设置为False: {param.requires_grad}")
                        
        elif finetuning_args.only_attention:
            for name, param in model.named_parameters():
                for layer in finetuning_args.special_train_layers:
                    # if "layers.{}.".format(layer) in name:
                    if f"layers.{layer}.self_attn.q_proj" in name or f"layers.{layer}.self_attn.k_proj" or f"layers.{layer}.self_attn.v_proj"or f"layers.{layer}.self_attn.o_proj"in name:
                        param.requires_grad = True
                        print(f"{layer} 的梯度被设置为False: {param.requires_grad}")
        else:
            for name, param in model.named_parameters():
                for layer in random_layers:
                    if "layers.{}.".format(layer) in name:
                        param.requires_grad = True
                        print(f"{layer} 的梯度被设置为False: {param.requires_grad}")
     
        print("输出随机的层{}".format(random_layers))
    #训练Embedding层
    if finetuning_args.train_emb:
        for name, param in model.named_parameters():
            if f"embed_tokens" in name:
                param.requires_grad = True
    model = init_adapter(model, model_args, finetuning_args, is_trainable)
    

    print("训练最终的梯度")

    for name, param in model.named_parameters():
        print(f"{name} 的梯度: {param.requires_grad}")
        
    if add_valuehead:
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        model = model.to(model_args.compute_dtype) if not getattr(model, "quantization_method", None) else model
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    logger.info(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model, tokenizer