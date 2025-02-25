# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model_and_tokenizer
from ...train.sft.metric import ComputeMetrics
from ...train.sft.trainer import CustomSeq2SeqTrainer
from ...train.utils import create_modelcard_and_push
from typing import Dict, Optional, Sequence
from functools import partial
import torch
import os
import json
import transformers
from itertools import islice
if TYPE_CHECKING:
    from transformers import TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

####################
##训练MetaMath开始####
####################   
# DEFAULT_PAD_TOKEN = "<pad>"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"
####################
##训练MetaMath结束####
####################  
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "</s>"
# DEFAULT_UNK_TOKEN = "</s>"

def set_global_tokens(model_name):
    global DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN

    if "MetaMath" in model_name:
        DEFAULT_PAD_TOKEN = "<pad>"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
    else:
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "</s>"
        DEFAULT_UNK_TOKEN = "</s>"

        
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
def hook_grad_up(module, output, up_weight_list):
     # 获取输入梯度的第一个元素，它对应于线性层的权重梯度
    grad_weight = output[0]  # 假设是权重梯度
    import pdb
    pdb.set_trace()
    # 获取梯度的形状，例如 [2, 264, 11008]
    # 想要随机将一半神经元的梯度设为0，需要对最后一个维度操作
    mask = torch.zeros(output[0].shape[-1], dtype=torch.bool)
    up_weight_list = [int(num) for num in up_weight_list]
    output_shape = output[0].size()
    mask[up_weight_list] = True
    mask = mask[None, None, :]
    mask_expanded = mask.expand(output_shape[0], output_shape[1], output_shape[2])
    output[0][~mask_expanded] = 0  # 使用~mask将mask为False的位置设置为0
    for index in up_weight_list:
        output[0][:,:, int(index)] = 0

    return (grad_weight,) + output[1:]

def hook_grad_down(module, output, down_weight_list):
     # 获取输入梯度的第一个元素，它对应于线性层的权重梯度
    grad_weight = output[0]  # 假设是权重梯度

    # 获取梯度的形状，例如 [2, 264, 11008]
    # 想要随机将一半神经元的梯度设为0，需要对最后一个维度操作
    if grad_weight is not None:
        neuron_count = grad_weight.shape[-1]  # 获取神经元数量
        mask = torch.rand(neuron_count) < 0.5  # 随机生成一个掩码，50% 的几率为 True
        
        # 创建一个全局 mask，扩展到 grad_weight 的形状
        mask = mask[None, None, :].expand_as(grad_weight)  # 假设 grad_weight 形状为 [2, 264, 11008]

        # 将选中的神经元梯度设置为0
        grad_weight[mask] = 0

    # 返回修改后的梯度
    return (grad_weight,) + output[1:]

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    # for layer in finetuning_args.special_train_layers:
    #     target_dir = "/localnvme/application/sc_new/fmm/evaluate/evaluate/weight_analsiy"
        
    #     down_file = os.path.join(target_dir, "down_weight_{}.json".format(layer))
    #     up_file  = os.path.join(target_dir, "up_weight_{}.json".format(layer))
        
    #     with open(up_file, 'r') as file:
    #         up_weight = json.load(file)
    #     with open(down_file, 'r') as file:
    #         down_weight = json.load(file)
            
    #     num_up_elements = int(len(up_weight) * 0.5)
    #     num_down_elements = int(len(down_weight) * 0.5)
        
    #     up_weight_select = dict(islice(up_weight.items(), num_up_elements))
    #     down_weight_select = dict(islice(down_weight.items(), num_down_elements))
        
    #     up_weight_list = list(up_weight_select.keys())
    #     down_weight_list = list(down_weight_select.keys())

    #     hook_grad_up_ = partial(hook_grad_up, up_weight_list=up_weight_list)
    #     hook_grad_down_ = partial(hook_grad_down , down_weight_list=down_weight_list)
        
    #     handle1 = model.model.layers[layer].mlp.up_proj.register_full_backward_pre_hook(hook_grad_up_)
    #     handle2 = model.model.layers[layer].mlp.down_proj.register_full_backward_pre_hook(hook_grad_down_)
    # 设置全局变量
    set_global_tokens(model_args.model_name_or_path)  
    if "MetaMath" in model_args.model_name_or_path:   
        ####################
        ##训练MetaMath开始####
        ####################
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        print("训练MetaMath")
        ####################
        ##训练MetaMath结束####
        ####################  
    elif "WizardMath" in model_args.model_name_or_path:  
        ####################
        ##训练WizardMath开始####
        ####################
        smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )
        print("训练WizardMath")
    #     ####################
    #     ##训练WizardMath结束####
    #     ####################
    # elif "MuggleMath" in model_args.model_name_or_path:
    #     if tokenizer.pad_token is None:
    #         smart_tokenizer_and_embedding_resize(
    #             special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    #             tokenizer=tokenizer,
    #             model=model,
    #         )
    #     # if "llama" in base_model:
    #     tokenizer.add_special_tokens(
    #         {
    #             "eos_token": DEFAULT_EOS_TOKEN,
    #             "bos_token": DEFAULT_BOS_TOKEN,
    #             "unk_token": DEFAULT_UNK_TOKEN,
    #         }
    #     )
        
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation
    

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        # pin_memory=False
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(
        dict(
            generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
            generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams,
        )
    )
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
