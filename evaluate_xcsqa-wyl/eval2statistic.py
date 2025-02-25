import json
import re
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Sequence, List
import argparse
import os
import shutil
import pdb
import csv
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

def main(
        args,
        xcsqa_test_jsonl: str = "/localnvme/application/sc_new/fmm/LLaMA-Factory-backup/X-CSR_datasets/X-CSQA",
        is_bf16: bool = True,
        save_dir: str = None,
):
    batch_size = args.batch_size
    print(f"main start, is_bf16:{is_bf16}, batch_size:{batch_size}")

    model_path = args.model_path
    model, tokenizer = get_model(model_path, is_bf16=is_bf16)
    print("model loaded")

    batch_llama = get_batch_llama(model, tokenizer)

    if save_dir is None:
        save_dir = f"{model_path}/x_csqa_with_CoT"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if args.lang_only is None:
        langs = ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "sw", "ur", "vi", "zh"]
    else:
        langs = args.lang_only
    sources = []
    targets = []
    results = {}
    for lang in langs:
        print(f'===========we are testing in {lang}====================')

        if args.streategy == 'Parallel':
            if lang == 'En_gsm8k':
                with open('./data/test_use.jsonl', "r", encoding='utf-8') as f:
                    xcsqa_datas = [json.loads(line) for line in f]

            else:
                with open(f'{xcsqa_test_jsonl}/{lang}/dev-input.json', "r", encoding='utf-8') as f:
                    xcsqa_datas = [json.loads(line) for line in f]

        gen_datas_jsonl = Path(save_dir) / f"gen_x_csqa_{lang}_datas-zhu2.jsonl"
        start_index = (
            len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
        )
        print(f"start_index: {start_index}")

        for i in tqdm(range(start_index, len(xcsqa_datas), batch_size)):
            cur_xcsqa_batch = xcsqa_datas[i: i + batch_size]
            input_str_list, output_str_list = xcsqa_batch_gen(lang,
                                                              [d["instruction"] for d in cur_xcsqa_batch], batch_llama
                                                              # [d["instruction"] for d in cur_gsm8k_batch], batch_llama
                                                              )
            for j, (xcsqa_data, input_str, output_str) in enumerate(
                    zip(cur_xcsqa_batch, input_str_list, output_str_list)
            ):
                with open(gen_datas_jsonl, "a", encoding='utf-8') as f:
                    json.dump(
                        dict(
                            index=i + j,
                            xcsqa_data=xcsqa_data,
                            input_str=input_str,
                            output_str=output_str,
                        ),
                        f,
                        ensure_ascii=False  # 设置此参数为False以避免转义非ASCII字符
                    )
                    f.write("\n")

        # calculate acc
        # with open(gen_datas_jsonl) as f:
        #     gen_datas = [json.loads(line) for line in f]

    #     correct_results = []
    #     wrong_results = []
    #     for gen in gen_datas:
    #         result = dict(
    #             **gen,
    #             extract_true_num=extract_last_num(gen["xcsqa_data"]["output"]),
    #             extract_pred_num=extract_last_num(gen["output_str"]),
    #             is_correct=None,
    #         )
    #         if abs(result["extract_true_num"] - result["extract_pred_num"]) < 1e-3:
    #             result["is_correct"] = True
    #             correct_results.append(result)
    #         else:
    #             result["is_correct"] = False
    #             wrong_results.append(result)

    #     print(f'=======done {lang}============')
    #     result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results) / (len(correct_results) + len(wrong_results))}"
    #     print(result)
    #     with open(Path(save_dir) / f"{lang}_correct.json", "w", encoding='utf-8') as f:
    #         json.dump(correct_results, f, ensure_ascii=False, indent=4)
    #     with open(Path(save_dir) / f"{lang}_wrong.json", "w", encoding='utf-8') as f:
    #         json.dump(wrong_results, f, ensure_ascii=False, indent=4)
    #     num_result = float(result.split('=')[-1])
    #     if lang != 'En_gsm8k':
    #         results[lang] = num_result
    #     else:
    #         gsm8k = num_result
    # average = sum(results.values()) / len(results)
    # print(average)
    # import csv
    # with open(Path(save_dir) / f"MSGM_evaluate_results_bs{batch_size}.csv", 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Language', 'Accuracy'])
    #     for key, value in results.items():
    #         writer.writerow([key, value])
    #     writer.writerow(['Average', average])
    #     # writer.writerow(['GSM8K', gsm8k])
    # #加入res_Statistic内容
    res_statistic(langs, args, save_dir)



def eval_res(data_path,have_error_model,model, basepath, lang):
    correct_results = []
    wrong_results = []
    with open(data_path, "r+", encoding="utf-8") as f:
        lines = f.readlines()
        total_num = len(lines)
        gold_cnt = 0
        for idx, line in enumerate(lines):
            line = json.loads(line)
            label = line['xcsqa_data']['output']
            res = line['output_str'].strip("\n").strip()

            pattern = re.compile(r"The answer is \((.)\)\.")
            match = pattern.search(res)
            if match:
                res = match.group(1)
            else:
                print(f"res_path : {data_path} idx : {idx} ; line:{line} Empty!")
                res = "F"
            if res not in ['A', 'B', 'C', 'D', 'E']:
                print(f"res_path : {data_path} idx : {idx} ; line:{line} output Error!")
                # 记录错误结果
                wrong_results.append(line)
                have_error_model.add(model)
            # 提取label
            match = pattern.search(label)
            label = match.group(1)

            if res == label:
                gold_cnt += 1
                correct_results.append(line)
            else:
                wrong_results.append(line)
    # 将正确和错误的结果写入对应的文件

    write_results(os.path.join(basepath, f"{lang}_correct.jsonl"), correct_results)
    write_results(os.path.join(basepath, f"{lang}_wrong.jsonl"), wrong_results)
    return gold_cnt / total_num

def write_results(file_path, results):
    with open(file_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def res_statistic(langs, args, save_dir):
    # langs=[ 'PL_trans', 'RU_trans', 'ZH_trans' ,'ar_trans', 'he_trans', 'en', 'DE_trans', 'JA_trans', 'FR_trans',  'IT_trans']# x_csqa
    #langs = ['ZH_trans']  # x_csqa s
    copa_langs = ['DE', 'JA', 'FR', 'PL', 'RU', 'ar', 'he', 'it', 'en', 'zh']  # x_copa
    name_geo_langs = ['it', 'pl', 'ru', 'en', 'de', 'ja', 'fr', 'zh', 'ar', 'he']  # x_name,x_geo
    basepath = f"{args.model_path}/x_csqa_with_CoT"
    models = ['baichuan2_7b_base']
    #langs = ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "sw", "ur", "vi", "zh"]
    # task_zu =[['x_csqa'],['x_name','x_geo'],['x_copa']]
    task_zu = [['x_csqa']]
    lans_zu = [langs, name_geo_langs, copa_langs]
    total_sum = 0
    for x, y in zip(task_zu, lans_zu):
        tasks = x
        use_langs = y
        have_error_model = set()

        for task in tasks:
            statistic_data = []
            for model in models:
                single_data = {"model": model, "ar": 0, "de": 0, "en": 0, "fr": 0, "hi": 0, "it": 0, "ja": 0, "nl": 0,
                               "pl": 0, "pt": 0, "ru": 0, "sw": 0, "ur": 0, "vi": 0, "zh": 0}
                for lang in use_langs:
                    if model == "chatgpt":
                        res_path = f"./{task}/{lang}/{model}_res.jsonl"
                    else:
                        res_path = f"{basepath}/gen_x_csqa_{lang}_datas-zhu2.jsonl"
                    ratio = eval_res(res_path, have_error_model, model, basepath, lang)
                    print(f"Task : {task} Lang:{lang} Model:{model} F1: {ratio}")
                    # print(f"lang : {lang.lower().replace('_trans','')}")
                    #single_data[lang.lower().replace("_trans", '')] = ratio * 100
                    # 累加每个比例的值
                    total_sum += ratio
                    single_data[lang.replace("_trans", '')] = ratio * 100
                # 计算平均值
                avg_value = total_sum * 100 / len(use_langs) 
                # 将平均值存入字典
                single_data['Average'] = avg_value
                statistic_data.append(single_data)
            csv_file_path = f'{basepath}/{task}.csv'
            csv_T_file_path = f'{basepath}/{task}_T.csv'
            print(f"statistic_data: {statistic_data}")
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["model", "ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "sw",
                              "ur", "vi", "zh", "Average"]

                csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                csv_writer.writeheader()

                for row in statistic_data:
                    csv_writer.writerow(row)
        for task in tasks:
            csv_file_path = f'{basepath}/{task}.csv'
            csv_T_file_path = f'{basepath}/{task}_T.csv'

            df = pd.read_csv(csv_file_path)

            df_transposed = df.transpose()

            df_transposed.to_csv(csv_T_file_path)


def xcsqa_batch_gen(
        lang_, xcsqa_questions, batch_llm
):
    lang = lang_ if lang_ != 'En_gsm8k' else 'English'

    prompt_no_input = (
        "Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
    )
    # prompt_no_input = (
    #   "Below is an instruction that describes a task. "
    #     f"Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n{query}\n\n### Response: Let's think step by step."
    # )
    input_str_list = [prompt_no_input.format(query=q) for q in xcsqa_questions]
    # input_str_list = [q for q in gsm8k_questions]
    output_str_list = batch_llm(input_str_list)
    return input_str_list, output_str_list


def get_batch_llama(model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
    @torch.inference_mode()
    def batch_llama(input_strs):
        input_ids_w_attnmask = tokenizer(
            input_strs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(
            input_ids=input_ids_w_attnmask.input_ids,
            attention_mask=input_ids_w_attnmask.attention_mask,
            generation_config=GenerationConfig(
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,  # t=0.0 raise error if do_sample=True
            ),
        ).tolist()
        # pad_token_id=tokenizer.eos_token_id
        real_output_ids = [
            output_id[len(input_ids_w_attnmask.input_ids[i]):] for i, output_id in enumerate(output_ids)
        ]
        output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
        return output_strs

    return batch_llama


def get_model(model_path: str, is_bf16: bool = False):
    print(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
    print(tokenizer.pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('new pad ', tokenizer.pad_token)
    print(tokenizer.bos_token)
    print(tokenizer.unk_token)
    print(tokenizer.eos_token)
    print(tokenizer.truncation_side)
    print(tokenizer.padding_side)

    if is_bf16:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
        ).cuda()
    model.eval()
    print(model.dtype)

    return model, tokenizer


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0



if __name__ == "__main__":
    import fire

    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--streategy",
        type=str,
        help="which streategy to evaluate the model",
        required=True,
        choices=['Parallel', 'Cross']
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batchsize",
        required=True
    )
    parser.add_argument(
        "--lang_only",
        type=str,
        #nargs='+',
        help="specific language to test",
        default=None
    )
    args = parser.parse_args()

    fire.Fire(main(args=args))