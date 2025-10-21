import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1' #表示dataset加载数据时候，优先在cache里加载，而不是优先网络
from functools import partial
from sympy import false
import torch,datasets
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_py.modeling_mamba import *
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate,parse_eval_args,_int_or_none_list_arg_type
import argparse
from datetime import datetime
import random
from copy import deepcopy
from quantize.utils import set_seed,Logger
from mamba_py.normalized_modules import MatMul,MulAdd
from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
from quantize.hm_model_utils import fuse_layer_norms, fuse_layer_norms_2, fuse_ln_linear, RotateModule, RQuantLinear
from quantize.hadmard import random_hadamard_matrix
from quantize.plot_utils.utils import plot_line_fig,plot_quantile_fig,plot_box_data_perchannel_fig, plot_bar_fig, plot_bar3d_fig,concat_images,find_images
set_seed(10)

# 将 sys.stdout 重定向到 Logger 类实例
folder = 'logs'
sys.stdout = Logger(folder=folder)

def resize_tensor(tensor, size=(500, 500)):
    """
    将二维 tensor 重新采样到指定大小。

    Args:
        tensor: 输入的二维 tensor。
        size: 输出的大小，默认为 (500, 500)。

    Returns:
        重新采样后的 tensor。
    """
    h, w = tensor.shape
    target_h, target_w = size

    # 如果某个维度小于目标大小，不进行降采样
    new_h = min(h, target_h)
    new_w = min(w, target_w)
    
    # 调整到新尺寸
    resized_tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='nearest')
    
    # 将 tensor 的维度恢复
    return resized_tensor.squeeze(0).squeeze(0)

def analyse_hook(module, input, output): 
    module_name = module_to_name.get(module, "Unnamed module")
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    # 分析权重
    if isinstance(module, (QuantConv2d,QuantLinear)):
        weight = module.weight.data
        print(module_name,"_weight  shape: ",weight.shape)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg", axis=1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg", axis=0)
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg"):
        #     plot_box_data_perchannel_fig(weight, f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg", axis=-1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar_data.jpg"):
            plot_bar_fig(weight, f"{save_dir}/{module_name}_weight_bar_data.jpg")
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_bar3d_data.jpg"):
        #     plot_bar3d_fig(weight, f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
            # plot_bar3d_fig(resize_tensor(weight), f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
    
    # 分析权重
    if isinstance(module, QuantConv1d):
        weight = module.weight.data[:,0]
        print(module_name,"_weight  shape: ",weight.shape)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg", axis=1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg", axis=0)
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg"):
        #     plot_box_data_perchannel_fig(weight, f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg", axis=-1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar_data.jpg"):
            plot_bar_fig(weight, f"{save_dir}/{module_name}_weight_bar_data.jpg")
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_bar3d_data.jpg"):
        #     plot_bar3d_fig(weight, f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
            # plot_bar3d_fig(resize_tensor(weight), f"{save_dir}/{module_name}_weight_bar3d_data.jpg")

    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            print(module_name,f"_input_{i}  shape: ",temp_input.shape)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_token_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_token_quantile.jpg", axis=1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg", axis=2)
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg"):
            #     plot_box_data_perchannel_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar_data.jpg"):
                plot_bar_fig(temp_input, f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg"):
            #     plot_bar3d_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")
                # plot_bar3d_fig(resize_tensor(torch.amax(temp_input,dim=0)), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")

def analyse_hook_2(module, input, output):  # matmul画图
    module_name = module_to_name.get(module, "Unnamed module")
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            print(module_name,f"_input_{i}  shape: ",temp_input.shape)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_token_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_token_quantile.jpg", axis=1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg", axis=2)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg"):
                plot_box_data_perchannel_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar_data.jpg"):
                plot_bar_fig(temp_input, f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg"):
            #     plot_bar3d_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")
                # plot_bar3d_fig(resize_tensor(torch.amax(temp_input,dim=0)), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")

def analyse_hook_3(module, input, output): 
    module_name = "lm_head"
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    # 分析权重
    if isinstance(module, (QuantConv2d,QuantLinear,nn.Linear)):
        weight = module.weight.data
        print(module_name,"_weight  shape: ",weight.shape)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg", axis=1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg", axis=0)
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg"):
        #     plot_box_data_perchannel_fig(weight, f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg", axis=-1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar_data.jpg"):
            plot_bar_fig(weight, f"{save_dir}/{module_name}_weight_bar_data.jpg")
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_bar3d_data.jpg"):
        #     plot_bar3d_fig(weight, f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
            # plot_bar3d_fig(resize_tensor(weight), f"{save_dir}/{module_name}_weight_bar3d_data.jpg")

    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            print(module_name,f"_input_{i}  shape: ",temp_input.shape)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_token_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_token_quantile.jpg", axis=1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg", axis=2)
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg"):
            #     plot_box_data_perchannel_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar_data.jpg"):
                plot_bar_fig(temp_input, f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg"):
            #     plot_bar3d_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")
                # plot_bar3d_fig(resize_tensor(torch.amax(temp_input,dim=0)), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")


def register_hooks(model,name="fig"):
    global module_to_name 
    global quant_name
    global input_name
    input_name = name
    quant_name = "fp_data"
    module_to_name = {module: name for name, module in model.named_modules()}
    handles = []
    for i,layer in enumerate(model):

        # 新增后处理hook
        handles.append(layer.mixer.conv1d.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.in_proj_states.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.in_proj_gates.register_forward_hook(analyse_hook))
        # handles.append(layer.mixer.in_proj.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.x_proj_b.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.x_proj_c.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.x_proj_dt.register_forward_hook(analyse_hook))
        # handles.append(layer.mixer.x_proj.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.dt_proj.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.out_proj.register_forward_hook(analyse_hook))
        # handles.append(layer.mixer.conv_matmul.register_forward_hook(analyse_hook_2))
        handles.append(layer.mixer.ssm_matmul.register_forward_hook(analyse_hook_2))
    return handles

def register_hooks_2(model,name="fig"):
    global module_to_name 
    global quant_name
    global input_name
    input_name = name
    quant_name = "fp_data"
    module_to_name = {module: name for name, module in model.named_modules()}
    handles = []
    handles.append(model.lm_head.register_forward_hook(analyse_hook_3))
    return handles


def count_suffixes(directory):
    """
    统计指定文件夹中图片文件名中'mixer.'后面部分的后缀名称及其数量。
    
    参数:
    - directory: 包含图片的文件夹路径。
    
    返回:
    - 一个字典，键为后缀名称，值为该后缀名称出现的次数。
    """
    suffix_counts = {}
    for filename in os.listdir(directory):
        # 分割文件名以找到'mixer.'后面的部分
        parts = filename.split('mixer.')
        if len(parts) > 1:
            suffix = parts[-1]  # 获取'mixer.'后面的部分
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return suffix_counts

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`")
    parser.add_argument("--tasks",
        "-t",
        default=None,
        type=str,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        type=str,
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument("--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument("--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="If True, applies the chat template to the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`."
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument(
        "--wandb_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument("--use_smoothquant", action="store_true")
    parser.add_argument("--use_gptq", action="store_true")
    parser.add_argument("--use_hadmard", action="store_true")
    parser.add_argument('--use_klt', action="store_true")
    parser.add_argument('--use_weight_klt', action="store_true")
    parser.add_argument('--use_S_head', action="store_true")
    parser.add_argument('--use_S1', action="store_true")
    parser.add_argument('--use_S2', action="store_true")
    parser.add_argument('--use_S3', action="store_true")
    parser.add_argument('--use_S4', action="store_true")
    parser.add_argument('--use_S5', action="store_true")
    parser.add_argument('--use_S7', action="store_true")
    parser.add_argument('--use_hadmard_R1', action="store_true")
    parser.add_argument('--use_hadmard_R2', action="store_true")
    parser.add_argument('--use_hadmard_R3', action="store_true")
    parser.add_argument('--use_hadmard_R4', action="store_true")
    parser.add_argument('--use_hadmard_R5', action="store_true")
    parser.add_argument('--use_hadmard_R6', action="store_true")
    parser.add_argument('--use_pertoken', action="store_true")
    parser.add_argument("--static_quant", action="store_true")
    parser.add_argument('--quant_weight', action="store_true")
    parser.add_argument('--quant_act', action="store_true")
    parser.add_argument('--w_bit', type=int,default=8)
    parser.add_argument('--a_bit', type=int,default=8)
    parser.add_argument('--w_perchannel', action="store_true")
    parser.add_argument('--observe', default="minmax",type=str)
    parser.add_argument('--fake_online_hadamard', action="store_true")
    parser.add_argument("--use_perkernel", action="store_true")
    parser.add_argument("--use_reduce_mean", action="store_true")
    parser.add_argument('--analyse_and_plot', action="store_true")
    return parser

@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="/data01/home/xuzk/datasets/lm_mamba_weight/mamba-130m-hf", max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.float16,add_bos_token=False):
        super().__init__(pretrained)
        LM.__init__(self)
        from mamba_py.modeling_mamba import MambaForCausalLM
        self._model = MambaForCausalLM.from_pretrained(pretrained)
        # self._model = AutoModelForCausalLM.from_pretrained(pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        self.add_bos_token = add_bos_token

        self._model.half()
        self._model.to(device)

        global args
        # from utils.utils import convert_vim_2_vim_torch      

        def replace_layers(model, target_class, replacement_class):
            for name, child in model.named_children():
                # if False\
                #     or "conv1d" in name \
                #     or "in_proj" in name \
                #     or "dt_proj" in name \
                #     or "x_proj" in name \
                #     or "out_proj" in name \
                #     or "matmul" in name \
                #     or "lm_head" in name:
                #     continue
                # if "in_proj" in name:
                #     continue
                if isinstance(child, target_class):
                    # Replace the layer with the new quantized version
                    if target_class == MatMul:
                        if args.use_pertoken:
                            setattr(model, name, replacement_class(
                                x1_quant_params={"dynamic_method":"per_channel","per_channel_axes":[1],"n_bits":8,"percent":0.999},
                                x2_quant_params={"dynamic_method":"per_channel","per_channel_axes":[1],"n_bits":8,"percent":0.999},
                                observe="minmax"))#args.observe
                        else:
                            setattr(model, name, replacement_class(x1_quant_params=args.a_cfg,x2_quant_params=args.a_cfg,observe=args.observe))
                    elif "conv1d" in name:
                        setattr(model, name, replacement_class(child,weight_quant_params=args.conv1d_w_cfg,act_quant_params=args.a_cfg,observe=args.observe))
                    else:
                        setattr(model, name, replacement_class(child,weight_quant_params=args.w_cfg,act_quant_params=args.a_cfg,observe=args.observe))
                else:
                    # Recursively call this function on the child module
                    replace_layers(child, target_class, replacement_class)

        replace_layers(self._model, MatMul, QuantMatMul)
        replace_layers(self._model, nn.Conv1d, QuantConv1d)
        replace_layers(self._model, nn.Linear, QuantLinear)
        
        from quantize.utils import set_quant_state,set_static_quant,set_observing
        set_quant_state(self._model,weight_quant=args.quant_weight,act_quant=args.quant_act)

        traindata = datasets.load_dataset("hellaswag",split='test')
        trainenc = self.tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
        dataloader = []
        num_samples = 128
        for _ in range(num_samples):
            inps = []
            tars = []
            for  i in range(8):
                i = random.randint(0, trainenc.input_ids.shape[1] - 2048 - 1)
                j = i + 2048
                inp = trainenc.input_ids[:, i:j]
                tar = inp.clone()
                tar[:, :-1] = -100
                inps.append(inp)
                tars.append(tar)
            dataloader.append((torch.cat(inps,dim=0), torch.cat(tars,dim=0)))

        if args.use_hadmard: 
            from mamba_py.modeling_mamba import MambaBlock_optimize
            length_layers = len(self._model.backbone.layers)
            for i in range(length_layers):#将模型的x_proj和in_proj拆分，方便做旋转
                self._model.backbone.layers[i] = MambaBlock_optimize(self._model.backbone.layers[i]) 
            
            if args.use_klt:
                if args.use_weight_klt:
                    if not os.path.exists(pretrained+"_weight_klt.pt"):
                        from quantize.get_klt_matrix import get_llm_weight_klt
                        klt_matrix = get_llm_weight_klt(self._model,dataloader)
                        torch.save(klt_matrix,pretrained+"_weight_klt.pt")
                    else:
                        klt_matrix = torch.load(pretrained+"_weight_klt.pt",map_location=device)
                else:
                    if not os.path.exists(pretrained+"_act_klt.pt"):
                        from quantize.get_klt_matrix import get_llm_act_klt
                        klt_matrix = get_llm_act_klt(self._model,dataloader)
                        torch.save(klt_matrix,pretrained+"_act_klt.pt")
                    else:
                        klt_matrix = torch.load(pretrained+"_act_klt.pt",map_location=device)
            
            if not os.path.exists(pretrained+"_smooth.pt") :
                from quantize.smoothquant_generate_act_scale_shift import mamba_llm_generate_act_scale
                act_scales = mamba_llm_generate_act_scale(self._model)
                torch.save(act_scales,pretrained+"_smooth.pt")
            else:
                act_scales = torch.load(pretrained+"_smooth.pt")
                
            if not os.path.exists(pretrained+"_smooth_shifts.pt"):
                from quantize.smoothquant_generate_act_scale_shift import mamba_llm_generate_act_shift
                act_shifts = mamba_llm_generate_act_shift(self._model)
                torch.save(act_shifts,pretrained+"_smooth_shifts.pt")
            else:
                act_shifts = torch.load(pretrained+"_smooth_shifts.pt")
            
            # if not os.path.exists(pretrained+"_channel_mean.pt"):
            #     from quantize.smoothquant_generate_act_scale_shift import mamba_llm_generate_act_shift
            #     channel_means = mamba_llm_generate_act_shift(self._model)
            #     torch.save(act_shifts,pretrained+"_channel_mean.pt")
            # else:
            #     channel_means = torch.load(pretrained+"_channel_mean.pt")
            
                   
            if not os.path.exists(pretrained+"_smooth_matmul_S3.pt"):
                from quantize.smoothquant_get_matmul_scale import mamba_llm_generate_matmul_scale_S3
                S3_scales = mamba_llm_generate_matmul_scale_S3(self._model)
                torch.save(S3_scales,pretrained+"_smooth_matmul_S3.pt")
            else:
                S3_scales = torch.load(pretrained+"_smooth_matmul_S3.pt")  
            
            if not os.path.exists(pretrained+"_smooth_matmul_S4.pt"):
                from quantize.smoothquant_get_matmul_scale import mamba_llm_generate_matmul_scale_S4
                S4_scales = mamba_llm_generate_matmul_scale_S4(self._model)
                torch.save(S4_scales,pretrained+"_smooth_matmul_S4.pt")
            else: 
                S4_scales = torch.load(pretrained+"_smooth_matmul_S4.pt")
                
            device, dtype = self._model.backbone.layers[0].mixer.out_proj.weight.device, self._model.backbone.layers[0].mixer.out_proj.weight.dtype
            
            if args.use_reduce_mean:
                for i,layer in enumerate(self._model.backbone.layers): 
                    shift = act_shifts[f"backbone.layers.{i}.mixer.dt_proj"].to(device=device, dtype=dtype)
                    layer.mixer.x_proj_dt.bias = nn.Parameter(-shift).to(device).to(dtype)
                    layer.mixer.dt_proj.bias.data = layer.mixer.dt_proj.bias.data + \
                        (shift@layer.mixer.dt_proj.weight.data.T).data.reshape(-1)
              
            if args.use_S_head:
                self._model.S_R1 = act_scales["lm_head"].to(device)
            
            if args.use_S2:
                class Swiglu(nn.Module):
                    def __init__(self, s):
                        super().__init__()
                        self.s = s
                        self.sigmod = nn.Sigmoid()
                    def forward(self, x):
                        return x*self.sigmod(x*self.s)

                for i,layer in enumerate(self._model.backbone.layers):    
                    act = act_scales[f"backbone.layers.{i}.mixer.out_proj"].to(device=device, dtype=dtype)
                    weight_scales = layer.mixer.out_proj.weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)
                    alpha = 0.5
                    scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device).to(dtype))

                    layer.mixer.in_proj_gates.weight.data = (1/scales.reshape(-1,1))*layer.mixer.in_proj_gates.weight.data
                    layer.mixer.out_proj.weight.data = scales*layer.mixer.out_proj.weight.data
                    layer.mixer.act_gate = Swiglu(scales)
            
            if args.use_S3:
                for name,module in self._model.backbone.named_modules():
                    if 'ssm_matmul' in name  and 'quantizer' not in name:    
                        module.register_parameter("S3",torch.nn.Parameter(S3_scales["backbone."+name].clamp(min=1e-2)))
            
            if args.use_S4:
                for i,layer in enumerate(self._model.backbone.layers):
                    name = "backbone." + f"layers.{i}.mixer.ssm_matmul"
                    s4 = S4_scales[name].clamp(min=1e-2,max=100).to(device).to(dtype)
                    layer.mixer.x_proj_c.weight.data = (s4.reshape(-1,1))*layer.mixer.x_proj_c.weight.data
                    layer.mixer.x_proj_b.weight.data = (1/s4.reshape(-1,1))*layer.mixer.x_proj_b.weight.data
                    layer.mixer.mul_delta_A = MulAdd(torch.log(1/s4))
                    
                # for name,module in self._model.backbone.named_modules():
                #     if 'matmul' in name and 'quantizer' not in name:    
                #         module.register_parameter("S4",torch.nn.Parameter(S4_scales["backbone."+name].clamp(min=1e-2)))
   
   
            
            if args.use_S5:
                for i,layer in enumerate(self._model.backbone.layers): 
                    act = act_scales[f"backbone.layers.{i}.mixer.dt_proj"].to(device=device, dtype=dtype)
                    weight_scales = layer.mixer.dt_proj.weight.abs().max(dim=0, keepdim=True)[0].reshape(-1).clamp(min=1e-5)
                    alpha = 0.5
                    scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device).to(dtype))
                    if layer.mixer.x_proj_dt.bias is not None:
                        layer.mixer.x_proj_dt.bias.data = layer.mixer.x_proj_dt.bias.data*(1/scales)
                    layer.mixer.x_proj_dt.weight.data = (1/scales.reshape(-1,1))*layer.mixer.x_proj_dt.weight.data
                    layer.mixer.dt_proj.weight.data = scales*layer.mixer.dt_proj.weight.data

            if args.use_S7:
                for i,layer in enumerate(self._model.backbone.layers):    
                    act = act_scales[f"backbone.layers.{i}.mixer.conv1d"].to(device=device, dtype=dtype)
                    weight_scales = (layer.mixer.conv1d.weight.abs().max(dim=-1)[0]).reshape(-1).clamp(min=1e-5)
                    alpha = 0.5
                    scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device).to(dtype))
                    layer.mixer.in_proj_states.weight.data = (1/scales.reshape(-1,1))*layer.mixer.in_proj_states.weight.data
                    layer.mixer.conv1d.weight.data = scales.reshape(-1,1,1)*layer.mixer.conv1d.weight.data
            
            R1 = random_hadamard_matrix(self._model.backbone.layers[0].mixer.hidden_size,device).to(device=device, dtype=dtype)
            R2 = random_hadamard_matrix(self._model.backbone.layers[0].mixer.out_proj.in_features,device).to(device=device, dtype=dtype)
            R3 = random_hadamard_matrix(self._model.backbone.layers[0].mixer.out_proj.in_features,device).to(device=device, dtype=dtype)
            R4 = random_hadamard_matrix(self._model.backbone.layers[0].mixer.ssm_state_size,device).to(device=device, dtype=dtype)
            R5 = random_hadamard_matrix(self._model.backbone.layers[0].mixer.time_step_rank,device).to(device=device, dtype=dtype)
            R6 = random_hadamard_matrix(self._model.backbone.layers[0].mixer.x_proj_b.in_features,device).to(device=device, dtype=dtype)
            if args.fake_online_hadamard:
                R1,R2,R3,R4,R5,R6 = R1.T,R2.T,R3.T,R4.T,R5.T,R6.T   

            if args.use_hadmard_R1:
                K = torch.eye(R1.shape[0]).to(R1) \
                          if not args.use_klt else klt_matrix[f"backbone.layers.0.mixer.in_proj_states"].to(device=device, dtype=dtype)
                R1 = K@R1
                
                if hasattr(self._model.backbone.layers[0].mixer,"in_proj"):
                    fuse_layer_norms(self._model.backbone)
                else:
                    fuse_layer_norms_2(self._model.backbone)

                for i,layer in enumerate(self._model.backbone.layers):
                    if hasattr(layer.mixer,"in_proj"):
                        layer.mixer.in_proj.weight.data = layer.mixer.in_proj.weight.data@R1
                    else:
                        layer.mixer.in_proj_states.weight.data = layer.mixer.in_proj_states.weight.data@R1
                        layer.mixer.in_proj_gates.weight.data = layer.mixer.in_proj_gates.weight.data@R1
                    layer.mixer.out_proj.weight.data = R1.T@layer.mixer.out_proj.weight.data
                
                self._model.backbone.R1 =R1             
            
            if args.use_hadmard_R2:
                for layer in self._model.backbone.layers:
                    K = torch.eye(R2.shape[0]).to(R2) \
                          if not args.use_klt else klt_matrix[f"backbone.layers.{i}.mixer.out_proj"].to(R2)
                    R2 = K@R2
                    
                    layer.mixer.R2 = R2
                    layer.mixer.out_proj.weight.data = (layer.mixer.out_proj.weight.data.to(R2)@R2).to(dtype)
                   
            if args.use_hadmard_R3:
                for layer in self._model.backbone.layers:
                    layer.mixer.R3 = R3.data

            if args.use_hadmard_R4:
                length_layers = len(self._model.backbone.layers)
                for i in range(length_layers):
                    self._model.backbone.layers[i].mixer.R4 = R4
                    self._model.backbone.layers[i].mixer.x_proj_c.weight.data = \
                        R4.T.to(dtype)@self._model.backbone.layers[i].mixer.x_proj_c.weight.data

            if args.use_hadmard_R5:
                length_layers = len(self._model.backbone.layers)
                for i in range(length_layers):
                    K = torch.eye(R5.shape[0]).to(R5) \
                          if not args.use_klt else klt_matrix[f"backbone.layers.{i}.mixer.dt_proj"].to(R5)
                    R5 = (K@R5).to(dtype)

                    if self._model.backbone.layers[i].mixer.x_proj_dt.bias is not None:
                        self._model.backbone.layers[i].mixer.x_proj_dt.bias.data = (self._model.backbone.layers[i].mixer.x_proj_dt.bias.data.reshape(1,-1)@R5.T).reshape(-1)
                    self._model.backbone.layers[i].mixer.x_proj_dt.weight.data = R5@self._model.backbone.layers[i].mixer.x_proj_dt.weight.data
                    self._model.backbone.layers[i].mixer.dt_proj.weight.data = self._model.backbone.layers[i].mixer.dt_proj.weight.data@R5.T

            if args.use_hadmard_R6:        
                length_layers = len(self._model.backbone.layers)
                for i in range(length_layers):
                    self._model.backbone.layers[i].mixer.R6 = R6
                    self._model.backbone.layers[i].mixer.x_proj_dt.weight.data = self._model.backbone.layers[i].mixer.x_proj_dt.weight.data@R6
                    self._model.backbone.layers[i].mixer.x_proj_c.weight.data = self._model.backbone.layers[i].mixer.x_proj_c.weight.data@R6
                    self._model.backbone.layers[i].mixer.x_proj_b.weight.data = self._model.backbone.layers[i].mixer.x_proj_b.weight.data@R6

            if args.use_S1:
                for i,layer in enumerate(self._model.backbone.layers): 
                    # act = act_scales[f"backbone.layers.{i}.mixer.in_proj_states"].to(device=device, dtype=dtype)
                    # weight_states = layer.mixer.in_proj_states.weight.data
                    # weight_gates  = layer.mixer.in_proj_gates.weight.data
                    # weight = torch.cat([weight_states,weight_gates],dim=0)
                    # weight_scales = weight.abs().max(dim=0, keepdim=True)[0].reshape(-1).clamp(min=1e-5)
                    # alpha = 0.5
                    # scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype))

                    weight_states = layer.mixer.in_proj_states.weight.data
                    weight_gates  = layer.mixer.in_proj_gates.weight.data
                    weight = torch.cat([weight_states,weight_gates],dim=0)
                    weight_scales = weight.abs().max(dim=0, keepdim=True)[0].reshape(-1).clamp(min=1e-2)
                    max_val = torch.max(weight_scales)
                    scales = ((max_val / (2*weight_scales)).clamp(min=1e-2).to(device).to(dtype))

                    layer.norm.weight.data = (1/scales.reshape(-1))*layer.norm.weight.data
                    
                    layer.mixer.in_proj_states.weight.data = scales*layer.mixer.in_proj_states.weight.data
                    layer.mixer.in_proj_gates.weight.data = scales*layer.mixer.in_proj_gates.weight.data

        if args.use_gptq:
            path = pretrained+"_gptq_{}bit.pt".format(args.w_cfg['n_bits'])
            if os.path.exists(path):
                self._model.load_state_dict(torch.load(path),strict=False)
            else:
                import argparse
                args_ = argparse.Namespace()
                args_.seqlen = 2048
                args_.nsamples = 128
                args_.w_bits=args.w_cfg.get('n_bits')
                traindata = datasets.load_dataset("hellaswag",split='test')
                trainenc = self.tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
                trainloader = []
                for _ in range(args_.nsamples):
                    i = random.randint(0, trainenc.input_ids.shape[1] - args_.seqlen - 1)
                    j = i + args_.seqlen
                    inp = trainenc.input_ids[:, i:j]
                    tar = inp.clone()
                    tar[:, :-1] = -100
                    trainloader.append((inp, tar))

                from quantize.gptq import gptq_fwrd_llm
                gptq_fwrd_llm(self._model,trainloader,dev='cuda',args=args_)
                self._model.to(device)
                torch.save(self._model.state_dict(),path)

        if args.use_smoothquant:
            path = pretrained+"_smooth.pt"
            if not os.path.exists(path):
                from quantize.smoothquant_generate_act_scale_shift import mamba_llm_generate_act_scale_shift
                act_scales,act_shifts = mamba_llm_generate_act_scale_shift(self._model,trainloader)
                torch.save(act_scales,path)
                torch.save(act_shifts,pretrained+"_smooth_shifts.pt")
            act_scales = torch.load(path)
            device,dtype = next(self._model.parameters()).device,next(self._model.parameters()).dtype
            def smooth_ln_fcs(model,act_scales,alpha=0.5):
                for name, module in model.named_modules():
                    if isinstance(module,nn.Linear) and 'in_proj' in name:
                        weight_scale = module.weight.max(dim=0)[0].clamp(min=1e-2)
                        act_scale = act_scales[name].to(device)
                        scales = ((act_scale.pow(alpha)/weight_scale.pow(1-alpha)).clamp(min=1e-2)).to(device).to(dtype)
                        with torch.no_grad():
                            module.weight.div_(scales)

                        parts = name.split('.')
                        parent_name = '.'.join(parts[:-2])
                        rms_name = parent_name + '.norm'
                        if 'in_proj_states' in name and 'in_proj_gates' not in name:
                            for subname, submodule in model.named_modules():
                                if subname == rms_name:
                                    with torch.no_grad():
                                        submodule.weight.mul_(scales)
            smooth_ln_fcs(self._model,act_scales)
    
        if args.static_quant:#先较准
            set_static_quant(self.model,True)
            set_observing(self.model,True)
            seqlen = 258
            nsamples = 64
            traindata = datasets.load_dataset("hellaswag",split='test', download_mode="reuse_cache_if_exists")
            trainenc = self.tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
            inps = []
            for _ in range(nsamples):    
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                with torch.no_grad():
                    self._model(inp.to(device))
            set_observing(self.model,False)
    
        if args.analyse_and_plot:
            input_name = "fig_s1-s7_r1-r6_noklt_790m"
            seqlen = 258
            nsamples = 64
            traindata = datasets.load_dataset("hellaswag",split='test', download_mode="reuse_cache_if_exists")
            trainenc = self.tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
            inps = []
            for _ in range(nsamples):    
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                inps.append(inp)
            inp = torch.concat(inps, dim=0).to(self._device)
            handles = register_hooks(self._model.backbone.layers,input_name)
            # handles = register_hooks_2(self._model,input_name)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    self._model(inp)
            for handle in handles:handle.remove() 
    
    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()

parser = setup_parser()
args=parse_eval_args(parser)
if __name__ == "__main__":
    # 清除cache
    if os.path.exists('/data01/home/xuzk/.cache/huggingface/datasets/_rank0.db'):
        os.remove('/data01/home/xuzk/.cache/huggingface/datasets/_rank0.db')
    
    w_bit = args.w_bit
    a_bit = args.a_bit
    if args.w_perchannel:
        args.w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[0],"n_bits":args.w_bit}
    else:
        args.w_cfg = {"dynamic_method":"per_tensor","n_bits":args.w_bit}
    
    if args.use_perkernel:
        args.conv1d_w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[2],"n_bits":args.w_bit}
    else:
        args.conv1d_w_cfg = args.w_cfg
    args.a_cfg = {"dynamic_method":"per_tensor","n_bits":args.a_bit}

    args_dict = vars(args)
    for var_name, var_value in args_dict.items():
        print(f"{var_name}: {var_value}")

    for task in args.tasks.split(','):
        test_args = deepcopy(args)
        test_args.tasks = task
        cli_evaluate(test_args)
    # cli_evaluate(args)