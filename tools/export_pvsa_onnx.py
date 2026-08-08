#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出完整 PVSA 模型的固定形状 ONNX（含 PVSA 自定义节点）。

完整部署链路：

    完整 PVSA 模型 -> 固定形状 ONNX -> TensorRT 完整引擎 -> trtexec 测速

本脚本把 PyTorch 推理路径中的两个自定义 CUDA 算子
（topp_flash_kernel.topp_route_cuda / topp_flash_attention）通过 ONNX
symbolic 映射导出为与 deploy/tensorrt 插件同名的自定义节点：

    PVSA_TopP_Route   （route：topk 路由）
    PVSA_TopP_Flash   （flash：稀疏窗口注意力）

TensorRT 的 onnx parser 加载 libpvsa_tensorrt_plugins.so 后，会按节点
op_type 匹配插件 creator，从而把完整网络转成引擎。

要求：
- 运行机器有 GPU，且 PVSA CUDA 扩展可编译/已缓存（走 cuda 推理路径）。
- 输入尺寸固定，特征图 H、W 能被 7 整除（auto_pad 会自动补齐）。
- 首次建议导出 FP32 ONNX，验证数值后再考虑 FP16。
"""

import argparse
import os

import torch
import torch.nn as nn

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules



# ============================================================
# 1) ONNX symbolic 映射

# 原始 CUDA 算子函数（全局引用，避免闭包在 TorchScript tracing 中失效）
_ORIGINAL_ROUTE = None
_ORIGINAL_FLASH = None
#    top_p_bra 推理时调用 topp_route_cuda / topp_flash_attention，
#    这里把它们替换成带 symbolic 的导出版本，仅导出时启用。
# ============================================================

def _patch_symbolic_export():
    global _ORIGINAL_ROUTE, _ORIGINAL_FLASH
    import mmseg.models.utils.top_p_bra as _tpb
    _ORIGINAL_ROUTE = _tpb.topp_route_cuda
    _ORIGINAL_FLASH = _tpb.topp_flash_attention
    class PVSARouteExport(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, topk, p, temperature, energy, scale,
                    full_route):
            r_weight, r_idx, keep_len = _ORIGINAL_ROUTE(
                query=query,
                key=key,
                topk=int(topk),
                p=float(p),
                temperature=float(temperature),
                energy=float(energy),
                scale=float(scale),
                full_route=bool(full_route),
                debug=False)
            return r_weight, r_idx, keep_len

        @staticmethod
        def symbolic(g, query, key, topk, p, temperature, energy, scale,
                     full_route):
            return g.op(
                'PVSA_TopP_Route',
                query,
                key,
                topk_i=int(topk),
                p_f=float(p),
                temperature_f=float(temperature),
                energy_f=float(energy),
                scale_f=float(scale),
                full_route_i=int(full_route),
                outputs=3)

    class PVSAFlashExport(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q_pix, kv_pix, r_weight, r_idx, keep_len, num_heads,
                    qk_dim, dim, scale, n_win, height, width,
                    use_route_weight):
            out = _ORIGINAL_FLASH(
                q_pix=q_pix,
                kv_pix=kv_pix,
                r_weight=r_weight,
                r_idx=r_idx,
                r_mask=None,
                num_heads=int(num_heads),
                qk_dim=int(qk_dim),
                dim=int(dim),
                scale=float(scale),
                n_win=int(n_win),
                H=int(height),
                W=int(width),
                backend='cuda',
                debug=False,
                keep_len=keep_len,
                use_route_weight=bool(use_route_weight))
            return out

        @staticmethod
        def symbolic(g, q_pix, kv_pix, r_weight, r_idx, keep_len, num_heads,
                     qk_dim, dim, scale, n_win, height, width,
                     use_route_weight):
            return g.op(
                'PVSA_TopP_Flash',
                q_pix,
                kv_pix,
                r_weight,
                r_idx,
                keep_len,
                num_heads_i=int(num_heads),
                qk_dim_i=int(qk_dim),
                dim_i=int(dim),
                n_win_i=int(n_win),
                height_i=int(height),
                width_i=int(width),
                scale_f=float(scale),
                use_route_weight_i=int(use_route_weight))

    def export_route(query, key, topk, p, temperature, energy, scale,
                     full_route=False, debug=False):
        return PVSARouteExport.apply(query, key, int(topk), float(p),
                                     float(temperature), float(energy),
                                     float(scale), bool(full_route))

    def export_flash(q_pix, kv_pix, r_weight, r_idx, r_mask=None,
                     num_heads=8, qk_dim=None, dim=None, scale=1.0, n_win=7,
                     H=None, W=None, block_windows=64, backend=None,
                     debug=False, keep_len=None, use_route_weight=True):
        if keep_len is None:
            if r_mask is None:
                raise ValueError('either r_mask or keep_len must be provided.')
            keep_len = r_mask.sum(dim=-1).contiguous().long()
        return PVSAFlashExport.apply(
            q_pix, kv_pix, r_weight, r_idx, keep_len,
            int(num_heads), int(qk_dim), int(dim), float(scale),
            int(n_win), int(H), int(W), bool(use_route_weight))

    _tpb.topp_route_cuda = export_route
    _tpb.topp_flash_attention = export_flash


# ============================================================
# 2) 导出入口
# ============================================================

class MMSegONNXWrapper(nn.Module):
    """mmseg 部署专用包装：_forward 只跑网络前向，不依赖数据样本。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # _forward 是 mmseg EncoderDecoder 的纯前向（返回 logits）；
        # 老版本 mmseg 的 forward_dummy 效果相同，这里统一用 _forward。
        return self.model._forward(x)


def _setup_cuda_env():

    # torch.onnx.export 与结构校验都需要 onnx 包
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise SystemExit('缺少 onnx 包：请先执行 pip install onnx，再重新运行。')
    """保证 torch.utils.cpp_extension 使用有效的 CUDA_HOME / nvcc。

    torch 的 cpp_extension 在首次 import 时会缓存 CUDA_HOME（模块变量），
    仅改 os.environ 不生效，必须同时覆盖 torch.utils.cpp_extension.CUDA_HOME。
    否则 JIT 编译扩展时仍会用失效路径（如 /usr/local/cuda-12.0/bin/nvcc）。
    """
    import glob
    import shutil

    nvcc_candidates = []
    found = shutil.which("nvcc")
    if found and os.path.exists(found):
        nvcc_candidates.append(os.path.realpath(found))
    for cand in ("/usr/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
        if cand not in nvcc_candidates and os.path.exists(cand):
            nvcc_candidates.append(os.path.realpath(cand))

    cuda_home = None
    for nvcc in nvcc_candidates:
        home = os.path.dirname(os.path.dirname(nvcc))
        if os.path.exists(os.path.join(home, "include", "cuda_runtime.h")):
            cuda_home = home
            break

    if cuda_home is None:
        for home in ["/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda-*"),
                                                 reverse=True):
            if (os.path.exists(os.path.join(home, "bin", "nvcc")) and
                    os.path.exists(os.path.join(home, "include",
                                               "cuda_runtime.h"))):
                cuda_home = home
                break

    if cuda_home is None:
        print("警告: 未找到有效 CUDA_HOME（缺少 include/cuda_runtime.h），请手动设置 CUDA_HOME 后重试。")
        return

    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CUDACXX"] = os.path.join(cuda_home, "bin", "nvcc")
    # 关键：覆盖 torch 已缓存的 CUDA_HOME 模块变量
    try:
        import torch.utils.cpp_extension as _cpp_ext
        _cpp_ext.CUDA_HOME = cuda_home
    except Exception:
        pass
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    print('CUDA_HOME=' + cuda_home)

def main():
    parser = argparse.ArgumentParser(
        description='导出完整 PVSA 固定形状 ONNX（含自定义节点）')
    parser.add_argument('--config', required=True,
                        help='模型配置文件路径，如 configs-h/biformer/'
                             'biformer_mm-20k_chase_db1-512x512.py')
    parser.add_argument('--checkpoint', nargs='?', default='0',
                        help='权重文件路径；传 0 或省略时随机初始化参数'
                             '（不加载权重）')
    parser.add_argument('--onnx', required=True, help='输出 ONNX 文件路径')
    parser.add_argument('--input-size', nargs=4, type=int,
                        default=[1, 3, 512, 512],
                        help='固定输入形状 B C H W（不要用动态轴）')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset，TensorRT 8.6 建议 11')
    args = parser.parse_args()

    input_shape = tuple(args.input_size)

    # 解析有效的 CUDA_HOME / nvcc，避免 JIT 编译扩展时找不到 nvcc
    _setup_cuda_env()

    # torch.onnx.export 与结构校验都需要 onnx 包
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise SystemExit('缺少 onnx 包：请先执行 pip install onnx，再重新运行。')

    # 注册 mmseg 所有模块（SegDataPreProcessor 等）
    register_all_modules()

    # 启用 symbolic 导出，必须在构建模型之前 patch
    _patch_symbolic_export()

    # 构建模型
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    # 强制走 CUDA 自定义算子路径，ONNX 中才会出现 PVSA 自定义节点
    cfg.merge_from_dict({'model.backbone.topp_flash_backend': 'cuda'})

    model = MODELS.build(cfg.model)
    if args.checkpoint not in (None, '0'):
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        print(f'已加载权重: {args.checkpoint}')
    else:
        print('未指定 checkpoint，使用随机初始化参数。')
    model.eval()
    # 关键：ONNX tracing 时若参数 requires_grad=True，qkv 输出也会
    # requires_grad=True，导致 can_run_topp_route_cuda 拒绝走 CUDA 路由，
    # 从而导出结果缺少 PVSA_TopP_Route 节点。
    model.requires_grad_(False)

    if not torch.cuda.is_available():
        raise RuntimeError('导出 PVSA 自定义节点需要 CUDA 与已编译的 PVSA '
                           'CUDA 扩展（cuda 推理后端）。')
    model = model.cuda()

    wrapper = MMSegONNXWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(input_shape, device='cuda')

    # 预热一次 forward：触发 optimize_for_inference 完成 Conv+BN 融合，
    # 避免 tracer 前后 state_dict 不一致（state_dict changed）报错。
    with torch.no_grad():
        _ = wrapper(dummy_input)

    os.makedirs(os.path.dirname(args.onnx) or '.', exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            args.onnx,
            input_names=['input'],
            output_names=['logits'],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    # 结构校验 + 确认自定义节点存在
    import onnx
    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)

    n_route = sum(1 for node in onnx_model.graph.node
                  if node.op_type == 'PVSA_TopP_Route')
    n_flash = sum(1 for node in onnx_model.graph.node
                  if node.op_type == 'PVSA_TopP_Flash')
    print(f'ONNX 结构校验通过: {args.onnx}')
    print(f'输入形状: {input_shape}')
    print(f'PVSA_TopP_Route 节点数: {n_route}')
    print(f'PVSA_TopP_Flash 节点数: {n_flash}')
    if n_route == 0 or n_flash == 0:
        raise RuntimeError('导出结果缺少 PVSA 自定义节点，请确认 '
                           'topp_flash_backend=cuda 且 CUDA 扩展可用。')


if __name__ == '__main__':
    main()
