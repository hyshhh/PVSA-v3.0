#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 onnxsim 简化导出的 PVSA ONNX 图。

背景：torch.onnx.export 导出的完整 PVSA 图里含有大量动态形状辅助算子
（Constant / Shape / Gather / Unsqueeze / Slice / Concat 等，通常数千个）。
这些算子在 PyTorch 里几乎零开销，但进入 TensorRT 引擎后会变成真实 GPU
层，且动态形状会阻止层融合，导致引擎明显慢于 PyTorch。
onnxsim 可以折叠这类常量链、消除冗余算子。

用法：
    pip install onnxsim

    python tools/analysis_tools/simplify_pvsa_onnx.py work_dirs/pvsa_full.onnx \
        --output work_dirs/pvsa_full_sim.onnx \
        --input-shape 1 3 512 512

简化成功后用简化后的 onnx 重新构建 TensorRT 引擎并测速：
    trtexec --onnx=work_dirs/pvsa_full_sim.onnx \
      --staticPlugins=.../libpvsa_tensorrt_plugins.so --saveEngine=...
"""

import argparse
import os
import time
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description='用 onnxsim 简化 PVSA ONNX 图')
    parser.add_argument('onnx', help='输入 ONNX 文件路径')
    parser.add_argument('--output', default=None,
                        help='输出 ONNX 路径（默认 <输入名>_sim.onnx）')
    parser.add_argument('--input-shape', nargs=4, type=int,
                        default=[1, 3, 512, 512],
                        help='固定输入形状 B C H W')
    parser.add_argument('--check-n', type=int, default=0,
                        help='onnxsim 数值验证次数；自定义节点无法在 '
                             'onnxruntime 上执行，保持 0')
    args = parser.parse_args()

    try:
        import onnxsim
    except ImportError:
        raise SystemExit('缺少 onnxsim 包：请先执行 pip install onnxsim，'
                         '再重新运行。')

    import onnx
    model = onnx.load(args.onnx)
    cnt = Counter(n.op_type for n in model.graph.node)
    print('简化前节点数:', sum(cnt.values()))
    print('简化前 PVSA_TopP_Route:', cnt.get('PVSA_TopP_Route', 0),
          ' PVSA_TopP_Flash:', cnt.get('PVSA_TopP_Flash', 0))

    if args.output is None:
        root, ext = os.path.splitext(args.onnx)
        args.output = root + '_sim' + (ext or '.onnx')

    t0 = time.time()
    model_sim, check_ok = onnxsim.simplify(
        model,
        overwrite_input_shapes={'input': args.input_shape},
        check_n=args.check_n,
    )
    print(f'简化耗时: {time.time() - t0:.1f}s，验证通过: {check_ok}')

    cnt_sim = Counter(n.op_type for n in model_sim.graph.node)
    print('简化后节点数:', sum(cnt_sim.values()))
    print('简化后 PVSA_TopP_Route:', cnt_sim.get('PVSA_TopP_Route', 0),
          ' PVSA_TopP_Flash:', cnt_sim.get('PVSA_TopP_Flash', 0))
    print('简化后算子构成（前 20）:')
    for op, num in cnt_sim.most_common(20):
        print(f'  {op}: {num}')

    onnx.save(model_sim, args.output)
    print('已保存:', args.output)


if __name__ == '__main__':
    main()
