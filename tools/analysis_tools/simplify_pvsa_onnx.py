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
    from onnx import helper
    model = onnx.load(args.onnx)
    cnt = Counter(n.op_type for n in model.graph.node)
    print('简化前节点数:', sum(cnt.values()))
    print('简化前 PVSA_TopP_Route:', cnt.get('PVSA_TopP_Route', 0),
          ' PVSA_TopP_Flash:', cnt.get('PVSA_TopP_Flash', 0))

    # onnxsim 的 C++ shape inference 不认识空 domain 的自定义节点，会报
    # "No Op registered"。简化前临时把 PVSA_TopP_* 移到自定义 domain
    # pvsa.custom（ONNX 规范允许未注册的自定义域，推断会跳过），简化完
    # 再恢复为空 domain，保持与 TensorRT 插件（空 namespace）匹配。
    custom_domain = 'pvsa.custom'
    for node in model.graph.node:
        if node.op_type in ('PVSA_TopP_Route', 'PVSA_TopP_Flash'):
            node.domain = custom_domain
    if custom_domain not in {o.domain for o in model.opset_import}:
        model.opset_import.append(helper.make_opsetid(custom_domain, 1))

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

    def _restore_domain(graph):
        for node in graph.node:
            if (node.op_type in ('PVSA_TopP_Route', 'PVSA_TopP_Flash')
                    and node.domain == custom_domain):
                node.domain = ''
            # 递归处理子图（前馈网络一般没有，防御性保留）
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _restore_domain(attr.g)

    _restore_domain(model_sim.graph)
    # repeated message 字段不支持切片赋值，先清空再逐个追加
    keep_opset = [o for o in model_sim.opset_import
                  if o.domain != custom_domain]
    del model_sim.opset_import[:]
    for o in keep_opset:
        model_sim.opset_import.append(o)

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
