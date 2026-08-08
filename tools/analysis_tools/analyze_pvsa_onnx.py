#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析导出的 PVSA ONNX 图：节点构成 / 自定义节点 / 输入输出形状。

用于排查"完整 TensorRT 引擎比 PyTorch 慢"的问题：
统计图中 Transpose、Pad、Gather 等可能拖慢引擎的算子数量，
并确认 PVSA_TopP_Route / PVSA_TopP_Flash 自定义节点存在且数量合理。

用法：
    python tools/analysis_tools/analyze_pvsa_onnx.py work_dirs/pvsa_full.onnx
"""

import argparse
from collections import Counter


def _dims(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField('dim_value'):
            dims.append(str(d.dim_value))
        else:
            dims.append(d.dim_param or '?')
    return 'x'.join(dims)


def main():
    parser = argparse.ArgumentParser(description='分析 PVSA ONNX 图')
    parser.add_argument('onnx', help='ONNX 文件路径')
    args = parser.parse_args()

    import onnx
    model = onnx.load(args.onnx)
    g = model.graph

    cnt = Counter(n.op_type for n in g.node)
    total = sum(cnt.values())
    print('总节点数:', total)
    print('算子构成（前 30）:')
    for op, num in cnt.most_common(30):
        print(f'  {op}: {num}')

    n_route = cnt.get('PVSA_TopP_Route', 0)
    n_flash = cnt.get('PVSA_TopP_Flash', 0)
    print('PVSA_TopP_Route 节点数:', n_route)
    print('PVSA_TopP_Flash 节点数:', n_flash)

    print('图输入:')
    for vi in g.input:
        print('  ', vi.name, _dims(vi))
    print('图输出:')
    for vo in g.output:
        print('  ', vo.name, _dims(vo))

    # 自定义节点输入输出形状（首尾各取一个）
    route_nodes = [n for n in g.node if n.op_type == 'PVSA_TopP_Route']
    flash_nodes = [n for n in g.node if n.op_type == 'PVSA_TopP_Flash']
    if route_nodes:
        n0 = route_nodes[0]
        print('首个 PVSA_TopP_Route 节点属性:')
        for a in n0.attribute:
            v = a.i if a.type == 2 else (a.f if a.type == 1 else a.s)
            print(f'  {a.name} = {v}')
        print('  输入:')
        for inp in n0.input:
            print('    ', inp)
    if flash_nodes:
        n0 = flash_nodes[0]
        print('首个 PVSA_TopP_Flash 节点属性:')
        for a in n0.attribute:
            v = a.i if a.type == 2 else (a.f if a.type == 1 else a.s)
            print(f'  {a.name} = {v}')
        print('  输入:')
        for inp in n0.input:
            print('    ', inp)


if __name__ == '__main__':
    main()
