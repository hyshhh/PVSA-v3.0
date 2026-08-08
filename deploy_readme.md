# PVSA-Net v3.0 部署命令

## 1. 设置环境

```bash
cd /media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-v3.0

export CUDACXX=/usr/bin/nvcc
export TRT_ROOT=$HOME/opt/TensorRT-8.6.1.6
export CUDNN_ROOT=$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib
export PYTHONPATH=$PWD:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export PATH=$TRT_ROOT/bin:/usr/bin:$PATH
export LD_LIBRARY_PATH=$CUDNN_ROOT:$TRT_ROOT/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

检查 CUDA：

```bash
nvcc --version
cmake --version
```

## 2. 检查 cuDNN 8

TensorRT 8.6.1 需要 `libcudnn.so.8`。当前 `openmmlab` 环境使用：

```bash
find "$CONDA_PREFIX" -type f -name 'libcudnn.so.8' -print
ls -lh "$CUDNN_ROOT/libcudnn.so.8"
ldd "$CUDNN_ROOT/libcudnn.so.8" | grep "not found"
```

如果路径不同，按 `find` 的结果修改 `CUDNN_ROOT`；找不到 `libcudnn.so.8` 时先安装 cuDNN 8。

## 3. 编译普通版插件

```bash
rm -rf build/tensorrt

cmake -S deploy/tensorrt \
  -B build/tensorrt \
  -DTENSORRT_INCLUDE_DIR="$TRT_ROOT/include" \
  -DTENSORRT_LIBRARY="$TRT_ROOT/lib/libnvinfer.so" \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12

cmake --build build/tensorrt -j$(nproc)
```

## 4. 编译快速版插件

```bash
rm -rf build/tensorrt_fast

cmake -S deploy/tensorrt \
  -B build/tensorrt_fast \
  -DTENSORRT_INCLUDE_DIR="$TRT_ROOT/include" \
  -DTENSORRT_LIBRARY="$TRT_ROOT/lib/libnvinfer.so" \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DPVSA_TRT_FAST_MATH=ON

cmake --build build/tensorrt_fast -j$(nproc)
```

## 5. 构建插件测试引擎

`86` 对应 RTX A6000。GPU 2 当前负载较高，优先使用 GPU 1：

```bash
mkdir -p work_dirs

CUDA_VISIBLE_DEVICES=1 \
build/tensorrt/pvsa_build_plugin_engine \
  --output work_dirs/pvsa_plugin_smoke.engine \
  --batch 1 \
  --num-heads 8 \
  --qk-dim 256 \
  --dim 256 \
  --height 56 \
  --width 56 \
  --kv-len 64 \
  --topk 8
```

快速版：

```bash
CUDA_VISIBLE_DEVICES=1 \
build/tensorrt_fast/pvsa_build_plugin_engine \
  --output work_dirs/pvsa_plugin_smoke_fast.engine \
  --batch 1 \
  --num-heads 8 \
  --qk-dim 256 \
  --dim 256 \
  --height 56 \
  --width 56 \
  --kv-len 64 \
  --topk 8
```

## 6. 使用 `trtexec` 测试插件引擎

普通版：

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --plugins="$PWD/build/tensorrt/libpvsa_tensorrt_plugins.so" \
  --loadEngine=work_dirs/pvsa_plugin_smoke.engine \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --warmUp=200 \
  --iterations=1000
```

快速版：

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --plugins="$PWD/build/tensorrt_fast/libpvsa_tensorrt_plugins.so" \
  --loadEngine=work_dirs/pvsa_plugin_smoke_fast.engine \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --warmUp=200 \
  --iterations=1000
```

## 7. 完整 PVSA TensorRT 框架测速

完整部署流程：完整 PVSA 模型 -> 固定形状 ONNX（含 PVSA 自定义节点）-> TensorRT 完整引擎 -> trtexec 测速

### 7.1 导出完整 ONNX

先安装依赖（用 `python -m pip`，避免系统 Python 的 PEP 668 限制）：

```bash
python -m pip install onnx
python -m pip install onnxsim
```

导出 512 与 256 固定形状 ONNX（`--checkpoint 0` 表示随机初始化，不加载权重）：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/export_pvsa_onnx.py \
  --config configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --checkpoint 0 \
  --onnx work_dirs/pvsa_full.onnx \
  --input-size 1 3 512 512
python tools/export_pvsa_onnx.py \
  --config configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --checkpoint 0 \
  --onnx work_dirs/pvsa_full_256.onnx \
  --input-size 1 3 256 256
```

如需加载训练好的权重，把 `--checkpoint` 换成权重路径即可：

```bash
python tools/export_pvsa_onnx.py \
  --config configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --checkpoint work_dirs/PVSA/epoch_10.pth \
  --onnx work_dirs/pvsa_full.onnx \
  --input-size 1 3 512 512
```

### 7.2 简化 ONNX（强烈建议）

完整导出的 ONNX 含有数千个动态形状辅助算子（Shape/Gather/Unsqueeze 等），
进入 TensorRT 后会变成真实 GPU 层并阻止层融合，导致引擎明显慢于 PyTorch。
先用 onnxsim 折叠这些算子，再用简化后的 ONNX 构建引擎：

```bash
cd /media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-v3.0 && git pull

python tools/analysis_tools/simplify_pvsa_onnx.py work_dirs/pvsa_full.onnx \
  --output work_dirs/pvsa_full_sim.onnx \
  --input-shape 1 3 512 512
python tools/analysis_tools/simplify_pvsa_onnx.py work_dirs/pvsa_full_256.onnx \
  --output work_dirs/pvsa_full_256_sim.onnx \
  --input-shape 1 3 256 256
```

### 7.3 构建完整 TensorRT 引擎

```bash
export FULL_ONNX=work_dirs/pvsa_full_256_sim.onnx
export FULL_ENGINE=work_dirs/pvsa_full_256.engine

CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --onnx="$FULL_ONNX" \
  --staticPlugins="$PWD/build/tensorrt/libpvsa_tensorrt_plugins.so" \
  --saveEngine="$FULL_ENGINE" \
  --verbose
```

若未执行 7.2，把 `FULL_ONNX` 换成未简化的 `work_dirs/pvsa_full_256.onnx` 即可。

### 7.4 完整 TensorRT 引擎测速

普通推理：

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --staticPlugins="$PWD/build/tensorrt/libpvsa_tensorrt_plugins.so" \
  --loadEngine="$FULL_ENGINE" \
  --warmUp=500 \
  --duration=0 \
  --iterations=1000 \
  --useSpinWait
```

CUDA Graph 推理：

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --staticPlugins="$PWD/build/tensorrt/libpvsa_tensorrt_plugins.so" \
  --loadEngine="$FULL_ENGINE" \
  --warmUp=500 \
  --duration=0 \
  --iterations=1000 \
  --useCudaGraph \
  --useSpinWait
```
