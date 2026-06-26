"""对比不同设备（NPU/GPU/CPU）和模型的推理性能。

生成详细的性能报告，包括延迟、吞吐量、首次推理耗时（含编译）等指标。

用法:
    python examples/benchmark_compare.py
    python examples/benchmark_compare.py --models resnet18.xml resnet50.xml
    python examples/benchmark_compare.py --devices NPU GPU CPU
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from npu_engine import NPUEngine


def benchmark_device(model_path: str, device: str, input_shape: tuple,
                     warmup: int = 5, iterations: int = 50) -> dict:
    """在指定设备上测试模型性能。"""
    # 首次编译计时
    t0 = time.perf_counter()
    try:
        engine = NPUEngine(model_path, device=device, input_shape=input_shape)
    except Exception as e:
        return {"error": str(e)}

    compile_time = (time.perf_counter() - t0) * 1000
    actual_device = engine.device  # 可能发生了回退

    # 预热
    dummy = np.random.rand(*input_shape).astype(np.float32)
    for _ in range(warmup):
        engine.infer(dummy)

    # 正式测速
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.infer(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    fps = 1000.0 / avg_ms

    return {
        "requested_device": device,
        "actual_device": actual_device,
        "compile_time_ms": compile_time,
        "avg_latency_ms": avg_ms,
        "std_latency_ms": std_ms,
        "min_latency_ms": min_ms,
        "max_latency_ms": max_ms,
        "throughput_fps": fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="设备与模型性能对比")
    parser.add_argument("--models", nargs="+", default=["models/resnet50.xml"],
                        help="模型路径列表")
    parser.add_argument("--devices", nargs="+", default=["NPU", "GPU", "CPU"],
                        help="设备列表")
    parser.add_argument("--size", type=int, default=224, help="输入尺寸")
    parser.add_argument("--iterations", type=int, default=50, help="测试迭代次数")
    parser.add_argument("--warmup", type=int, default=5, help="预热次数")
    args = parser.parse_args()

    input_shape = (1, 3, args.size, args.size)

    print(f"{'='*80}")
    print(f"性能对比测试")
    print(f"{'='*80}")
    print(f"输入尺寸: {input_shape}")
    print(f"预热: {args.warmup} 次，测试: {args.iterations} 次\n")

    results = []
    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"⚠️  模型不存在: {model_path}，跳过")
            continue

        model_name = Path(model_path).stem
        print(f"📦 模型: {model_name}")
        print(f"{'-'*80}")

        for device in args.devices:
            print(f"  测试 {device}...", end=" ", flush=True)
            result = benchmark_device(model_path, device, input_shape,
                                     args.warmup, args.iterations)

            if "error" in result:
                print(f"❌ 失败: {result['error']}")
                continue

            # 如果发生了回退，标注
            fallback = ""
            if result["actual_device"] != result["requested_device"]:
                fallback = f" (回退到 {result['actual_device']})"

            print(f"✅ {result['avg_latency_ms']:.2f}ms  {result['throughput_fps']:.0f}FPS{fallback}")

            results.append({
                "model": model_name,
                **result,
            })

        print()

    # 汇总表格
    if results:
        print(f"{'='*80}")
        print("汇总表格")
        print(f"{'='*80}")
        print(f"{'模型':<20} {'请求设备':<10} {'实际设备':<10} {'编译(ms)':<12} {'平均(ms)':<12} {'吞吐(FPS)':<12}")
        print(f"{'-'*80}")
        for r in results:
            print(f"{r['model']:<20} {r['requested_device']:<10} {r['actual_device']:<10} "
                  f"{r['compile_time_ms']:<12.1f} {r['avg_latency_ms']:<12.2f} {r['throughput_fps']:<12.0f}")


if __name__ == "__main__":
    main()
