"""隔离 CPU/GPU 差异；选择 GPU 时出错会直接报告，不会悄悄改用 CPU。"""

import platform
import sys

import numpy as np


class Backend:
    def __init__(self, device="gpu"):
        if device not in ("cpu", "gpu"):
            raise ValueError("device 必须为 cpu 或 gpu")
        self.device = device
        self.xp = np
        if device == "gpu":
            try:
                import cupy as cp

                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("没有可用的 CUDA GPU")
                cp.cuda.Device(0).use()
                self.xp = cp
            except Exception as exc:
                raise RuntimeError(
                    "GPU 初始化失败，请确认已激活 bp_manual_gpu 并通过 CuPy 环境检查。"
                    f"原始错误：{exc}"
                ) from exc

    def array(self, value, dtype=None):
        return self.xp.asarray(value, dtype=dtype)

    def cpu(self, value):
        return np.asarray(value) if self.device == "cpu" else self.xp.asnumpy(value)

    def scalar(self, value):
        return float(value.item())

    def synchronize(self):
        # CUDA 是异步执行的。计时前后同步，才能量到实际运算时间。
        if self.device == "gpu":
            self.xp.cuda.Stream.null.synchronize()

    def info(self):
        result = {
            "device": self.device,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        }
        if self.device == "gpu":
            cp = self.xp
            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            result.update(
                cupy=cp.__version__,
                gpu=name.decode() if isinstance(name, bytes) else str(name),
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                cuda_driver=cp.cuda.runtime.driverGetVersion(),
            )
        return result
