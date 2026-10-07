from typing import Any

import lightning as L
import psutil


class SysMetricsCallback(L.Callback):
    """Host RAM and, when present, GPU utilisation/memory. Sampled by the heartbeat, so it
    runs on the heartbeat interval. Degrades to CPU-only fields without a GPU."""

    def __init__(self) -> None:
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:
            self._nvml = None

    def fields(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        out: dict[str, Any] = {
            "host_ram_used_gb": round(vm.used / 2**30, 2),
            "host_ram_frac": round(vm.percent / 100, 3),
            "cpu_util": round(psutil.cpu_percent(interval=None) / 100, 3),
        }
        if self._nvml is not None:
            nv, h = self._nvml
            try:
                out["gpu_util"] = nv.nvmlDeviceGetUtilizationRates(h).gpu / 100
                out["gpu_mem_used_gb"] = round(nv.nvmlDeviceGetMemoryInfo(h).used / 2**30, 2)
            except Exception:
                pass
        return out
