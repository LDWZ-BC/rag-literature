"""Model management — lifecycle management for model services.

Provides:
- LLamaServerManager: auto-start/stop llama-server with idle sleep
- check_model_exists, check_gpu_available: utility checks
"""

import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from ..config import (MODEL_GENERATION, MODEL_TRANSLATE_1_5B, MODEL_TRANSLATE_7B,
    LLAMA_SERVER_BIN, SERVER_HOST, SERVER_PORT, IDLE_SLEEP_SECONDS,
    SERVER_START_TIMEOUT, SERVER_TIMEOUT)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Utility functions (status, checks)
# ------------------------------------------------------------------

def check_gpu_available() -> bool:
    """Check if CUDA GPU is available via torch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_model_exists() -> dict:
    """Check which model files exist on disk."""
    models = {
        "HY-MT1.5-7B-Q4_K_M.gguf": MODEL_TRANSLATE_7B,
        "HY-MT1.5-1.8B_bf16_Q4_K_M.gguf": MODEL_TRANSLATE_1_5B,
        "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf": MODEL_GENERATION,
    }
    status = {}
    for name, path_str in models.items():
        p = Path(path_str) if path_str else None
        status[name] = p.exists() if p else False
    return status


def model_status() -> dict:
    """Return a dict describing current system status."""
    return {
        "gpu_available": check_gpu_available(),
        "models": check_model_exists(),
        "server_running": _check_port(),
    }


def _cuda_lib_dir() -> str:
    """Find a directory containing the CUDA runtime libs llama-server needs.

    The bundled lib dir next to the binary used to work but its symlinks
    are broken on this machine, so fall back to the dp1 conda env lib.
    """
    candidates = [
        Path(LLAMA_SERVER_BIN).parent / "lib",
        Path.home() / "miniconda3/envs/dp1/lib",
        Path.home() / "anaconda3/envs/dp1/lib",
    ]
    for d in candidates:
        if (d / "libcudart.so.12").exists():
            return str(d)
    return str(candidates[0])




def _check_port(host: str = SERVER_HOST, port: int = SERVER_PORT,
                timeout: float = 2.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex((host, port))
        s.close()
        return result == 0
    except Exception:
        return False


class LLamaServerManager:
    """Manages llama-server lifecycle with idle auto-sleep.

    Singleton — only one server instance per process.
    Starts server on first translation request.
    Server auto-sleeps after IDLE_SLEEP_SECONDS of inactivity (frees GPU memory).
    Wakes automatically on next request.

    Supports multiple models (translation + generation), time-shared on GPU.
    Only one model server runs at a time; switch_model() stops the current
    one before starting the next.
    """

    _instance = None
    _lock = threading.Lock()

    MODELS = {
        "translate": {
            "path": MODEL_TRANSLATE_7B,
            "port": SERVER_PORT,
        },
        "translate_1_5b": {
            "path": MODEL_TRANSLATE_1_5B,
            "port": 8083,
        },
        "gemma4": {
            "path": MODEL_GENERATION,
            "port": 8082,
        },
    }

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._server_lock = threading.Lock()  # Serialize start/stop
        self.current_model = "translate"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_running(self) -> bool:
        """Start or adopt the server for the current model. Returns True if ready."""
        return self._start()

    def mark_activity(self) -> None:
        """No-op: idle tracking handled by --sleep-idle-seconds."""
        pass

    def stop(self) -> None:
        """Stop the server if running."""
        self._kill_server()

    def is_running(self) -> bool:
        """Check if server process is alive AND port is open."""
        return self._is_alive()

    def current_port(self) -> int:
        """Port of the currently selected model."""
        return self.MODELS.get(self.current_model, {}).get("port", SERVER_PORT)

    def switch_model(self, target: str) -> bool:
        """Switch to another model, stopping the current server first.

        Args:
            target: one of MODELS keys ('translate', 'gemma4', ...)

        Returns:
            True if the target server is running after the switch.
        """
        if target not in self.MODELS:
            logger.error("Unknown model target: %s", target)
            return False
        if target == self.current_model and self._is_alive():
            return True

        logger.info("Switching model: %s -> %s", self.current_model, target)
        self._kill_server()
        self.current_model = target
        return self._start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_alive(self) -> bool:
        """Check if server is running by checking the process + port."""
        if self.process is None:
            return False
        if self.process.poll() is not None:
            return False
        if not _check_port(port=self.current_port()):
            return False
        return True

    def _find_pid_on_port(self) -> Optional[int]:
        """Find PID listening on the current model's port via /proc.

        /proc/net/tcp lists an inode, not a PID, so map inode -> PID by
        scanning /proc/*/fd socket symlinks.
        """
        port_hex = f"{self.current_port():04x}"
        try:
            with open("/proc/net/tcp") as f:
                lines = f.readlines()
            inode = None
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                local_addr = parts[1]
                if local_addr.lower().endswith(f":{port_hex}"):
                    inode = parts[9]
                    break
            if inode is None:
                return None
            for proc in Path("/proc").iterdir():
                if not (proc.name.isdigit() and (proc / "fd").is_dir()):
                    continue
                try:
                    fds = list((proc / "fd").iterdir())
                except PermissionError:
                    continue
                for fd in fds:
                    try:
                        if fd.resolve().name == f"socket:[{inode}]":
                            return int(proc.name)
                    except (OSError, ValueError):
                        continue
        except Exception:
            pass
        return None

    def _start(self) -> bool:
        """Start llama-server subprocess or adopt existing one. Thread-safe."""
        cfg = self.MODELS.get(self.current_model, {})
        model_path = cfg.get("path", MODEL_TRANSLATE_7B)
        port = cfg.get("port", SERVER_PORT)
        with self._server_lock:
            if self._is_alive():
                return True

            existing_pid = self._find_pid_on_port()
            if existing_pid:
                logger.info("Adopting existing llama-server (PID %d)", existing_pid)
                self.process = subprocess.Popen(["true"])
                return True

            if not Path(LLAMA_SERVER_BIN).exists():
                logger.error("llama-server binary not found: %s", LLAMA_SERVER_BIN)
                return False
            if not Path(model_path).exists():
                logger.error("Model not found: %s", model_path)
                return False

            logger.info("Starting llama-server (model: %s) ...", Path(model_path).name)
            try:
                llama_env = {**dict(os.environ),
                             "LD_LIBRARY_PATH": _cuda_lib_dir()}
                self.process = subprocess.Popen(
                    [str(LLAMA_SERVER_BIN), "-m", model_path,
                     "-ngl", "999", "--flash-attn", "on",
                     "-c", "4096", "--host", SERVER_HOST,
                     "--port", str(port),
                     "--sleep-idle-seconds", str(IDLE_SLEEP_SECONDS)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=llama_env,
                )
            except Exception as e:
                logger.error("Failed to launch llama-server: %s", e)
                self.process = None
                return False

            for i in range(SERVER_START_TIMEOUT):
                time.sleep(1)
                if _check_port(port=port):
                    logger.info("llama-server started on port %d (PID %d)", port, self.process.pid)
                    return True

            logger.error("llama-server failed to start within %ds", SERVER_START_TIMEOUT)
            self._kill_server()
            return False

    def _kill_server(self):
        """Force-kill the server process (by PID or by port)."""
        with self._server_lock:
            if self.process is not None and self.process.pid:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except Exception:
                    try:
                        self.process.kill()
                        self.process.wait(timeout=3)
                    except Exception:
                        pass

            if _check_port(port=self.current_port()):
                port_pid = self._find_pid_on_port()
                if port_pid:
                    try:
                        os.kill(port_pid, signal.SIGTERM)
                        time.sleep(0.5)
                        if _check_port(port=self.current_port()):
                            os.kill(port_pid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                self.process = None

    def __del__(self):
        """Cleanup on destruction."""
        self.stop()


# --- Singleton instance ---
server_manager = LLamaServerManager()
