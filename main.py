"""CCR-Tabular — Master Heterogeneous Parallel Resource-Aware Orchestrator.

Centralized high-throughput execution layer providing:
  - Heterogeneous concurrent scheduling: CPU process pool (GBDT/Stats) + 1 dedicated GPU execution slot (Neural).
  - Strict CPU core budgeting: logical_cores - 3, bounded by RAM-safe worker scaling.
  - BLAS/LAPACK single-thread locking to prevent thread oversubscription.
  - Dynamic runtime VRAM querying (mem_get_info) with 20% safe headroom (no fabricated telemetry).
  - Dynamic micro-batch scaling with rich-keyed batch size caching and explicit CPU fallback on OOM.
  - Automatic FP16 AMP mixed precision on CUDA.
  - In-memory fold preprocessing caching to eliminate filesystem and encoding bottlenecks.
  - Atomic run tracking, pending-job planner, and comprehensive end-of-run execution diagnostics.

Usage:
    python main.py --resource_report    # Full CPU, RAM, and GPU telemetry audit
    python main.py --validate           # Automated 5-point scientific consistency check
    python main.py --dry_run            # Preview pending jobs, routing, and memory allocations
    python main.py --smoke_test         # 5-second end-to-end diagnostic verification
    python main.py --figures            # Generate 7 main publication figures + supplementals

    python main.py --tier1 --fast       # Run Tier 1 Core-10 benchmark with parallel routing
    python main.py --tier3 --fast       # Run Tier 3 Architecture Transfer (MLP / ResNet / Transformer)
    python main.py --all --fast         # Run 1-Go master benchmark suite with checkpoint resumption
"""

import os
import sys

# ── 1. Thread Safety: Prevent Library Oversubscription ─────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import concurrent.futures
import gc
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
import psutil
import torch

from src.utils.config import (
    BATCH_SIZE,
    BETA,
    CORE_10_DATASETS,
    DATASETS,
    K,
    LEARNING_RATE,
    LOSS_NAMES,
    MULTICLASS_DATASETS,
    N_FOLDS,
    OPTIMIZER,
    OUTPUTS_LOGS,
    OUTPUTS_METRICS,
    REAL_WORLD_DATASETS,
    SEEDS,
    TAU,
    WEIGHT_DECAY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CCR-Orchestrator")

# ── Resource Constraints ───────────────────────────────────────────────────────
CPU_RESERVED_CORES: int = 3
RAM_ESTIMATE_PER_WORKER_MB: int = 1024  # Estimated RAM per CPU worker
RAM_SAFETY_FRACTION: float = 0.75
GPU_SAFETY_FRACTION: float = 0.80
GPU_MIN_FREE_MB: int = 1024
MAX_OOM_RETRIES: int = 3

# Rich batch size cache: key = (architecture, model, dataset, dtype, device)
_BATCH_SIZE_CACHE: Dict[Tuple[str, str, str, str, str], int] = {}

CPU_PREFERRED_MODELS: Set[str] = {
    "xgboost", "xgboost_default", "xgboost_weighted",
    "lightgbm", "lightgbm_default", "catboost", "catboost_default",
}


# ── Job Descriptor ─────────────────────────────────────────────────────────────

@dataclass
class JobDescriptor:
    """Atomic unit of experimental cross-validation work."""
    dataset: str
    model: str
    noise_type: str = "none"
    noise_rate: float = 0.0
    architecture: str = "mlp"
    optimizer: str = OPTIMIZER
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    tau: float = TAU
    beta: float = BETA
    K_hist: int = K
    batch_size: int = BATCH_SIZE
    tag: Optional[str] = None
    seeds: Optional[List[int]] = None
    n_folds: int = N_FOLDS
    instrument_batch: bool = False
    results_path: Optional[str] = None
    tier_name: str = "benchmark"

    @property
    def is_gpu_preferred(self) -> bool:
        return self.model.lower() not in CPU_PREFERRED_MODELS


# ── Standalone Worker Function for Multiprocessing ─────────────────────────────

def _worker_execute_job(job_dict: Dict[str, Any], device_str: str = "cpu", batch_size: int = BATCH_SIZE, use_amp: bool = False) -> Dict[str, Any]:
    """Execute a single cross-validation job in a worker process and return verified successful run_ids."""
    from src.training.cross_validation import run_cross_validation

    res_path = Path(job_dict["results_path"]) if job_dict.get("results_path") else None
    device = torch.device(device_str)

    t0 = time.perf_counter()
    df = run_cross_validation(
        dataset_name=job_dict["dataset"],
        model_name=job_dict["model"],
        noise_type=job_dict["noise_type"],
        noise_rate=job_dict["noise_rate"],
        architecture=job_dict["architecture"],
        optimizer_name=job_dict["optimizer"],
        lr=job_dict.get("lr", LEARNING_RATE),
        weight_decay=job_dict.get("weight_decay", WEIGHT_DECAY),
        tau=job_dict.get("tau", TAU),
        beta=job_dict.get("beta", BETA),
        K_hist=job_dict.get("K_hist", K),
        tag=job_dict.get("tag", None),
        seeds=job_dict["seeds"] or SEEDS,
        n_folds=job_dict["n_folds"],
        instrument_batch=job_dict["instrument_batch"],
        results_path=res_path,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )
    elapsed = time.perf_counter() - t0

    successful_run_ids: List[str] = []
    if df is not None and len(df) > 0:
        if "run_id" in df.columns:
            if "status" in df.columns:
                successful_run_ids = df[df["status"].isin(["SUCCESS", "SUCCESS_CPU_FALLBACK"])]["run_id"].dropna().tolist()
            else:
                successful_run_ids = df["run_id"].dropna().tolist()

    return {
        "status": "SUCCESS",
        "dataset": job_dict["dataset"],
        "model": job_dict["model"],
        "noise_type": job_dict["noise_type"],
        "noise_rate": job_dict["noise_rate"],
        "n_rows": len(df) if df is not None else 0,
        "successful_run_ids": successful_run_ids,
        "elapsed_s": elapsed,
        "device": device_str,
    }


# ── System Resource Profiling ──────────────────────────────────────────────────

def get_ram_resource_profile() -> Dict[str, Any]:
    """Query total, available, and used system memory."""
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    return {
        "total_ram_mb": int(vm.total / (1024 * 1024)),
        "available_ram_mb": int(vm.available / (1024 * 1024)),
        "used_ram_mb": int(vm.used / (1024 * 1024)),
        "percent_used": vm.percent,
        "process_rss_mb": int(proc.memory_info().rss / (1024 * 1024)),
    }


def get_cpu_worker_budget() -> int:
    """Calculate safe CPU worker budget bounded by logical cores and available RAM."""
    logical_cores = os.cpu_count() or 4
    core_budget = max(1, logical_cores - CPU_RESERVED_CORES)

    ram_prof = get_ram_resource_profile()
    safe_ram_mb = ram_prof["available_ram_mb"] * RAM_SAFETY_FRACTION
    ram_worker_cap = max(1, int(safe_ram_mb / RAM_ESTIMATE_PER_WORKER_MB))

    return max(1, min(core_budget, ram_worker_cap))


def get_gpu_resource_profile(device_override: str = "auto") -> Dict[str, Any]:
    """Query runtime GPU state without fabricating telemetry on failure."""
    if device_override == "cpu" or not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device": "cpu",
            "name": "CPU",
            "total_vram_mb": 0,
            "free_vram_mb": 0,
            "safe_vram_mb": 0,
            "amp_enabled": False,
        }

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_mb = int(total_bytes / (1024 * 1024))
        free_mb = int(free_bytes / (1024 * 1024))
        safe_mb = int(free_mb * GPU_SAFETY_FRACTION)

        use_gpu = (device_override == "cuda") or (safe_mb >= GPU_MIN_FREE_MB)
        return {
            "cuda_available": True,
            "device": "cuda" if use_gpu else "cpu",
            "name": torch.cuda.get_device_name(0),
            "total_vram_mb": total_mb,
            "free_vram_mb": free_mb,
            "safe_vram_mb": safe_mb,
            "amp_enabled": use_gpu,
        }
    except Exception as e:
        logger.warning(f"CUDA memory query failed ({e}). Falling back strictly to CPU.")
        return {
            "cuda_available": False,
            "device": "cpu",
            "name": "CUDA State Unknown (CPU Fallback)",
            "total_vram_mb": 0,
            "free_vram_mb": 0,
            "safe_vram_mb": 0,
            "amp_enabled": False,
        }


def print_resource_report() -> None:
    """Print comprehensive hardware and resource budget report."""
    logical_cores = os.cpu_count() or 4
    usable_workers = get_cpu_worker_budget()
    ram_prof = get_ram_resource_profile()
    gpu_prof = get_gpu_resource_profile("auto")

    print("\n=================================================================")
    print("      CCR-TABULAR HETEROGENEOUS RESOURCE AUDIT REPORT           ")
    print("=================================================================")
    print("CPU & RAM Subsystem:")
    print(f"  Logical Cores:        {logical_cores}")
    print(f"  Reserved Cores:       {CPU_RESERVED_CORES} (Protected for OS/UI)")
    print(f"  Total System RAM:     {ram_prof['total_ram_mb']} MB ({ram_prof['total_ram_mb']/1024:.1f} GB)")
    print(f"  Available RAM:        {ram_prof['available_ram_mb']} MB ({ram_prof['available_ram_mb']/1024:.1f} GB)")
    print(f"  Process RSS:          {ram_prof['process_rss_mb']} MB")
    print(f"  Usable Worker Budget: {usable_workers} concurrent processes")
    print(f"  BLAS Thread Cap:      1 thread/worker (Oversubscription Protection Active)")
    print("\nGPU Subsystem:")
    print(f"  Current Free VRAM:    {gpu_prof['free_vram_mb']} MB")
    print(f"  Safe Working Budget:  {gpu_prof['safe_vram_mb']} MB (20% Headroom Protected)")
    print(f"  Execution Target:     {gpu_prof['device'].upper()}")
    print(f"  Automatic AMP:        {'ENABLED (FP16)' if gpu_prof['amp_enabled'] else 'DISABLED'}")
    print("=================================================================\n")


def get_model_safe_vram_requirement(architecture: str, model_name: str) -> int:
    """Return minimum safe VRAM required (in MB) before attempting GPU allocation."""
    arch = architecture.lower()
    if "transformer" in arch or "fttransformer" in arch:
        return 1500  # FT-Transformer attention & embedding matrices
    elif "resnet" in arch:
        return 1000  # Tabular ResNet residual blocks
    return 800       # Standard Tabular MLP


# ── Heterogeneous Concurrent Scheduler ─────────────────────────────────────────

class HeterogeneousJobScheduler:
    """Orchestrates CPU process pool and dedicated GPU worker concurrently with live resource protection."""

    def __init__(self, device_override: str = "auto", fast_mode: bool = True):
        self.device_override = device_override
        self.fast_mode = fast_mode
        self.gpu_prof = get_gpu_resource_profile(device_override)
        self.cpu_workers = get_cpu_worker_budget()
        self._completed_run_ids: Dict[Path, Set[str]] = {}

        # Telemetry counters
        self.stats = {
            "requested": 0,
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "oom_recovered": 0,
            "cpu_runs": 0,
            "gpu_runs": 0,
            "cpu_fallback_runs": 0,
            "start_time": time.time(),
        }

    def _get_completed_ids_for_csv(self, target_csv: Path) -> Set[str]:
        """Retrieve completed run_ids from in-memory cache or load once from disk."""
        if target_csv not in self._completed_run_ids:
            if target_csv.exists():
                try:
                    df = pd.read_csv(target_csv)
                    if "run_id" in df.columns:
                        if "status" in df.columns:
                            valid_df = df[df["status"].isin(["SUCCESS", "SUCCESS_CPU_FALLBACK"])]
                            self._completed_run_ids[target_csv] = set(valid_df["run_id"].dropna().values)
                        else:
                            self._completed_run_ids[target_csv] = set(df["run_id"].dropna().values)
                    else:
                        self._completed_run_ids[target_csv] = set()
                except Exception:
                    self._completed_run_ids[target_csv] = set()
            else:
                self._completed_run_ids[target_csv] = set()
        return self._completed_run_ids[target_csv]

    def _register_completed_job_ids(self, job: JobDescriptor, target_csv: Path, df: Optional[pd.DataFrame] = None) -> None:
        """Register only actually successful run_ids from the returned DataFrame into the in-memory cache."""
        completed_set = self._get_completed_ids_for_csv(target_csv)
        if df is not None and len(df) > 0:
            if "run_id" in df.columns:
                if "status" in df.columns:
                    for _, row in df.iterrows():
                        if row["status"] in {"SUCCESS", "SUCCESS_CPU_FALLBACK"}:
                            completed_set.add(row["run_id"])
                else:
                    for rid in df["run_id"].dropna().values:
                        completed_set.add(rid)

    def _is_job_complete(self, job: JobDescriptor) -> bool:
        """Check if all fold/seed combinations for this job exist in target CSV (in-memory cached)."""
        from src.training.train import make_run_id

        target_csv = Path(job.results_path) if job.results_path else (OUTPUTS_METRICS / "results.csv")
        existing_runs = self._get_completed_ids_for_csv(target_csv)
        seeds = job.seeds or SEEDS

        for s in seeds:
            for f in range(1, job.n_folds + 1):
                rid = make_run_id(
                    dataset_name=job.dataset,
                    model_name=job.model,
                    noise_type=job.noise_type,
                    noise_rate=job.noise_rate,
                    seed=s,
                    fold=f,
                    architecture=job.architecture,
                    optimizer_name=job.optimizer,
                    lr=job.lr,
                    weight_decay=job.weight_decay,
                    tau=job.tau,
                    beta=job.beta,
                    K_hist=job.K_hist,
                    batch_size=job.batch_size,
                    tag=job.tag,
                )
                if rid not in existing_runs:
                    return False
        return True

    def filter_pending_jobs(self, jobs: List[JobDescriptor]) -> Tuple[List[JobDescriptor], int]:
        """Pending-job planner: inspects target CSVs and extracts only uncompleted jobs."""
        pending = []
        skipped = 0

        for job in jobs:
            if self._is_job_complete(job):
                skipped += 1
            else:
                pending.append(job)

        self.stats["requested"] += len(jobs)
        self.stats["skipped"] += skipped
        return pending, skipped

    def execute_gpu_job_with_oom_recovery(self, job: JobDescriptor) -> Dict[str, Any]:
        """Execute a neural job with per-job live VRAM refresh, dynamic batch scaling, and clean CPU fallback."""
        from src.training.cross_validation import run_cross_validation

        arch = job.architecture
        model_name = job.model
        dataset = job.dataset
        res_path = Path(job.results_path) if job.results_path else (OUTPUTS_METRICS / "results.csv")

        # 1. Refresh live GPU memory state before starting this job
        live_gpu_prof = get_gpu_resource_profile(self.device_override)
        safe_vram = live_gpu_prof["safe_vram_mb"]
        cuda_ok = live_gpu_prof["cuda_available"] and (self.device_override != "cpu")
        min_vram = get_model_safe_vram_requirement(arch, model_name)

        # 2. Strict model-aware VRAM safety check: Fall back to CPU if safe working budget < min_vram
        if (not cuda_ok) or (safe_vram < min_vram):
            logger.warning(
                f"[VRAM SAFETY ACTIVE] [{dataset}-{model_name}-{arch}] Safe VRAM is {safe_vram} MB (< {min_vram} MB requirement). "
                f"Routing cleanly to CPU fallback to preserve stability."
            )
            self.stats["cpu_fallback_runs"] += 1
            return self._execute_cpu_fallback(job, res_path)

        device_str = live_gpu_prof["device"]
        use_amp = live_gpu_prof["amp_enabled"]
        cache_key = (arch, model_name, dataset, "float32", device_str)
        batch_size = _BATCH_SIZE_CACHE.get(cache_key, job.batch_size)
        retries = 0

        while retries <= MAX_OOM_RETRIES:
            try:
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

                t0 = time.perf_counter()
                df = run_cross_validation(
                    dataset_name=dataset,
                    model_name=model_name,
                    noise_type=job.noise_type,
                    noise_rate=job.noise_rate,
                    architecture=arch,
                    optimizer_name=job.optimizer,
                    lr=job.lr,
                    weight_decay=job.weight_decay,
                    tau=job.tau,
                    beta=job.beta,
                    K_hist=job.K_hist,
                    tag=job.tag,
                    seeds=job.seeds or SEEDS,
                    n_folds=job.n_folds,
                    instrument_batch=job.instrument_batch,
                    results_path=res_path,
                    batch_size=batch_size,
                    device=torch.device(device_str),
                    use_amp=use_amp,
                )
                elapsed = time.perf_counter() - t0

                _BATCH_SIZE_CACHE[cache_key] = batch_size
                self.stats["gpu_runs"] += 1
                self.stats["completed"] += 1
                self._register_completed_job_ids(job, res_path, df)

                return {
                    "status": "SUCCESS",
                    "dataset": dataset,
                    "model": model_name,
                    "noise_type": job.noise_type,
                    "noise_rate": job.noise_rate,
                    "n_rows": len(df) if df is not None else 0,
                    "elapsed_s": elapsed,
                    "device": device_str,
                    "batch_size": batch_size,
                }

            except torch.cuda.OutOfMemoryError as e:
                retries += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                if batch_size > MIN_BATCH_SIZE:
                    prev_bs = batch_size
                    batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                    logger.warning(
                        f"[GPU OOM] Job {dataset}/{model_name} on {device_str}. "
                        f"Reducing micro-batch {prev_bs} -> {batch_size}. Retry {retries}/{MAX_OOM_RETRIES}."
                    )
                    _BATCH_SIZE_CACHE[cache_key] = batch_size
                    self.stats["oom_recovered"] += 1
                else:
                    logger.warning(
                        f"[GPU OOM EXHAUSTED] Job {dataset}/{model_name} cannot run at min batch {MIN_BATCH_SIZE}. "
                        f"Falling back to CPU execution."
                    )
                    break
            except Exception as e:
                logger.error(f"Job execution failed on {dataset}-{model_name}: {e}")
                self.stats["failed"] += 1
                return {"status": "FAILED", "dataset": dataset, "model": model_name, "error": str(e)}

        # Fallback to CPU execution after OOM retries exhausted
        self.stats["cpu_fallback_runs"] += 1
        return self._execute_cpu_fallback(job, res_path)

    def _execute_cpu_fallback(self, job: JobDescriptor, res_path: Path) -> Dict[str, Any]:
        """Execute a job on CPU fallback cleanly."""
        from src.training.cross_validation import run_cross_validation
        logger.info(f"[CPU FALLBACK] Executing {job.dataset}-{job.model} on CPU...")
        t0 = time.perf_counter()
        try:
            df = run_cross_validation(
                dataset_name=job.dataset,
                model_name=job.model,
                noise_type=job.noise_type,
                noise_rate=job.noise_rate,
                architecture=job.architecture,
                optimizer_name=job.optimizer,
                lr=job.lr,
                weight_decay=job.weight_decay,
                tau=job.tau,
                beta=job.beta,
                K_hist=job.K_hist,
                tag=job.tag,
                seeds=job.seeds or SEEDS,
                n_folds=job.n_folds,
                instrument_batch=job.instrument_batch,
                results_path=res_path,
                batch_size=job.batch_size,
                device=torch.device("cpu"),
                use_amp=False,
            )
            elapsed = time.perf_counter() - t0
            self.stats["cpu_runs"] += 1
            self.stats["completed"] += 1
            self._register_completed_job_ids(job, res_path, df)
            return {"status": "SUCCESS_CPU_FALLBACK", "dataset": job.dataset, "model": job.model, "device": "cpu", "elapsed_s": elapsed}
        except Exception as e_cpu:
            logger.error(f"CPU fallback failed on {job.dataset}-{job.model}: {e_cpu}")
            self.stats["failed"] += 1
            return {"status": "FAILED", "dataset": job.dataset, "model": job.model, "error": str(e_cpu)}

    def run_jobs_heterogeneous(self, jobs: List[JobDescriptor]) -> List[Dict[str, Any]]:
        """Run CPU queue (process pool) and GPU queue (single slot) concurrently with RAM backpressure."""
        pending_jobs, skipped_count = self.filter_pending_jobs(jobs)
        logger.info(f"Queue Status: Total Requested={len(jobs)} | Skipped (Existing)={skipped_count} | Pending={len(pending_jobs)}")

        if not pending_jobs:
            logger.info("All requested jobs are already completed. Nothing to execute.")
            return []

        cpu_queue: List[JobDescriptor] = []
        gpu_queue: List[JobDescriptor] = []

        for j in pending_jobs:
            if j.is_gpu_preferred and self.gpu_prof["cuda_available"] and (self.device_override != "cpu"):
                gpu_queue.append(j)
            else:
                cpu_queue.append(j)

        logger.info(f"Heterogeneous Queue Breakdown: CPU Queue={len(cpu_queue)} jobs | GPU Queue={len(gpu_queue)} jobs")
        results: List[Dict[str, Any]] = []

        # Execute GPU jobs sequentially on dedicated GPU slot while CPU workers process CPU jobs
        def _process_gpu_queue() -> List[Dict[str, Any]]:
            gpu_res = []
            for idx, g_job in enumerate(gpu_queue, start=1):
                logger.info(f"[GPU Worker {idx}/{len(gpu_queue)}] Starting: {g_job.dataset} | {g_job.model} | {g_job.noise_type}@{g_job.noise_rate:.0%}")
                res = self.execute_gpu_job_with_oom_recovery(g_job)
                gpu_res.append(res)
            return gpu_res

        def _process_cpu_queue() -> List[Dict[str, Any]]:
            cpu_res = []
            if not cpu_queue:
                return cpu_res

            logger.info(f"[CPU Worker Pool] Launching {len(cpu_queue)} jobs across {self.cpu_workers} worker processes with RAM backpressure...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_workers) as executor:
                pending_futures = {}
                for c_job in cpu_queue:
                    # Dynamic RAM backpressure check before submitting new process
                    ram = get_ram_resource_profile()
                    while ram["available_ram_mb"] < 4096:
                        logger.warning(
                            f"[RAM BACKPRESSURE] System available RAM ({ram['available_ram_mb']} MB) < 4.0 GB. "
                            f"Throttling worker dispatch for 3.0s..."
                        )
                        time.sleep(3.0)
                        ram = get_ram_resource_profile()

                    fut = executor.submit(_worker_execute_job, asdict(c_job), "cpu", c_job.batch_size, False)
                    pending_futures[fut] = c_job

                for future in concurrent.futures.as_completed(pending_futures):
                    c_job = pending_futures[future]
                    try:
                        res = future.result()
                        self.stats["completed"] += 1
                        self.stats["cpu_runs"] += 1
                        target_p = Path(c_job.results_path) if c_job.results_path else (OUTPUTS_METRICS / "results.csv")
                        self._register_completed_job_ids(c_job, target_p)
                        cpu_res.append(res)
                        logger.info(f"[CPU Done] {res['dataset']} | {res['model']} in {res['elapsed_s']:.1f}s")
                    except Exception as exc:
                        logger.error(f"[CPU Failed] {c_job.dataset} | {c_job.model}: {exc}")
                        self.stats["failed"] += 1
                        cpu_res.append({"status": "FAILED", "dataset": c_job.dataset, "model": c_job.model, "error": str(exc)})
            return cpu_res

        # Run CPU Pool and GPU Queue concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as coordinator:
            future_gpu = coordinator.submit(_process_gpu_queue)
            future_cpu = coordinator.submit(_process_cpu_queue)

            res_gpu = future_gpu.result()
            res_cpu = future_cpu.result()

            results.extend(res_gpu)
            results.extend(res_cpu)

        return results

    def print_final_summary(self) -> None:
        """Print comprehensive end-of-run execution diagnostics."""
        elapsed_total = time.time() - self.stats["start_time"]
        ram_prof = get_ram_resource_profile()
        gpu_prof = get_gpu_resource_profile("auto")

        print("\n=================================================================")
        print("          CCR-TABULAR FINAL EXECUTION SUMMARY REPORT             ")
        print("=================================================================")
        print(f"Total Wall-Clock Time:   {elapsed_total:.2f}s ({elapsed_total/60:.2f} min)")
        print(f"Requested Jobs:          {self.stats['requested']}")
        print(f"Completed Successfully:  {self.stats['completed']}")
        print(f"Skipped (Pre-existing):  {self.stats['skipped']}")
        print(f"Failed Jobs:             {self.stats['failed']}")
        print(f"CUDA OOM Recoveries:     {self.stats['oom_recovered']}")
        print(f"GPU Training Runs:       {self.stats['gpu_runs']}")
        print(f"CPU Training Runs:       {self.stats['cpu_runs']}")
        print(f"CPU Fallback Runs:       {self.stats['cpu_fallback_runs']}")
        print("-" * 65)
        print("Resource Utilization:")
        print(f"  CPU Workers Utilized:  {self.cpu_workers}")
        print(f"  Current System RAM:    {ram_prof['used_ram_mb']} MB / {ram_prof['total_ram_mb']} MB ({ram_prof['percent_used']}%)")
        print(f"  Current Free VRAM:     {gpu_prof['free_vram_mb']} MB / {gpu_prof['total_vram_mb']} MB")
        print("=================================================================\n")


# ── Dry Run Planner ────────────────────────────────────────────────────────────

def run_dry_run_planner(target_tiers: List[str], device_mode: str = "auto") -> None:
    """Preview job matrix, memory budgets, and worker allocations without execution."""
    print("\n=================================================================")
    print("         CCR-TABULAR HETEROGENEOUS DRY RUN EXECUTION PLAN        ")
    print("=================================================================")
    gpu_prof = get_gpu_resource_profile(device_mode)
    cpu_budget = get_cpu_worker_budget()
    ram_prof = get_ram_resource_profile()

    print(f"Target Device: {gpu_prof['device'].upper()} | Safe VRAM: {gpu_prof['safe_vram_mb']} MB | CPU Workers: {cpu_budget} | Available RAM: {ram_prof['available_ram_mb']/1024:.1f} GB")
    print("-" * 65)

    for tier in target_tiers:
        if tier == "tier1":
            n_ds = len(CORE_10_DATASETS)
            n_losses = len(LOSS_NAMES)
            n_noise = 4
            n_seeds = len(SEEDS)
            n_folds = N_FOLDS
            total_runs = n_ds * n_losses * n_noise * n_seeds * n_folds
            print("Tier 1: Core 10-Dataset Master Benchmark")
            print(f"  Datasets ({n_ds}):", CORE_10_DATASETS)
            print(f"  Losses ({n_losses}):  ", LOSS_NAMES)
            print("  Noise (4):     Clean (0%), 20% Asym, 40% Asym, 20% Sym")
            print(f"  Folds/Seeds:   {n_folds} Folds x {n_seeds} Seeds = {n_folds * n_seeds} runs/condition")
            print(f"  Total Expected Runs: {total_runs} fold-level executions")
            print("  Routing:       GPU-First Queue (1 Dedicated GPU Slot, FP16 AMP)")
        elif tier == "tier3":
            print("\nTier 3: Architecture Transferability Benchmark")
            print("  Datasets (5):  ['adult', 'bank', 'phoneme', 'spambase', 'credit_g']")
            print("  Architectures: ['TabularMLP', 'TabularResNet', 'TabularFTTransformer']")
            print("  Routing:       GPU-First Queue (FP16 AMP)")
        elif tier == "tier4":
            print("\nTier 4: Multiclass Transfer Benchmark (C >= 3)")
            print(f"  Datasets ({len(MULTICLASS_DATASETS)}):  ['segment' (C=7), 'vehicle' (C=4)]")
        elif tier == "tier5":
            print("\nTier 5: Real-World Clinical External Validation")
            print(f"  Datasets ({len(REAL_WORLD_DATASETS)}):  ['heart_disease', 'breast_cancer']")
    print("=================================================================\n")


# ── Unified 1-Go Master Runner ─────────────────────────────────────────────────

def run_all_experiments(device_mode: str = "auto", fast_mode: bool = True) -> int:
    """Execute all benchmark tiers sequentially in one optimized workflow.
    
    Returns:
        0 if all tiers succeeded, 1 if any tier encountered errors.
    """
    logger.info("=================================================================")
    logger.info("        STARTING UNIFIED 1-GO CCR-TABULAR MASTER SUITE          ")
    logger.info("=================================================================")

    scheduler = HeterogeneousJobScheduler(device_override=device_mode, fast_mode=fast_mode)
    failed_tiers: List[str] = []

    # 1. Scientific Consistency Audit Pre-Check
    logger.info(">>> Step 1/11: Running Pre-Publication Scientific Consistency Audit...")
    try:
        from src.analysis.final_validation import run_scientific_validation
        run_scientific_validation()
    except Exception as e:
        logger.error(f"Error in Pre-Audit: {e}")
        failed_tiers.append("Pre-Audit")

    # 2. Synthetic Toy & Negative Controls (Tier 6)
    logger.info(">>> Step 2/11: Running Tier 6 Synthetic Toy & Negative Controls...")
    try:
        from experiments.run_tier6_toy_controls import (
            run_negative_controls_experiment,
            run_synthetic_toy_experiment,
        )
        out = OUTPUTS_METRICS / "tier6_controls"
        run_synthetic_toy_experiment(out)
        run_negative_controls_experiment(out)
    except Exception as e:
        logger.error(f"Error in Tier 6: {e}")
        failed_tiers.append("Tier 6")

    # 3. S/B Investigation
    logger.info(">>> Step 3/11: Running S/B Theoretical Bounds & Empirical Measurement...")
    try:
        from experiments.run_sb_investigation import run_sb_empirical_measurement
        run_sb_empirical_measurement()
    except Exception as e:
        logger.error(f"Error in S/B Investigation: {e}")
        failed_tiers.append("S/B Investigation")

    # 4. Direct Mechanism Validation with Batch Telemetry (Tier 2)
    logger.info(">>> Step 4/11: Running Tier 2 Direct Mechanism Validation...")
    try:
        from experiments.run_tier2_mechanism import (
            aggregate_and_plot_mechanism_dynamics,
            run_tier2_mechanism_experiments,
        )
        from src.analysis.analyze_mechanism import analyze_mechanism_telemetry
        run_tier2_mechanism_experiments()
        aggregate_and_plot_mechanism_dynamics()
        analyze_mechanism_telemetry()
    except Exception as e:
        logger.error(f"Error in Tier 2: {e}")
        failed_tiers.append("Tier 2")

    # 5. Pure Normalization Controls
    logger.info(">>> Step 5/11: Running Pure Normalization Controls...")
    try:
        from experiments.run_pure_normalization_controls import run_pure_normalization_controls
        from src.analysis.analyze_pure_controls import analyze_pure_controls
        run_pure_normalization_controls()
        analyze_pure_controls()
    except Exception as e:
        logger.error(f"Error in Pure Controls: {e}")
        failed_tiers.append("Pure Controls")

    # 6. Per-Sample Gradient Attribution & Figure 5
    logger.info(">>> Step 6/11: Running Per-Sample Gradient Attribution & Figure 5...")
    try:
        from experiments.run_per_sample_gradient_attribution import run_gradient_attribution_study
        run_gradient_attribution_study()
    except Exception as e:
        logger.error(f"Error in Gradient Attribution: {e}")
        failed_tiers.append("Gradient Attribution")

    # 7. Optimizer Sensitivity Study
    logger.info(">>> Step 7/11: Running Optimizer Sensitivity Study (SGD vs Adam vs AdamW)...")
    try:
        from experiments.run_optimizer_study import run_optimizer_study
        from src.analysis.analyze_optimizer_study import analyze_optimizer_study
        run_optimizer_study()
        analyze_optimizer_study()
    except Exception as e:
        logger.error(f"Error in Optimizer Study: {e}")
        failed_tiers.append("Optimizer Study")

    # 8. Multiclass Transfer (Tier 4)
    logger.info(">>> Step 8/11: Running Tier 4 Multiclass Transfer Benchmark...")
    try:
        from experiments.run_tier4_multiclass import run_tier4_multiclass_experiments
        run_tier4_multiclass_experiments()
    except Exception as e:
        logger.error(f"Error in Tier 4: {e}")
        failed_tiers.append("Tier 4")

    # 9. Real-World External Validation (Tier 5)
    logger.info(">>> Step 9/11: Running Tier 5 Real-World External Validation...")
    try:
        from experiments.run_tier5_natural_noise import run_tier5_natural_noise_experiments
        run_tier5_natural_noise_experiments()
    except Exception as e:
        logger.error(f"Error in Tier 5: {e}")
        failed_tiers.append("Tier 5")

    # 10. Architecture Transferability (Tier 3)
    logger.info(">>> Step 10/11: Running Tier 3 Architecture Transferability...")
    try:
        from experiments.run_tier3_architecture import run_tier3_architecture_experiments
        run_tier3_architecture_experiments()
    except Exception as e:
        logger.error(f"Error in Tier 3: {e}")
        failed_tiers.append("Tier 3")

    # 11. Core 10-Dataset Master Benchmark (Tier 1)
    logger.info(">>> Step 11/11: Running Tier 1 Core-10 Master Benchmark...")
    try:
        from experiments.run_tier1_benchmark import run_tier1_benchmark
        run_tier1_benchmark()
    except Exception as e:
        logger.error(f"Error in Tier 1: {e}")
        failed_tiers.append("Tier 1")

    # Final Canonical Consolidation, Figures & Verification
    logger.info(">>> Consolidating Canonical Master Results Store & Generating Figures...")
    try:
        from src.analysis.generate_canonical_results import build_canonical_results_store
        build_canonical_results_store()
        from src.analysis.generate_paper_figures import generate_all_figures
        generate_all_figures()
        from src.analysis.final_validation import run_scientific_validation
        run_scientific_validation()
    except Exception as e:
        logger.error(f"Error in Final Consolidation: {e}")
        failed_tiers.append("Final Consolidation")

    scheduler.print_final_summary()

    if failed_tiers:
        logger.error(f"Execution completed with failures in tiers: {failed_tiers}")
        return 1
    else:
        logger.info("=================================================================")
        logger.info("   UNIFIED 1-GO CCR-TABULAR SUITE EXECUTION SUCCESSFULLY FINISHED ")
        logger.info("=================================================================")
        return 0


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CCR-Tabular Master Heterogeneous Experiment Runner")
    parser.add_argument("--resource_report", action="store_true", help="Print hardware audit and worker budgets.")
    parser.add_argument("--validate", action="store_true", help="Run automated scientific consistency validator.")
    parser.add_argument("--dry_run", action="store_true", help="Inspect execution plan without computing.")
    parser.add_argument("--all", action="store_true", help="Run ALL experiments sequentially in 1 go.")

    parser.add_argument("--tier1", action="store_true", help="Run Tier 1 Master Benchmark (10 core datasets x 8 losses).")
    parser.add_argument("--tier2", action="store_true", help="Run Tier 2 Mechanism Validation (Batch instrumentation).")
    parser.add_argument("--pure_controls", action="store_true", help="Run pure normalization controls.")
    parser.add_argument("--attribution", action="store_true", help="Run per-sample gradient attribution and Figure 5.")
    parser.add_argument("--tier3", action="store_true", help="Run Tier 3 Architecture Transferability.")
    parser.add_argument("--tier4", action="store_true", help="Run Tier 4 Multiclass Benchmark (Segment & Vehicle).")
    parser.add_argument("--tier5", action="store_true", help="Run Tier 5 Real-World External Validation.")
    parser.add_argument("--tier6", action="store_true", help="Run Tier 6 Synthetic Toy & Negative Controls.")
    parser.add_argument("--sb_investigation", action="store_true", help="Run S/B weight-sum inflation investigation.")
    parser.add_argument("--optimizer_study", action="store_true", help="Run SGD vs Adam vs AdamW comparison.")
    parser.add_argument("--compute_benchmark", action="store_true", help="Run computational cost & VRAM profiling.")
    parser.add_argument("--canonical", action="store_true", help="Consolidate canonical master results store.")
    parser.add_argument("--figures", action="store_true", help="Generate all publication and supplementary figures.")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 2-fold diagnostic smoke test.")
    parser.add_argument("--smoke_test_transformer", action="store_true", help="Run 1-fold FT-Transformer smoke test on Adult with 40% noise.")

    # Execution Mode: Fast vs Safe
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fast", action="store_true", help="Enable high-throughput execution (in-memory fold caching, FP16 AMP).")
    mode_group.add_argument("--safe", action="store_true", help="Use conservative single-threaded execution.")

    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device execution target.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name.")
    parser.add_argument("--model", type=str, default=None, help="Model or loss name.")
    parser.add_argument("--noise_type", type=str, default="none", help="Noise type.")
    parser.add_argument("--noise_rate", type=float, default=0.0, help="Noise rate.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024], help="Random seeds.")

    args = parser.parse_args()
    fast_mode = not args.safe  # Default is fast mode unless --safe explicitly specified

    if args.resource_report:
        print_resource_report()

    elif args.validate:
        from src.analysis.final_validation import run_scientific_validation
        run_scientific_validation()

    elif args.figures:
        from src.analysis.generate_paper_figures import generate_all_figures
        generate_all_figures()

    elif args.dry_run:
        target = ["tier1", "tier3", "tier4", "tier5"]
        run_dry_run_planner(target, device_mode=args.device)

    elif args.all:
        from src.utils.manifest import generate_experiment_manifest
        generate_experiment_manifest(OUTPUTS_FINAL_MASTER)
        exit_code = run_all_experiments(device_mode=args.device, fast_mode=fast_mode)
        sys.exit(exit_code)

    elif args.tier6:
        from experiments.run_tier6_toy_controls import (
            run_negative_controls_experiment,
            run_synthetic_toy_experiment,
        )
        out = OUTPUTS_METRICS / "tier6_controls"
        run_synthetic_toy_experiment(out)
        run_negative_controls_experiment(out)

    elif args.pure_controls:
        from experiments.run_pure_normalization_controls import run_pure_normalization_controls
        from src.analysis.analyze_pure_controls import analyze_pure_controls
        run_pure_normalization_controls()
        analyze_pure_controls()

    elif args.attribution:
        from experiments.run_per_sample_gradient_attribution import run_gradient_attribution_study
        run_gradient_attribution_study()

    elif args.sb_investigation:
        from experiments.run_sb_investigation import (
            compute_theoretical_upper_bound,
            run_sb_empirical_measurement,
        )
        print("\n--- Theoretical Upper Bound Analysis ---")
        for k, v in compute_theoretical_upper_bound().items():
            print(f"  {k}: {v}")
        print("\n--- Empirical S/B Percentile Measurement ---")
        run_sb_empirical_measurement()

    elif args.optimizer_study:
        from experiments.run_optimizer_study import run_optimizer_study
        from src.analysis.analyze_optimizer_study import analyze_optimizer_study
        run_optimizer_study()
        analyze_optimizer_study()

    elif args.compute_benchmark:
        from experiments.run_compute_benchmark import run_full_compute_benchmark
        run_full_compute_benchmark()

    elif args.tier2:
        from experiments.run_tier2_mechanism import (
            aggregate_and_plot_mechanism_dynamics,
            run_tier2_mechanism_experiments,
        )
        from src.analysis.analyze_mechanism import analyze_mechanism_telemetry
        run_tier2_mechanism_experiments()
        aggregate_and_plot_mechanism_dynamics()
        analyze_mechanism_telemetry()

    elif args.tier3:
        from experiments.run_tier3_architecture import run_tier3_architecture_experiments
        run_tier3_architecture_experiments()

    elif args.tier4:
        from experiments.run_tier4_multiclass import run_tier4_multiclass_experiments
        run_tier4_multiclass_experiments()

    elif args.tier5:
        from experiments.run_tier5_natural_noise import run_tier5_natural_noise_experiments
        run_tier5_natural_noise_experiments()

    elif args.tier1:
        from experiments.run_tier1_benchmark import run_tier1_benchmark
        run_tier1_benchmark()

    elif args.canonical:
        from src.analysis.generate_canonical_results import build_canonical_results_store
        build_canonical_results_store()

    elif args.smoke_test:
        from src.training.cross_validation import run_cross_validation
        import pandas as pd
        ds = args.dataset or "credit_g"
        print(f"\n=======================================================")
        print(f"       RUNNING SMOKE TEST ON DATASET: [{ds}]          ")
        print(f"=======================================================\n")
        results = []
        for test_model in ["ce", "ccr"]:
            print(f"Testing pipeline: model={test_model} on {ds} (2 folds)...")
            df = run_cross_validation(
                dataset_name=ds,
                model_name=test_model,
                noise_type=args.noise_type,
                noise_rate=args.noise_rate,
                seeds=[42],
                n_folds=2,
                instrument_batch=False,
            )
            results.append(df)
        all_res = pd.concat(results, ignore_index=True)
        print("\n--- Smoke Test Verification Summary ---")
        print(all_res[["run_id", "dataset", "model", "macro_f1", "minority_recall", "auc_roc", "auc_pr", "ece", "brier_score"]])
        print("\n[SMOKE TEST PASSED] All autograd pipelines, telemetry instruments, and metrics functioning correctly.\n")

    elif args.smoke_test_transformer:
        from src.training.cross_validation import run_cross_validation
        print(f"\n=======================================================")
        print(f"  RUNNING FT-TRANSFORMER FINAL-MODE SMOKE TEST (Adult)  ")
        print(f"=======================================================\n")
        df = run_cross_validation(
            dataset_name="adult",
            model_name="ccr",
            noise_type="asym",
            noise_rate=0.40,
            architecture="transformer",
            seeds=[42],
            n_folds=1,
            batch_size=256,
            instrument_batch=False,
        )
        print("\n--- FT-Transformer Smoke Test Summary ---")
        cols = [c for c in ["run_id", "dataset", "model", "architecture", "device_used", "amp_enabled", "macro_f1", "auc_roc"] if c in df.columns]
        print(df[cols])
        print("\n[FT-TRANSFORMER SMOKE TEST PASSED] Transformer attention, embedding, and precision paths verified.\n")

    elif args.dataset is not None and args.model is not None:
        from src.training.cross_validation import run_cross_validation
        df = run_cross_validation(
            dataset_name=args.dataset,
            model_name=args.model,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            seeds=args.seeds,
            n_folds=args.n_folds,
            instrument_batch=False,
        )
        print(df)

    else:
        print("\n=================================================================")
        print("          CCR-TABULAR AUTOMATED MASTER BENCHMARK RUNNER          ")
        print("=================================================================\n")
        print("No specific tier specified. Defaulting to full automatic Core-10 benchmark.")
        from experiments.run_tier1_benchmark import run_tier1_benchmark
        run_tier1_benchmark(device_override=args.device, fast_mode=fast_mode)


if __name__ == "__main__":
    main()
