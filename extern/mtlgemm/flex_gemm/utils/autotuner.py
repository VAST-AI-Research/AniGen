import builtins
from typing import *
import os
import json
import importlib
import pkgutil
import torch
try:
    import triton
except ImportError:
    triton = None
import time
import inspect
from filelock import FileLock
from .. import (
    AUTOSAVE_AUTOTUNE_CACHE,
    AUTOTUNE_CACHE_PATH,
)


_AutotunerBase = triton.runtime.Autotuner if triton is not None else object

class TritonPersistentCacheAutotuner(_AutotunerBase):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
        )

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key = [_args[key] for key in self.keys if key in _args]
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = str(tuple(key))
            if key not in self.cache:
                # prune configs
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__} finished after "
                  f"{self.bench_time:.2f}s; best config selected: {self.best_config};")
        if AUTOSAVE_AUTOTUNE_CACHE and not used_cached_result:
            save_autotune_cache()
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for config in self.prune_configs(kwargs):
            ret.append(self.fn.warmup(
                *args,
                **kwargs,
                **config.all_kwargs(),
            ))
        self.nargs = None
        return ret


def triton_autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton_autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    """

    def decorator(fn):
        return TritonPersistentCacheAutotuner(
            fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
            post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
            use_cuda_graph=use_cuda_graph
        )

    return decorator


class PersistentCacheAutoTuner:
    def __init__(
        self,
        kernel,
        configs=None,
        key=None,
        config_fn=None,
        key_fn=None,
        warmup=3,
        runs=10,
        verbose=False,
    ):
        """
        AutoTuner is a wrapper class for a kernel that automatically tunes the kernel parameters to achieve the best performance.
        
        Args:
            kernel: A callable object that takes in input arguments and returns the output.
            configs: A list of Config objects that define the possible kernel parameters and their values.
            key: A list of argument names that retune the kernel on change.
            config_fn: A function that takes in the input arguments and returns configs to be used for autotuning.
            key_fn: A function that takes in the input arguments and returns the key used to cache the tuning results.
                    Once the key changes, the autotuning will be rerun.
            warmup: The number of warmup runs to discard before measuring the execution time.
            runs: The number of runs to measure the execution time.
            verbose: Whether to print the autotuning results.
        """
        assert config_fn or configs, "Either configs or config_fn must be provided"
        assert key_fn or key, "Either key or key_fn must be provided"
        self.kernel = kernel
        self.configs = configs
        self.key = key
        self.config_fn = config_fn
        self.key_fn = key_fn
        self.warmup = warmup
        self.runs = runs
        self.verbose = verbose or os.getenv('FLEX_GEMM_AUTOTUNER_VERBOSE', '0') == '1'
        self.kernel_arg_names = inspect.getfullargspec(kernel).args
        self.cache = {}
        
    def _args_to_kwargs(self, args, kwargs):
        # Convert args to kwargs
        arg_names = self.kernel_arg_names
        arg_dict = dict(zip(arg_names, args))
        arg_dict.update(kwargs)
        return arg_dict
    
    def __call__(self, *args, **kwargs):
        arg_dict = self._args_to_kwargs(args, kwargs)
        
        # Determine key
        key = self.key_fn(*args, **kwargs) if self.key_fn else tuple(arg_dict[k] for k in self.key)
        key = str(key)
        
        # If key changes, rerun autotune
        used_cached_result = True
        if key not in self.cache:
            used_cached_result = False
            if self.verbose:
                print(f"Running autotuning for {self.kernel.__name__} with key {key}")
            configs = self.configs if self.configs else self.config_fn(*args, **kwargs)
            if self.verbose:
                print(f"Configs: {configs}")
            best_config = self._benchmark(args, kwargs, configs)
            if self.verbose:
                print(f"Best config for {self.kernel.__name__} with key {key}: {best_config}")
            self.cache[key] = best_config
            
        if AUTOSAVE_AUTOTUNE_CACHE and not used_cached_result:
            save_autotune_cache()
        
        # Run the kernel with the best config
        return self.kernel(*args, **kwargs, **self.cache[key])
    
    def _benchmark(self, args, kwargs, configs):
        best_time = float('inf')
        best_config = None
        
        if len(configs) == 1:
            best_config = configs[0]
        else:
            for config in configs:
                # Run the kernel and measure execution time
                for _ in range(self.warmup):
                    self.kernel(*args, **kwargs, **config)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(self.runs):
                    self.kernel(*args, **kwargs, **config)
                torch.cuda.synchronize()
                elapsed = (time.time() - start) / self.runs
                if self.verbose:
                    print(f"Config {config}: {elapsed} seconds")
                # Update the best config if the execution time is better
                if elapsed < best_time:
                    best_time = elapsed
                    best_config = config
        
        return best_config
    

def autotune(
    configs=None,
    key=None,
    config_fn=None,
    key_fn=None,
    warmup=3,
    runs=10,
    verbose=False
):
    def decorator(kernel):
        return PersistentCacheAutoTuner(kernel, configs, key, config_fn, key_fn, warmup, runs, verbose)
    return decorator


def walk_package(package_name, fn):
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        print(f"Package {package_name} not found.")
        return

    if not hasattr(package, '__path__'):
        print(f"{package_name} is not a package.")
        return

    for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{package_name}.{module_name}"
        if is_pkg:
            walk_package(full_module_name, fn)
        else:
            fn(full_module_name)
            

def _get_device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    import platform
    if platform.system() == 'Darwin':
        return "apple_silicon"
    return "cpu"

def get_autotune_cache():
    cache = {}
    device_name = _get_device_name()
    if device_name not in cache:
        cache[device_name] = {}

    def save_cache(full_module_name):
        module = importlib.import_module(full_module_name)
        for attr_name, attr in module.__dict__.items():
            cache_key = f"{full_module_name}.{attr_name}"
            if isinstance(attr, PersistentCacheAutoTuner):
                cache[device_name][cache_key] = attr.cache
            elif isinstance(attr, TritonPersistentCacheAutotuner):
                cache[device_name][cache_key] = {k: v.__dict__ for k, v in attr.cache.items()}

    walk_package('flex_gemm', save_cache)

    return cache


def save_autotune_cache(path=None):
    path = path or AUTOTUNE_CACHE_PATH
    lock_path = path + ".lock"

    with FileLock(lock_path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}
        # Merge existing cache with new cache
        cache.update(get_autotune_cache())

        tmp_path = path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(cache, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)


def load_autotune_cache(path_or_cache=None):
    cache = None

    # Preserve path-based loading, but allow callers to provide a preloaded cache object.
    if path_or_cache is None or isinstance(path_or_cache, (str, os.PathLike)):
        path = path_or_cache or AUTOTUNE_CACHE_PATH
        lock_path = path + ".lock"

        if not os.path.exists(path):
            return

        with FileLock(lock_path):
            with open(path, 'r') as f:
                cache = json.load(f)
    elif isinstance(path_or_cache, Mapping):
        cache = path_or_cache
    else:
        raise TypeError("load_autotune_cache expects a path or a mapping")

    if cache is None:
        return

    device_name = _get_device_name()
    if device_name not in cache and "*" not in cache:
        return
    if "*" in cache and device_name not in cache:
        device_name = "*"

    def load_cache(full_module_name):
        module = importlib.import_module(full_module_name)
        for attr_name, attr in module.__dict__.items():
            cache_key = f"{full_module_name}.{attr_name}"
            if isinstance(attr, PersistentCacheAutoTuner):
                if cache_key in cache[device_name]:
                    attr.cache = cache[device_name][cache_key]
            elif isinstance(attr, TritonPersistentCacheAutotuner):
                if cache_key in cache[device_name]:
                    for k, v in cache[device_name][cache_key].items():
                        attr.cache[k] = triton.runtime.Config(None)
                        attr.cache[k].__dict__.update(v)

    walk_package('flex_gemm', load_cache)
