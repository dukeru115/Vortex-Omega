"""
NFCS Performance Benchmarking Suite
===================================

Comprehensive benchmarking tools for measuring and optimizing NFCS performance
across different components and configurations.

Author: Team Ω (Omega)  
Date: September 13, 2025
Version: 2.4.3
"""

import asyncio
import time
import psutil
import gc
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Tuple, Optional
import logging
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns

# Import optimized solvers
from ..core.cgl_solver_optimized import OptimizedCGLSolver
from ..core.kuramoto_solver_optimized import OptimizedKuramotoSolver
from ..core.state import CGLConfig, KuramotoConfig


@dataclass 
class BenchmarkResult:
    """Individual benchmark result"""
    name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    iterations_per_second: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def add_result(self, result: BenchmarkResult):
        """Add benchmark result to suite"""
        self.results.append(result)
    
    def finish(self):
        """Mark suite as finished"""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark suite summary"""
        if not self.results:
            return {"message": "No benchmark results"}
        
        total_time = sum(r.execution_time for r in self.results if r.error is None)
        avg_memory = np.mean([r.memory_usage_mb for r in self.results if r.error is None])
        avg_cpu = np.mean([r.cpu_percent for r in self.results if r.error is None])
        
        success_count = sum(1 for r in self.results if r.error is None)
        error_count = len(self.results) - success_count
        
        return {
            'suite_name': self.name,
            'total_benchmarks': len(self.results),
            'successful': success_count,
            'failed': error_count,
            'total_execution_time': total_time,
            'average_memory_usage_mb': avg_memory,
            'average_cpu_percent': avg_cpu,
            'suite_duration': (self.end_time or time.time()) - self.start_time,
            'fastest_benchmark': min(self.results, key=lambda x: x.execution_time).name if self.results else None,
            'slowest_benchmark': max(self.results, key=lambda x: x.execution_time).name if self.results else None
        }


class PerformanceBenchmark:
    """Advanced performance benchmarking framework"""
    
    def __init__(self, warmup_iterations: int = 3, measurement_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.logger = logging.getLogger(f"{__name__}.PerformanceBenchmark")
        
        # System baseline
        self.baseline_memory = self._get_memory_usage()
        self.baseline_cpu = psutil.cpu_percent(interval=1.0)
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    @contextmanager
    def _measure_resources(self):
        """Context manager for resource measurement"""
        # Force garbage collection before measurement
        gc.collect()
        
        start_memory = self._get_memory_usage()
        start_cpu_time = time.process_time()
        cpu_percent_start = psutil.cpu_percent()
        
        yield
        
        end_memory = self._get_memory_usage()
        end_cpu_time = time.process_time()
        cpu_percent_end = psutil.cpu_percent()
        
        # Store results in context
        self._last_memory_delta = end_memory - start_memory
        self._last_cpu_time = end_cpu_time - start_cpu_time
        self._last_cpu_percent = (cpu_percent_start + cpu_percent_end) / 2
    
    def benchmark_function(self, func: Callable, *args, name: Optional[str] = None, **kwargs) -> BenchmarkResult:
        """Benchmark a single function"""
        func_name = name or f"{func.__module__}.{func.__name__}"
        
        try:
            # Warmup phase
            for _ in range(self.warmup_iterations):
                func(*args, **kwargs)
            
            # Measurement phase
            times = []
            
            for i in range(self.measurement_iterations):
                with self._measure_resources():
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                
                # Use measurements from last iteration
                if i == self.measurement_iterations - 1:
                    memory_usage = self._last_memory_delta
                    cpu_percent = self._last_cpu_percent
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            return BenchmarkResult(
                name=func_name,
                execution_time=avg_time,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                iterations_per_second=1.0 / avg_time if avg_time > 0 else 0,
                additional_metrics={
                    'time_std': std_time,
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'all_times': times
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name=func_name,
                execution_time=0.0,
                memory_usage_mb=0.0,
                cpu_percent=0.0,
                error=str(e)
            )
    
    def benchmark_cgl_solver(self, grid_sizes: List[Tuple[int, int]] = None) -> BenchmarkSuite:
        """Benchmark CGL solver with different grid sizes"""
        if grid_sizes is None:
            grid_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
        
        suite = BenchmarkSuite("CGL Solver Performance")
        
        for nx, ny in grid_sizes:
            self.logger.info(f"Benchmarking CGL solver with grid size {nx}x{ny}")
            
            # Setup configuration
            config = CGLConfig(
                grid_size=(nx, ny),
                spatial_extent=(10.0, 10.0),
                time_step=0.01,
                c1=1.0,
                c3=1.0
            )
            
            # Create solver
            solver = OptimizedCGLSolver(config)
            
            # Create initial field
            phi = np.random.random((nx, ny)) + 1j * np.random.random((nx, ny))
            phi = phi.astype(np.complex128)
            
            # Benchmark single step
            def single_step():
                return solver.step(phi)
            
            result = self.benchmark_function(
                single_step, 
                name=f"CGL_step_{nx}x{ny}"
            )
            suite.add_result(result)
            
            # Benchmark multiple steps evolution
            def evolution_10_steps():
                return solver.evolve(phi, 10)
            
            result = self.benchmark_function(
                evolution_10_steps,
                name=f"CGL_evolve_10_{nx}x{ny}"
            )
            suite.add_result(result)
            
            # Get solver performance report
            solver_stats = solver.get_performance_report()
            result.additional_metrics.update({
                'solver_stats': solver_stats,
                'grid_points': nx * ny,
                'memory_footprint_mb': solver_stats.get('memory_footprint_mb', 0)
            })
        
        suite.finish()
        return suite
    
    def benchmark_kuramoto_solver(self, oscillator_counts: List[int] = None) -> BenchmarkSuite:
        """Benchmark Kuramoto solver with different network sizes"""
        if oscillator_counts is None:
            oscillator_counts = [10, 50, 100, 500, 1000]
        
        suite = BenchmarkSuite("Kuramoto Solver Performance")
        
        for n_osc in oscillator_counts:
            self.logger.info(f"Benchmarking Kuramoto solver with {n_osc} oscillators")
            
            # Setup configuration
            config = KuramotoConfig(
                n_oscillators=n_osc,
                coupling_strength=1.0,
                time_step=0.01
            )
            
            # Create solver
            solver = OptimizedKuramotoSolver(config)
            
            # Create initial phases
            phases = np.random.uniform(0, 2*np.pi, n_osc)
            
            # Benchmark single step
            def single_step():
                return solver.step(phases)
            
            result = self.benchmark_function(
                single_step,
                name=f"Kuramoto_step_{n_osc}"
            )
            suite.add_result(result)
            
            # Benchmark evolution
            def evolution_100_steps():
                return solver.evolve(phases, 100)
            
            result = self.benchmark_function(
                evolution_100_steps,
                name=f"Kuramoto_evolve_100_{n_osc}"
            )
            suite.add_result(result)
            
            # Get solver performance report
            solver_stats = solver.get_performance_report()
            result.additional_metrics.update({
                'solver_stats': solver_stats,
                'oscillators': n_osc,
                'memory_footprint_mb': solver_stats.get('memory_footprint_mb', 0)
            })
        
        suite.finish()
        return suite
    
    def benchmark_memory_scaling(self) -> BenchmarkSuite:
        """Benchmark memory scaling characteristics"""
        suite = BenchmarkSuite("Memory Scaling Analysis")
        
        # Test different data sizes
        data_sizes = [1000, 10000, 100000, 1000000]
        
        for size in data_sizes:
            # Test numpy array allocation
            def allocate_array():
                return np.random.random((size, 100))
            
            result = self.benchmark_function(
                allocate_array,
                name=f"numpy_alloc_{size}"
            )
            result.additional_metrics['array_size'] = size
            suite.add_result(result)
            
            # Test complex array operations
            arr = np.random.random((size, 10)) + 1j * np.random.random((size, 10))
            
            def complex_operations():
                return np.fft.fft(arr, axis=0)
            
            result = self.benchmark_function(
                complex_operations,
                name=f"fft_complex_{size}"
            )
            result.additional_metrics['array_size'] = size
            suite.add_result(result)
        
        suite.finish()
        return suite
    
    def compare_optimized_vs_standard(self) -> BenchmarkSuite:
        """Compare optimized vs standard implementations"""
        suite = BenchmarkSuite("Optimized vs Standard Comparison")
        
        # This would compare against standard implementations
        # For now, just benchmark the optimized versions
        
        # CGL comparison
        config = CGLConfig(grid_size=(64, 64), time_step=0.01)
        solver = OptimizedCGLSolver(config)
        phi = np.random.random((64, 64)) + 1j * np.random.random((64, 64))
        
        result = self.benchmark_function(
            lambda: solver.evolve(phi, 50),
            name="optimized_cgl_evolution"
        )
        suite.add_result(result)
        
        # Kuramoto comparison
        config = KuramotoConfig(n_oscillators=100, time_step=0.01)
        solver = OptimizedKuramotoSolver(config)
        phases = np.random.uniform(0, 2*np.pi, 100)
        
        result = self.benchmark_function(
            lambda: solver.evolve(phases, 200),
            name="optimized_kuramoto_evolution"
        )
        suite.add_result(result)
        
        suite.finish()
        return suite
    
    def run_comprehensive_benchmark(self) -> Dict[str, BenchmarkSuite]:
        """Run comprehensive benchmark suite"""
        self.logger.info("Starting comprehensive NFCS performance benchmark")
        
        benchmarks = {}
        
        # CGL solver benchmarks
        self.logger.info("Benchmarking CGL solvers...")
        benchmarks['cgl'] = self.benchmark_cgl_solver()
        
        # Kuramoto solver benchmarks  
        self.logger.info("Benchmarking Kuramoto solvers...")
        benchmarks['kuramoto'] = self.benchmark_kuramoto_solver()
        
        # Memory scaling
        self.logger.info("Benchmarking memory scaling...")
        benchmarks['memory'] = self.benchmark_memory_scaling()
        
        # Optimization comparison
        self.logger.info("Comparing optimization levels...")
        benchmarks['comparison'] = self.compare_optimized_vs_standard()
        
        self.logger.info("Comprehensive benchmark completed")
        
        return benchmarks
    
    def generate_report(self, benchmarks: Dict[str, BenchmarkSuite], 
                       save_path: Optional[str] = None) -> str:
        """Generate comprehensive benchmark report"""
        report_lines = []
        
        report_lines.append("NFCS Performance Benchmark Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total // (1024**3)}GB RAM")
        report_lines.append("")
        
        for suite_name, suite in benchmarks.items():
            summary = suite.get_summary()
            
            report_lines.append(f"## {summary['suite_name']}")
            report_lines.append(f"- Total benchmarks: {summary['total_benchmarks']}")
            report_lines.append(f"- Successful: {summary['successful']}")
            report_lines.append(f"- Failed: {summary['failed']}")
            report_lines.append(f"- Total execution time: {summary['total_execution_time']:.3f}s")
            report_lines.append(f"- Average memory usage: {summary['average_memory_usage_mb']:.1f}MB")
            report_lines.append(f"- Average CPU usage: {summary['average_cpu_percent']:.1f}%")
            
            if summary['fastest_benchmark']:
                report_lines.append(f"- Fastest: {summary['fastest_benchmark']}")
            if summary['slowest_benchmark']:
                report_lines.append(f"- Slowest: {summary['slowest_benchmark']}")
            
            report_lines.append("")
            
            # Individual results
            report_lines.append("### Individual Results")
            for result in suite.results:
                if result.error is None:
                    report_lines.append(
                        f"- {result.name}: {result.execution_time:.3f}s, "
                        f"{result.memory_usage_mb:.1f}MB, "
                        f"{result.iterations_per_second:.1f} ops/sec"
                    )
                else:
                    report_lines.append(f"- {result.name}: ERROR - {result.error}")
            
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Benchmark report saved to {save_path}")
        
        return report_text
    
    def plot_performance_comparison(self, benchmarks: Dict[str, BenchmarkSuite], 
                                  save_path: Optional[str] = None):
        """Create performance comparison plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('NFCS Performance Benchmark Results', fontsize=16)
            
            # Plot 1: CGL solver scaling
            if 'cgl' in benchmarks:
                cgl_results = [r for r in benchmarks['cgl'].results if 'step_' in r.name and r.error is None]
                if cgl_results:
                    grid_sizes = [r.additional_metrics.get('grid_points', 0) for r in cgl_results]
                    times = [r.execution_time for r in cgl_results]
                    
                    axes[0, 0].loglog(grid_sizes, times, 'bo-', label='CGL Step Time')
                    axes[0, 0].set_xlabel('Grid Points')
                    axes[0, 0].set_ylabel('Execution Time (s)')
                    axes[0, 0].set_title('CGL Solver Scaling')
                    axes[0, 0].grid(True)
            
            # Plot 2: Kuramoto solver scaling
            if 'kuramoto' in benchmarks:
                kuramoto_results = [r for r in benchmarks['kuramoto'].results if 'step_' in r.name and r.error is None]
                if kuramoto_results:
                    oscillators = [r.additional_metrics.get('oscillators', 0) for r in kuramoto_results]
                    times = [r.execution_time for r in kuramoto_results]
                    
                    axes[0, 1].loglog(oscillators, times, 'ro-', label='Kuramoto Step Time')
                    axes[0, 1].set_xlabel('Number of Oscillators')
                    axes[0, 1].set_ylabel('Execution Time (s)')
                    axes[0, 1].set_title('Kuramoto Solver Scaling')
                    axes[0, 1].grid(True)
            
            # Plot 3: Memory usage comparison
            all_results = []
            for suite in benchmarks.values():
                all_results.extend([r for r in suite.results if r.error is None])
            
            if all_results:
                names = [r.name[:20] for r in all_results]  # Truncate names
                memory_usage = [r.memory_usage_mb for r in all_results]
                
                axes[1, 0].bar(range(len(names)), memory_usage)
                axes[1, 0].set_xlabel('Benchmark')
                axes[1, 0].set_ylabel('Memory Usage (MB)')
                axes[1, 0].set_title('Memory Usage by Benchmark')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Performance summary
            if all_results:
                execution_times = [r.execution_time for r in all_results]
                iterations_per_sec = [r.iterations_per_second or 0 for r in all_results]
                
                axes[1, 1].scatter(execution_times, iterations_per_sec, alpha=0.6)
                axes[1, 1].set_xlabel('Execution Time (s)')
                axes[1, 1].set_ylabel('Iterations per Second')
                axes[1, 1].set_title('Performance Overview')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Performance plots saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")


# Convenience function for quick benchmarking
def quick_benchmark() -> Dict[str, Any]:
    """Run a quick performance benchmark"""
    benchmark = PerformanceBenchmark(warmup_iterations=1, measurement_iterations=3)
    
    # Quick CGL test
    config = CGLConfig(grid_size=(64, 64), time_step=0.01)
    solver = OptimizedCGLSolver(config)
    phi = np.random.random((64, 64)) + 1j * np.random.random((64, 64))
    
    cgl_result = benchmark.benchmark_function(
        lambda: solver.step(phi),
        name="quick_cgl_test"
    )
    
    # Quick Kuramoto test
    config = KuramotoConfig(n_oscillators=100, time_step=0.01)
    solver = OptimizedKuramotoSolver(config)
    phases = np.random.uniform(0, 2*np.pi, 100)
    
    kuramoto_result = benchmark.benchmark_function(
        lambda: solver.step(phases),
        name="quick_kuramoto_test"
    )
    
    return {
        'cgl': {
            'execution_time': cgl_result.execution_time,
            'memory_usage_mb': cgl_result.memory_usage_mb,
            'iterations_per_second': cgl_result.iterations_per_second
        },
        'kuramoto': {
            'execution_time': kuramoto_result.execution_time,
            'memory_usage_mb': kuramoto_result.memory_usage_mb,
            'iterations_per_second': kuramoto_result.iterations_per_second
        }
    }