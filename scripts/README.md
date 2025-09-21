# Scripts Overview

Utilities.

- run-tests.sh: Tests (PDF 7.5).
- ci_validation.py: Coherence CI (PDF 7.3).

Updated: Sept 21, 2025.

## 📁 Script Directory

```
scripts/
├── demo_basic_nfcs.py        # 🎯 Basic NFCS demonstration script
├── run_simulation.py         # 🔬 Advanced simulation runner
└── README.md                 # 📄 This documentation file
```

## 🎯 Available Scripts

### 1. **Basic NFCS Demo** (`demo_basic_nfcs.py`)

**Purpose**: Provides a comprehensive demonstration of NFCS capabilities with interactive examples.

**Features**:
- Complete system initialization walkthrough
- Interactive cognitive module demonstrations
- Mathematical core visualization
- Constitutional framework examples
- Performance benchmarking
- Safety protocol demonstrations

**Usage**:
```bash
# Run basic demonstration
python scripts/demo_basic_nfcs.py

# Run with specific mode
python scripts/demo_basic_nfcs.py --mode interactive
python scripts/demo_basic_nfcs.py --mode automated
python scripts/demo_basic_nfcs.py --mode benchmark

# Run specific demonstration modules
python scripts/demo_basic_nfcs.py --demo constitutional
python scripts/demo_basic_nfcs.py --demo mathematical
python scripts/demo_basic_nfcs.py --demo orchestrator

# Help and options
python scripts/demo_basic_nfcs.py --help
```

**Command Line Options**:
```
Options:
  --mode {interactive,automated,benchmark}
                        Demonstration mode (default: interactive)
  --demo {all,constitutional,mathematical,orchestrator,esc}
                        Specific demo to run (default: all)
  --duration SECONDS    Duration for timed demos (default: 30)
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Logging level (default: INFO)
  --output-dir PATH     Directory for output files (default: ./demo_output)
  --config PATH         Custom configuration file
  --no-visualizations   Disable matplotlib visualizations
  --save-results        Save demonstration results to files
```

**Example Interactive Session**:
```python
# Example of what the demo script showcases
"""
🧠 NFCS Basic Demonstration Script
==================================

1. System Initialization
   ✅ Orchestrator startup
   ✅ Cognitive modules initialized  
   ✅ Mathematical core ready
   ✅ Constitutional framework active

2. Constitutional Framework Demo
   📜 Policy creation and management
   ✅ Safety constraint verification
   ⚖️ Governance mechanism demonstration
   🚨 Violation detection and response

3. Mathematical Core Demo  
   🌊 CGL equation evolution
   🔄 Kuramoto synchronization
   📊 Topological defect analysis
   📈 Real-time metrics calculation

4. Cognitive Processing Demo
   🧠 Multi-modal input processing
   🎭 ESC token-level analysis
   💾 Memory system integration
   🔄 Meta-reflection capabilities

5. Performance Benchmarks
   ⏱️ Response time: 45ms (avg)
   💾 Memory usage: 1.2GB peak
   🔄 Coordination frequency: 10.2Hz
   📊 Throughput: 150 ops/sec
"""
```

### 2. **Simulation Runner** (`run_simulation.py`)

**Purpose**: Advanced simulation execution with configurable parameters for research and development.

**Features**:
- Parametric simulation studies
- Batch processing capabilities
- Data collection and analysis
- Visualization generation
- Export capabilities (CSV, JSON, HDF5)
- Multi-scenario comparison

**Usage**:
```bash
# Run default simulation
python scripts/run_simulation.py

# Run with custom configuration
python scripts/run_simulation.py --config simulations/custom_config.yaml

# Run parameter sweep
python scripts/run_simulation.py --sweep-params c1,c2 --sweep-range 0.1,2.0 --sweep-steps 20

# Run batch simulations
python scripts/run_simulation.py --batch simulations/batch_config.json

# Run with specific output format
python scripts/run_simulation.py --output-format hdf5 --output-file results.h5
```

**Command Line Options**:
```
Simulation Options:
  --config PATH               Configuration file (YAML/JSON)
  --duration FLOAT           Simulation duration (default: 10.0)
  --timestep FLOAT           Time step size (default: 0.01)
  --grid-size INT INT        Grid dimensions (default: 128 128)
  
Parameter Sweep:
  --sweep-params NAMES        Parameters to sweep (comma-separated)
  --sweep-range FLOAT FLOAT   Parameter range (min, max)
  --sweep-steps INT          Number of sweep steps (default: 10)
  
Batch Processing:
  --batch PATH               Batch configuration file
  --parallel INT             Parallel workers (default: 4)
  
Output Options:
  --output-dir PATH          Output directory (default: ./simulation_results)
  --output-format {csv,json,hdf5,npz}
                            Output format (default: npz)
  --save-visualizations      Generate and save plots
  --save-animations          Generate animation files
  
Analysis:
  --analyze                  Perform post-simulation analysis
  --compare PATHS            Compare multiple simulation results
  --metrics LIST             Specific metrics to calculate
```

**Example Configuration**:
```yaml
# simulations/example_config.yaml
simulation:
  name: "CGL_Kuramoto_Study"
  description: "Study of CGL-Kuramoto coupling dynamics"
  
parameters:
  cgl:
    c1: 0.5
    c2: 1.0  
    c3: 0.8
    grid_size: [128, 128]
    domain_size: [10.0, 10.0]
  
  kuramoto:
    N: 5
    coupling_strength: 2.0
    natural_frequencies: [0.8, 0.9, 1.0, 1.1, 1.2]
  
execution:
  duration: 20.0
  timestep: 0.01
  save_interval: 0.1
  
analysis:
  calculate_metrics: true
  generate_plots: true
  save_animations: false
  
output:
  directory: "./results/cgl_kuramoto_study"
  format: "hdf5"
  compress: true
```

## 📊 Utility Functions

Both scripts include common utility functions that can be imported for custom scripts:

### System Utilities
```python
from scripts.demo_basic_nfcs import (
    initialize_nfcs_system,
    run_performance_benchmark,
    demonstrate_constitutional_framework,
    visualize_system_state
)

# Initialize NFCS for custom use
orchestrator = await initialize_nfcs_system(config_path="custom_config.yaml")

# Run quick performance check
benchmark_results = await run_performance_benchmark(orchestrator, duration=60)
print(f"Average response time: {benchmark_results['avg_response_time']:.3f}s")
```

### Simulation Utilities
```python
from scripts.run_simulation import (
    load_simulation_config,
    execute_single_simulation,
    analyze_simulation_results,
    generate_comparison_report
)

# Custom simulation execution
config = load_simulation_config("my_simulation.yaml")
results = execute_single_simulation(config, output_dir="./my_results")

# Analysis and reporting
analysis = analyze_simulation_results(results)
report = generate_comparison_report([results1, results2], output="comparison.html")
```

## 🛠️ Development Scripts

### Creating Custom Scripts

**Script Template**:
```python
#!/usr/bin/env python3
"""
Custom NFCS Script Template
===========================

Description: Brief description of script purpose
Author: Your Name
Date: YYYY-MM-DD
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.nfcs_orchestrator import create_orchestrator, create_default_config

async def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description="Custom NFCS Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--output-dir", type=str, default="./output", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize NFCS system
        config = create_default_config()
        if args.config:
            config.update_from_file(args.config)
        
        orchestrator = create_orchestrator(config)
        await orchestrator.start()
        
        # Your custom logic here
        print("🚀 Custom NFCS script running...")
        
        # Example: Process some data
        result = await orchestrator.process_input({
            "text": "Custom script input",
            "metadata": {"script": "custom_script"}
        })
        
        print(f"✅ Processing result: {result}")
        
        # Save results
        import json
        with open(output_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        return 1
    
    finally:
        # Cleanup
        if 'orchestrator' in locals():
            await orchestrator.stop()
    
    print("✅ Custom script completed successfully")
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
```

### Batch Processing Script Template
```python
#!/usr/bin/env python3
"""
Batch Processing Template for NFCS
=================================
"""

import asyncio
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

async def process_batch_item(item_config, orchestrator):
    """Process a single batch item."""
    try:
        result = await orchestrator.process_input(item_config["input"])
        return {
            "item_id": item_config["id"],
            "success": True,
            "result": result,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "item_id": item_config["id"],
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }

async def run_batch_processing(batch_config_path, output_dir):
    """Run batch processing from configuration file."""
    
    # Load batch configuration
    with open(batch_config_path) as f:
        batch_config = json.load(f)
    
    # Initialize NFCS
    orchestrator = create_orchestrator(create_default_config())
    await orchestrator.start()
    
    try:
        # Process items
        tasks = []
        for item in batch_config["items"]:
            task = process_batch_item(item, orchestrator)
            tasks.append(task)
        
        # Execute with controlled concurrency
        semaphore = asyncio.Semaphore(batch_config.get("max_concurrent", 10))
        
        async def limited_process(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[limited_process(task) for task in tasks])
        
        # Save results
        output_path = Path(output_dir) / "batch_results.json"
        with open(output_path, "w") as f:
            json.dump({
                "batch_config": batch_config,
                "results": results,
                "summary": {
                    "total_items": len(results),
                    "successful": sum(1 for r in results if r["success"]),
                    "failed": sum(1 for r in results if not r["success"])
                }
            }, f, indent=2)
        
        print(f"✅ Batch processing complete. Results saved to {output_path}")
        
    finally:
        await orchestrator.stop()
```

## 📋 Script Configuration

### Configuration File Formats

**YAML Configuration**:
```yaml
# config/script_config.yaml
system:
  operational_mode: "automated"
  log_level: "INFO"
  safety_level: 0.8

demonstration:
  duration: 30
  visualizations: true
  save_outputs: true
  output_directory: "./demo_results"

simulation:
  grid_size: [64, 64]
  time_step: 0.01
  evolution_time: 10.0
  
benchmarking:
  iterations: 100
  concurrent_requests: 10
  measure_memory: true
```

**JSON Configuration**:
```json
{
  "system": {
    "operational_mode": "automated",
    "log_level": "INFO",
    "safety_level": 0.8
  },
  "demonstration": {
    "duration": 30,
    "visualizations": true,
    "save_outputs": true,
    "output_directory": "./demo_results"
  },
  "simulation": {
    "grid_size": [64, 64],
    "time_step": 0.01,
    "evolution_time": 10.0
  }
}
```

## 🚀 Quick Start Examples

### Running Basic Demo
```bash
# Interactive demonstration (recommended for first-time users)
python scripts/demo_basic_nfcs.py --mode interactive --save-results

# Automated benchmark run
python scripts/demo_basic_nfcs.py --mode benchmark --duration 60 --output-dir ./benchmarks

# Constitutional framework focus
python scripts/demo_basic_nfcs.py --demo constitutional --log-level DEBUG
```

### Running Simulations
```bash
# Quick simulation with default parameters
python scripts/run_simulation.py --duration 5.0 --save-visualizations

# Parameter sweep study
python scripts/run_simulation.py \
  --sweep-params c1,c2 \
  --sweep-range 0.1,2.0 \
  --sweep-steps 10 \
  --output-format hdf5 \
  --parallel 4

# Custom configuration
python scripts/run_simulation.py \
  --config simulations/research_config.yaml \
  --analyze \
  --save-animations
```

### Custom Script Development
```bash
# Create new script from template
cp scripts/demo_basic_nfcs.py scripts/my_custom_script.py

# Edit for your specific use case
# ... modify script logic ...

# Run custom script
python scripts/my_custom_script.py --config my_config.yaml
```

## 🔧 System Requirements

### Runtime Requirements
- **Python**: 3.8+ with asyncio support
- **Memory**: 2-4 GB for demonstrations, 4-8 GB for simulations
- **CPU**: Dual-core minimum, quad-core recommended for parallel processing
- **Storage**: 500 MB for outputs and temporary files

### Optional Dependencies
```bash
# For enhanced visualizations
pip install matplotlib>=3.7.0 seaborn>=0.12.0

# For data export capabilities
pip install h5py>=3.8.0 pandas>=2.0.0

# For parallel processing
pip install joblib>=1.3.0

# For progress bars and enhanced UI
pip install tqdm>=4.65.0 rich>=13.0.0
```

## 🧪 Testing Scripts

Scripts include built-in testing capabilities:

```bash
# Test script functionality
python scripts/demo_basic_nfcs.py --test --duration 5

# Validate simulation parameters
python scripts/run_simulation.py --validate-config simulations/config.yaml

# Dry run (no actual execution)
python scripts/run_simulation.py --dry-run --config test_config.yaml
```

## 📚 Documentation and Examples

### Script Documentation
Each script includes comprehensive help documentation:
```bash
python scripts/demo_basic_nfcs.py --help
python scripts/run_simulation.py --help
```

### Example Configurations
See the `config/` directory for example configuration files:
- `demo_config.yaml`: Demonstration script configuration
- `simulation_config.yaml`: Simulation runner configuration
- `batch_config.json`: Batch processing configuration

### Output Examples
Scripts generate structured outputs:
```
output_directory/
├── logs/                     # Execution logs
├── visualizations/           # Generated plots and animations
├── data/                     # Raw and processed data files
├── reports/                  # Analysis reports and summaries
└── config_used.yaml         # Configuration actually used
```

## 🤝 Contributing

### Adding New Scripts

**Guidelines for New Scripts**:
1. Follow the template structure provided
2. Include comprehensive argument parsing
3. Implement proper logging and error handling
4. Add configuration file support
5. Include help documentation and examples
6. Implement dry-run and testing modes

**Script Checklist**:
- [ ] Proper shebang and encoding
- [ ] Comprehensive docstring
- [ ] Command-line argument parsing
- [ ] Configuration file support
- [ ] Logging setup
- [ ] Error handling and cleanup
- [ ] Help documentation
- [ ] Testing capabilities

### Script Integration
New scripts should integrate with the NFCS ecosystem:
- Use standard NFCS imports and patterns
- Follow configuration conventions
- Implement proper async/await patterns
- Include performance monitoring
- Support standard output formats

---

## Russian Translation / Русский перевод

# Скрипты и утилиты - Инструменты NFCS

## Обзор

Данная директория содержит служебные скрипты, демонстрационные инструменты и скрипты автоматизации для Системы управления нейронными полями (NFCS). Эти скрипты обеспечивают удобные точки входа для демонстрации системы, выполнения симуляций, обработки данных и рабочих процессов разработки.

**Назначение**: Упростить операции NFCS через автоматизированные скрипты и интерактивные демонстрации.

---

*This README provides comprehensive documentation for NFCS scripts and utilities, enabling efficient system operation, demonstration, and development workflows.*

*Данный README предоставляет исчерпывающую документацию для скриптов и утилит NFCS, обеспечивая эффективные операции системы, демонстрацию и рабочие процессы разработки.*