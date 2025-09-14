#!/usr/bin/env python3
"""
Symbolic-Neural Bridge Demonstration
===================================

Interactive demonstration of the Symbolic-Neural Bridge implementation
showing S ↔ φ transformations and NFCS integration.

This demo showcases:
1. Symbolic clause extraction and canonicalization  
2. Neural field embedding (S → φ)
3. Field-to-symbolic extraction (φ → S)
4. Consistency verification
5. Real-time Hallucination Number monitoring
6. Constitutional oversight integration

Usage:
    python demo_symbolic_bridge.py

Author: Team Omega (GenSpark AI Implementation)
License: CC BY-NC 4.0
Date: 2025-09-14
"""

import asyncio
import sys
import os
import logging
from typing import Dict, List, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import NFCS components
try:
    from src.modules.symbolic.neural_bridge import SymbolicNeuralBridge
    from src.modules.symbolic.models import SymClause, ClauseType
    from src.modules.integration.symbolic_nfcs_integration import SymbolicNFCSIntegration
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please ensure you're running from the project root directory")
    sys.exit(1)


class SymbolicBridgeDemo:
    """Interactive demonstration of Symbolic-Neural Bridge"""
    
    def __init__(self):
        """Initialize the demonstration"""
        self.bridge = SymbolicNeuralBridge(
            field_dims=(64, 64),
            max_symbols=128
        )
        self.integration = SymbolicNFCSIntegration({
            'field_dims': (64, 64),
            'max_symbols': 128
        })
        
        # Demo scenarios
        self.test_scenarios = [
            {
                'name': 'Physics Equation',
                'input': 'Energy E equals mass m times the speed of light c squared: E = mc²',
                'domain': 'physics',
                'expected_ha': 'low'
            },
            {
                'name': 'Mathematical Statement',
                'input': 'The derivative of x squared equals 2x, and the integral gives x cubed over 3',
                'domain': 'mathematics', 
                'expected_ha': 'low'
            },
            {
                'name': 'Logical Contradiction',
                'input': 'All birds can fly, but penguins are birds that cannot fly',
                'domain': 'logic',
                'expected_ha': 'medium'
            },
            {
                'name': 'Nonsensical Statement',
                'input': 'Purple elephants multiply infinity with yesterday to produce quantum cheese',
                'domain': 'nonsense',
                'expected_ha': 'high'
            },
            {
                'name': 'Partial Information',
                'input': 'Force equals mass times something, maybe acceleration or velocity',
                'domain': 'physics',
                'expected_ha': 'medium'
            }
        ]
    
    async def run_demo(self):
        """Run the complete demonstration"""
        print("🚀 Symbolic-Neural Bridge Demonstration")
        print("=" * 50)
        print()
        
        # Run basic bridge tests
        await self.demo_basic_bridge_operations()
        
        # Run integration tests  
        await self.demo_integration_pipeline()
        
        # Run scenario analysis
        await self.demo_scenario_analysis()
        
        # Show visualizations
        self.demo_visualizations()
        
        print("\n✅ Demonstration completed successfully!")
        print("Check generated plots for visual analysis.")
    
    async def demo_basic_bridge_operations(self):
        """Demonstrate basic S ↔ φ operations"""
        print("🔗 Basic Symbolic-Neural Bridge Operations")
        print("-" * 40)
        
        # Create test symbolic clauses
        test_clauses = [
            SymClause(cid="physics_eq1", ctype=ClauseType.EQUATION, meta={'domain': 'physics'}),
            SymClause(cid="math_fact1", ctype=ClauseType.FACT, meta={'domain': 'mathematics'}),
            SymClause(cid="constraint1", ctype=ClauseType.CONSTRAINT, meta={'domain': 'general'})
        ]
        
        print(f"📝 Created {len(test_clauses)} test symbolic clauses")
        
        # Test S → φ (Fieldization)
        print("\n🔄 Testing S → φ transformation (Fieldization)...")
        field_state = await self.bridge.fieldize(test_clauses)
        
        field_energy = float(torch.sum(torch.abs(field_state) ** 2))
        field_complexity = float(torch.std(torch.abs(field_state)))
        
        print(f"   ✅ Neural field generated: {field_state.shape}")
        print(f"   📊 Field energy: {field_energy:.4f}")
        print(f"   📈 Field complexity: {field_complexity:.4f}")
        
        # Test φ → S (Symbolization)
        print("\n🔄 Testing φ → S transformation (Symbolization)...")
        extracted_symbols = await self.bridge.symbolize(field_state)
        
        print(f"   ✅ Extracted {len(extracted_symbols)} symbolic elements")
        for i, symbol in enumerate(extracted_symbols[:3]):
            amp = symbol.meta.get('amplitude', 0)
            print(f"   📋 Symbol {i+1}: {symbol.cid} (amplitude: {amp:.3f})")
        
        # Test consistency verification
        print("\n🔍 Testing consistency verification...")
        consistency_result = await self.bridge.verify_consistency(test_clauses, field_state)
        
        consistency_score = consistency_result['consistency_score']
        field_mse = consistency_result['field_mse']
        
        print(f"   ✅ Consistency score: {consistency_score:.3f}")
        print(f"   📏 Field MSE: {field_mse:.6f}")
        
        # Show bridge metrics
        metrics = self.bridge.get_metrics()
        print(f"\n📊 Bridge Performance Metrics:")
        print(f"   ⏱️  Symbolization time: {metrics['symbolization_time']:.4f}s")
        print(f"   ⏱️  Fieldization time: {metrics['fieldization_time']:.4f}s")
        print(f"   🔄 Total transformations: {metrics['total_transformations']}")
    
    async def demo_integration_pipeline(self):
        """Demonstrate complete NFCS integration pipeline"""
        print("\n\n🔗 Complete NFCS Integration Pipeline")
        print("-" * 40)
        
        test_input = """
        The fundamental equation of physics states that energy E equals 
        mass m multiplied by the speed of light c squared. This relationship 
        demonstrates the equivalence of mass and energy in special relativity.
        """
        
        print(f"📝 Processing input: {test_input[:60]}...")
        
        # Process through complete pipeline
        result = await self.integration.process_input(
            test_input,
            context={'domain': 'physics', 'confidence_required': 0.8}
        )
        
        # Display results
        print(f"\n📊 Integration Results:")
        print(f"   ✅ Success: {result['success']}")
        print(f"   🧠 Hallucination Number: {result['hallucination_number']:.3f}")
        print(f"   🌊 Coherence Measure: {result['coherence_measure']:.3f}")
        print(f"   ⚡ Field Energy: {result['field_energy']:.3f}")
        print(f"   ⏱️  Processing Time: {result['processing_time_ms']:.1f}ms")
        print(f"   🔢 Active Symbols: {result['active_symbols']}")
        print(f"   🚨 Emergency Active: {result['emergency_active']}")
        
        # Show component results
        if 'symbolic_report' in result:
            symbolic = result['symbolic_report']
            print(f"\n🔤 Symbolic Analysis:")
            print(f"   📋 Fields processed: {symbolic.get('fields_count', 0)}")
            print(f"   📝 Clauses analyzed: {symbolic.get('clauses_count', 0)}")
            print(f"   ✅ Answer confidence: {symbolic.get('answer_conf', 0):.3f}")
        
        if 'constitutional_result' in result:
            const = result['constitutional_result']
            print(f"\n⚖️  Constitutional Check:")
            print(f"   📊 Status: {const.get('status', 'unknown')}")
            if 'reason' in const:
                print(f"   📝 Reason: {const['reason'][:50]}...")
    
    async def demo_scenario_analysis(self):
        """Analyze different input scenarios"""
        print("\n\n🎯 Scenario Analysis")
        print("-" * 40)
        
        results = []
        
        for scenario in self.test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"   📝 Input: {scenario['input'][:50]}...")
            print(f"   🎯 Expected Ha: {scenario['expected_ha']}")
            
            # Process scenario
            result = await self.integration.process_input(
                scenario['input'],
                context={'domain': scenario['domain']}
            )
            
            ha_value = result['hallucination_number']
            coherence = result['coherence_measure']
            processing_time = result['processing_time_ms']
            
            print(f"   📊 Actual Ha: {ha_value:.3f}")
            print(f"   🌊 Coherence: {coherence:.3f}")
            print(f"   ⏱️  Time: {processing_time:.1f}ms")
            
            # Verify expectations
            expected_ranges = {
                'low': (0, 1.5),
                'medium': (1.0, 4.0), 
                'high': (3.0, 10.0)
            }
            
            expected_range = expected_ranges[scenario['expected_ha']]
            in_range = expected_range[0] <= ha_value <= expected_range[1]
            
            print(f"   ✅ Expected range met: {in_range}")
            
            results.append({
                'name': scenario['name'],
                'ha_value': ha_value,
                'coherence': coherence,
                'time': processing_time,
                'expected_ha': scenario['expected_ha']
            })
        
        # Summary statistics
        ha_values = [r['ha_value'] for r in results]
        coherence_values = [r['coherence'] for r in results]
        
        print(f"\n📈 Summary Statistics:")
        print(f"   📊 Ha range: {min(ha_values):.3f} - {max(ha_values):.3f}")
        print(f"   📊 Ha average: {np.mean(ha_values):.3f}")
        print(f"   🌊 Coherence range: {min(coherence_values):.3f} - {max(coherence_values):.3f}")
        print(f"   🌊 Coherence average: {np.mean(coherence_values):.3f}")
        
        self.scenario_results = results
    
    def demo_visualizations(self):
        """Create visualizations of the results"""
        print("\n\n📊 Generating Visualizations")
        print("-" * 40)
        
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Symbolic-Neural Bridge Demonstration Results', fontsize=16)
            
            # Plot 1: Hallucination Numbers by scenario
            if hasattr(self, 'scenario_results'):
                names = [r['name'] for r in self.scenario_results]
                ha_values = [r['ha_value'] for r in self.scenario_results]
                
                bars = ax1.bar(range(len(names)), ha_values, 
                              color=['green', 'blue', 'orange', 'red', 'purple'])
                ax1.set_title('Hallucination Number by Scenario')
                ax1.set_ylabel('Ha Value')
                ax1.set_xticks(range(len(names)))
                ax1.set_xticklabels(names, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'{height:.2f}', ha='center', va='bottom')
            
            # Plot 2: Field state visualization (if available)
            if hasattr(self.bridge, 'field_state') and self.bridge.field_state is not None:
                field = self.bridge.field_state.detach().cpu().numpy()
                if np.iscomplexobj(field):
                    field_vis = np.abs(field)
                else:
                    field_vis = field
                
                im2 = ax2.imshow(field_vis, cmap='viridis', interpolation='nearest')
                ax2.set_title('Neural Field State (Amplitude)')
                ax2.set_xlabel('X coordinate')
                ax2.set_ylabel('Y coordinate')
                plt.colorbar(im2, ax=ax2)
            
            # Plot 3: Coherence vs Hallucination scatter
            if hasattr(self, 'scenario_results'):
                coherence_vals = [r['coherence'] for r in self.scenario_results]
                colors = ['green', 'blue', 'orange', 'red', 'purple']
                
                for i, (ha, coh, name) in enumerate(zip(ha_values, coherence_vals, names)):
                    ax3.scatter(ha, coh, c=colors[i], s=100, label=name, alpha=0.7)
                
                ax3.set_xlabel('Hallucination Number')
                ax3.set_ylabel('Coherence Measure')
                ax3.set_title('Coherence vs Hallucination')
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Processing time analysis
            if hasattr(self, 'scenario_results'):
                times = [r['time'] for r in self.scenario_results]
                
                ax4.plot(range(len(names)), times, 'o-', linewidth=2, markersize=8)
                ax4.set_title('Processing Time by Scenario')
                ax4.set_ylabel('Time (ms)')
                ax4.set_xlabel('Scenario')
                ax4.set_xticks(range(len(names)))
                ax4.set_xticklabels(names, rotation=45, ha='right')
                ax4.grid(True, alpha=0.3)
                
                # Add time labels
                for i, time_val in enumerate(times):
                    ax4.annotate(f'{time_val:.1f}ms', 
                               (i, time_val), 
                               textcoords="offset points",
                               xytext=(0,10), 
                               ha='center')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save plots
            plots_dir = project_root / 'demo_plots'
            plots_dir.mkdir(exist_ok=True)
            
            plot_file = plots_dir / 'symbolic_bridge_demo.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            
            print(f"   📊 Visualization saved: {plot_file}")
            
            # Show plot if in interactive mode
            if hasattr(sys, 'ps1'):  # Interactive mode
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            print("   ❌ Visualization generation failed")
    
    async def interactive_mode(self):
        """Run in interactive mode for custom inputs"""
        print("\n\n🎮 Interactive Mode")
        print("-" * 40)
        print("Enter your text for symbolic-neural analysis (or 'quit' to exit):")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                print("🔄 Processing...")
                
                # Process input
                result = await self.integration.process_input(user_input)
                
                print(f"\n📊 Results:")
                print(f"   🧠 Ha: {result['hallucination_number']:.3f}")
                print(f"   🌊 Coherence: {result['coherence_measure']:.3f}")
                print(f"   ⏱️  Time: {result['processing_time_ms']:.1f}ms")
                print(f"   🔢 Symbols: {result['active_symbols']}")
                
                if result['hallucination_number'] > 2.0:
                    print("   ⚠️  High hallucination risk detected!")
                elif result['hallucination_number'] < 1.0:
                    print("   ✅ Low hallucination risk - good coherence!")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n👋 Interactive mode ended.")


async def main():
    """Main demonstration function"""
    print("Initializing Symbolic-Neural Bridge Demo...")
    
    try:
        demo = SymbolicBridgeDemo()
        
        # Check for interactive mode
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            await demo.interactive_mode()
        else:
            await demo.run_demo()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)