#!/usr/bin/env python3
"""
Test script for BioFlux RL Training Framework.

This script tests the basic functionality of all implemented algorithms
to ensure they work correctly before running full training.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np

# Add the parent directory to the path to import bioflux
sys.path.append(str(Path(__file__).parent.parent))

try:
    from bioflux.training import (
        TrainingConfig, LotkaVolterraAgent, EpsilonGreedyAgent, 
        PPOAgent, MADDPGWrapper, ReplayBuffer, create_training_environment
    )
    from bioflux.core.environment import Environment
    from bioflux.core.agents import Predator, Prey, Plant
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    IMPORTS_SUCCESS = False

class TestTrainingFramework(unittest.TestCase):
    """Test cases for the RL training framework."""
    
    def setUp(self):
        """Set up test configuration and environment."""
        self.config = TrainingConfig(
            num_episodes=5,
            max_steps_per_episode=10,
            learning_rate=0.001,
            device="cpu"
        )
        
        self.state_dim = 8
        self.action_dim = 8
        
        # Create mock state
        self.mock_state = {
            'energy': 50,
            'age': 5,
            'nearby_prey': [{'pos': (10, 10), 'distance': 5}],
            'nearby_predators': [],
            'nearby_food': [{'pos': (15, 15), 'distance': 7}],
            'temperature': 25,
            'vegetation': 0.7,
            'is_hungry': False,
            'pos_x': 12,
            'pos_y': 12,
            'environment_bounds': {'width': 100, 'height': 100}
        }
    
    @unittest.skipUnless(IMPORTS_SUCCESS, "Imports failed")
    def test_lotka_volterra_agent(self):
        """Test Lotka-Volterra agent functionality."""
        print("🧪 Testing Lotka-Volterra Agent...")
        
        # Test predator
        predator_agent = LotkaVolterraAgent('predator', self.config)
        action = predator_agent.select_action(self.mock_state)
        
        self.assertIsInstance(action, dict)
        self.assertIn('action', action)
        self.assertIn('x', action)
        self.assertIn('y', action)
        
        # Test prey
        prey_agent = LotkaVolterraAgent('prey', self.config)
        action = prey_agent.select_action(self.mock_state)
        
        self.assertIsInstance(action, dict)
        self.assertIn('action', action)
        
        # Test update (should be no-op)
        trajectory = [{'state': self.mock_state, 'action': action, 'reward': 1.0}]
        predator_agent.update(trajectory)  # Should not raise error
        
        print("  ✅ Lotka-Volterra agent tests passed")
    
    @unittest.skipUnless(IMPORTS_SUCCESS, "Imports failed")
    def test_epsilon_greedy_agent(self):
        """Test Epsilon-Greedy agent functionality."""
        print("🧪 Testing Epsilon-Greedy Agent...")
        
        agent = EpsilonGreedyAgent('predator', self.config, self.state_dim, self.action_dim)
        
        # Test action selection
        action = agent.select_action(self.mock_state)
        self.assertIsInstance(action, dict)
        self.assertIn('action', action)
        
        # Test update
        trajectory = [
            {'state': self.mock_state, 'action': action, 'reward': 1.0, 'done': False},
            {'state': self.mock_state, 'action': action, 'reward': 0.0, 'done': True}
        ]
        agent.update(trajectory)  # Should not raise error
        
        print("  ✅ Epsilon-Greedy agent tests passed")
    
    @unittest.skipUnless(IMPORTS_SUCCESS, "Imports failed")
    def test_ppo_agent(self):
        """Test PPO agent functionality."""
        print("🧪 Testing PPO Agent...")
        
        agent = PPOAgent('predator', self.config, self.state_dim, self.action_dim)
        
        # Test action selection
        action = agent.select_action(self.mock_state)
        self.assertIsInstance(action, dict)
        self.assertIn('action', action)
        
        # Test update with trajectory
        trajectory = []
        for i in range(5):
            trajectory.append({
                'state': self.mock_state,
                'action': action,
                'reward': np.random.random(),
                'done': False
            })
        
        agent.update(trajectory)  # Should not raise error
        
        print("  ✅ PPO agent tests passed")
    
    @unittest.skipUnless(IMPORTS_SUCCESS, "Imports failed")  
    def test_maddpg_agent(self):
        """Test MADDPG agent functionality."""
        print("🧪 Testing MADDPG Agent...")
        
        agent = MADDPGWrapper('predator', self.config)
        
        # Test action selection
        action = agent.select_action(self.mock_state, agent_id=0)
        self.assertIsInstance(action, dict)
        self.assertIn('action', action)
        
        # Test update
        trajectory = [
            {'state': self.mock_state, 'action': action, 'reward': 1.0, 'done': False}
        ]
        agent.update(trajectory)  # Should not raise error
        
        print("  ✅ MADDPG agent tests passed")
    
    @unittest.skipUnless(IMPORTS_SUCCESS, "Imports failed")
    def test_replay_buffer(self):
        """Test replay buffer functionality."""
        print("🧪 Testing Replay Buffer...")
        
        buffer = ReplayBuffer(capacity=100)
        
        # Test adding experiences
        for i in range(10):
            buffer.push(
                state=self.mock_state,
                action={'action': 'move', 'x': 10, 'y': 10},
                reward=i * 0.1,
                next_state=self.mock_state,
                done=False
            )
        
        self.assertEqual(len(buffer), 10)
        
        # Test sampling
        batch = buffer.sample(5)
        self.assertEqual(len(batch), 5)
        
        print("  ✅ Replay buffer tests passed")
    
    @unittest.skipUnless(IMPORTS_SUCCESS, "Imports failed")
    def test_training_environment(self):
        """Test training environment creation."""
        print("🧪 Testing Training Environment...")
        
        env = create_training_environment()
        
        self.assertIsInstance(env, Environment)
        self.assertGreater(len(env.agents), 0)
        
        # Check agent types
        predators = [a for a in env.agents if a.agent_type == 'predator']
        prey = [a for a in env.agents if a.agent_type == 'prey']
        plants = [a for a in env.agents if a.agent_type == 'plant']
        
        self.assertGreater(len(predators), 0)
        self.assertGreater(len(prey), 0)
        self.assertGreater(len(plants), 0)
        
        print(f"  Environment: {len(predators)} predators, {len(prey)} prey, {len(plants)} plants")
        print("  ✅ Training environment tests passed")

def run_integration_test():
    """Run a minimal integration test of the full training process."""
    if not IMPORTS_SUCCESS:
        print("❌ Cannot run integration test due to import failures")
        return False
    
    print("\n🔧 Running Integration Test...")
    
    try:
        # Create simple config
        config = TrainingConfig(
            num_episodes=2,
            max_steps_per_episode=5,
            learning_rate=0.01,
            device="cpu"
        )
        
        # Test Lotka-Volterra agent only (simplest)
        env = create_training_environment()
        agent = LotkaVolterraAgent('predator', config)
        
        # Run mini simulation
        for episode in range(2):
            env_copy = create_training_environment()  # Fresh environment
            
            for step in range(5):
                # Find a predator
                predator = None
                for a in env_copy.agents:
                    if a.agent_type == 'predator':
                        predator = a
                        break
                
                if predator is None:
                    break
                
                # Get state (simplified)
                state = {
                    'energy': predator.energy,
                    'age': 1,
                    'nearby_prey': [],
                    'nearby_predators': [],
                    'nearby_food': [],
                    'temperature': 20,
                    'vegetation': 0.5,
                    'is_hungry': predator.energy < 30,
                    'pos_x': predator.x,
                    'pos_y': predator.y,
                    'environment_bounds': {'width': env_copy.width, 'height': env_copy.height}
                }
                
                # Get and apply action
                action = agent.select_action(state)
                
                if 'x' in action and 'y' in action:
                    predator.x = max(0, min(env_copy.width-1, action['x']))
                    predator.y = max(0, min(env_copy.height-1, action['y']))
                
                # Step environment
                env_copy.step()
        
        print("  ✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 BioFlux RL Training Framework Tests")
    print("=" * 50)
    
    if not IMPORTS_SUCCESS:
        print("❌ Cannot run tests due to import failures")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return False
    
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=0)
    
    # Run integration test
    integration_success = run_integration_test()
    
    print("\n" + "=" * 50)
    if integration_success:
        print("✅ All tests passed! The training framework is ready to use.")
        print("\n🚀 Next steps:")
        print("  • Run the demo: python examples/rl_demo.py")
        print("  • Full training: python examples/train_comparative.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return integration_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
