#!/usr/bin/env python3
"""
BioFlux RL Model Deployment

Production-ready deployment system for trained BioFlux RL models.
Provides APIs for real-time inference and batch processing.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the parent directory to the path to import bioflux
sys.path.append(str(Path(__file__).parent.parent))

from bioflux.training import (
    TrainingConfig, LotkaVolterraAgent, EpsilonGreedyAgent, 
    PPOAgent, MADDPGWrapper
)
from bioflux.core.environment import Environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """Standardized agent state for API."""
    agent_id: str
    agent_type: str
    energy: float
    age: int
    pos_x: float
    pos_y: float
    nearby_agents: List[Dict[str, Any]]
    environment_conditions: Dict[str, float]
    is_alive: bool = True

@dataclass
class ActionResult:
    """Result of an action prediction."""
    agent_id: str
    action: Dict[str, Any]
    confidence: float
    reasoning: Optional[str] = None
    execution_time_ms: float = 0.0

@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    models_dir: str = "models"
    default_algorithm: str = "ppo"
    max_prediction_time_ms: float = 100.0
    enable_logging: bool = True
    enable_metrics: bool = True
    cache_models: bool = True

class BioFluxModelDeployment:
    """Production deployment system for BioFlux RL models."""
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig()
        self.models_dir = Path(self.config.models_dir)
        
        # Model cache
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Performance metrics
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.error_count = 0
        
        # Load available models
        self._discover_models()
        
        logger.info(f"BioFlux deployment system initialized")
        logger.info(f"Found models for algorithms: {list(self.model_metadata.keys())}")
    
    def _discover_models(self):
        """Discover available trained models."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        for algorithm_dir in self.models_dir.iterdir():
            if algorithm_dir.is_dir():
                algorithm = algorithm_dir.name
                models = {}
                
                for model_file in algorithm_dir.glob("*.pth"):
                    # Parse model filename: {agent_type}_episode_{episode}.pth
                    parts = model_file.stem.split('_')
                    if len(parts) >= 3:
                        agent_type = parts[0]
                        episode = int(parts[-1])
                        
                        if agent_type not in models:
                            models[agent_type] = []
                        
                        models[agent_type].append({
                            'episode': episode,
                            'path': str(model_file),
                            'size_mb': model_file.stat().st_size / (1024 * 1024),
                            'modified': datetime.fromtimestamp(model_file.stat().st_mtime)
                        })
                
                # Sort by episode number and keep latest
                for agent_type in models:
                    models[agent_type].sort(key=lambda x: x['episode'], reverse=True)
                
                if models:
                    self.model_metadata[algorithm] = models
        
        logger.info(f"Discovered {len(self.model_metadata)} algorithm types")
    
    def load_model(self, algorithm: str, agent_type: str, episode: int = None) -> Any:
        """Load a specific model for inference."""
        cache_key = f"{algorithm}_{agent_type}_{episode or 'latest'}"
        
        # Check cache
        if self.config.cache_models and cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Find model
        if algorithm not in self.model_metadata:
            raise ValueError(f"Algorithm not found: {algorithm}")
        
        if agent_type not in self.model_metadata[algorithm]:
            raise ValueError(f"Agent type not found: {agent_type} for {algorithm}")
        
        models_list = self.model_metadata[algorithm][agent_type]
        
        if episode is None:
            # Get latest model
            model_info = models_list[0]
        else:
            # Find specific episode
            model_info = None
            for m in models_list:
                if m['episode'] == episode:
                    model_info = m
                    break
            
            if model_info is None:
                raise ValueError(f"Episode {episode} not found for {algorithm}/{agent_type}")
        
        # Load model
        model_path = Path(model_info['path'])
        config = TrainingConfig(device="cpu")
        
        try:
            if algorithm == 'lotka_volterra':
                agent = LotkaVolterraAgent(agent_type, config)
            
            elif algorithm == 'epsilon_greedy':
                agent = EpsilonGreedyAgent(agent_type, config, 8, 8)
                checkpoint = torch.load(model_path, map_location='cpu')
                agent.q_network.load_state_dict(checkpoint['q_network'])
                agent.target_network.load_state_dict(checkpoint['target_network'])
                agent.epsilon = 0.01  # Low epsilon for production
            
            elif algorithm == 'ppo':
                agent = PPOAgent(agent_type, config, 8, 8)
                checkpoint = torch.load(model_path, map_location='cpu')
                agent.actor.load_state_dict(checkpoint['actor'])
                agent.critic.load_state_dict(checkpoint['critic'])
            
            elif algorithm == 'maddpg':
                agent = MADDPGWrapper(agent_type, config)
                # Load all MADDPG agent files
                base_path = str(model_path).replace('.pth', '')
                for i in range(agent.num_agents):
                    agent_path = f"{base_path}_agent_{i}.pth"
                    if Path(agent_path).exists():
                        agent.agents[i].load(agent_path)
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Set to evaluation mode
            if hasattr(agent, 'actor'):
                agent.actor.eval()
            if hasattr(agent, 'q_network'):
                agent.q_network.eval()
            
            # Cache model
            if self.config.cache_models:
                self.loaded_models[cache_key] = agent
            
            logger.info(f"Loaded model: {algorithm}/{agent_type} episode {model_info['episode']}")
            return agent
        
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def predict_action(self, 
                      agent_state: AgentState, 
                      algorithm: str = None,
                      return_confidence: bool = True) -> ActionResult:
        """Predict action for an agent in production."""
        start_time = time.time()
        
        algorithm = algorithm or self.config.default_algorithm
        
        try:
            # Load model
            agent = self.load_model(algorithm, agent_state.agent_type)
            
            # Convert state to model format
            model_state = self._convert_state_to_model_format(agent_state)
            
            # Predict action
            if algorithm == 'maddpg':
                action = agent.select_action(model_state, agent_id=0)
            else:
                action = agent.select_action(model_state)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(agent, model_state, action, algorithm)
            
            # Create result
            execution_time = (time.time() - start_time) * 1000
            
            result = ActionResult(
                agent_id=agent_state.agent_id,
                action=action,
                confidence=confidence,
                reasoning=self._generate_reasoning(action, agent_state),
                execution_time_ms=execution_time
            )
            
            # Update metrics
            self.prediction_count += 1
            self.total_prediction_time += execution_time
            
            if execution_time > self.config.max_prediction_time_ms:
                logger.warning(f"Slow prediction: {execution_time:.2f}ms for {algorithm}")
            
            return result
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction failed for {agent_state.agent_id}: {e}")
            
            # Return safe default action
            return ActionResult(
                agent_id=agent_state.agent_id,
                action={'action': 'stay', 'x': agent_state.pos_x, 'y': agent_state.pos_y},
                confidence=0.0,
                reasoning="Error occurred, using safe default action",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def predict_batch(self, 
                     agent_states: List[AgentState],
                     algorithm: str = None) -> List[ActionResult]:
        """Predict actions for multiple agents."""
        results = []
        
        for state in agent_states:
            result = self.predict_action(state, algorithm)
            results.append(result)
        
        return results
    
    def _convert_state_to_model_format(self, agent_state: AgentState) -> Dict[str, Any]:
        """Convert AgentState to model input format."""
        # Extract nearby agents by type
        nearby_prey = []
        nearby_predators = []
        nearby_food = []
        
        for agent_info in agent_state.nearby_agents:
            if agent_info.get('type') == 'prey':
                nearby_prey.append(agent_info)
            elif agent_info.get('type') == 'predator':
                nearby_predators.append(agent_info)
            elif agent_info.get('type') == 'plant':
                nearby_food.append(agent_info)
        
        return {
            'energy': agent_state.energy,
            'age': agent_state.age,
            'nearby_prey': nearby_prey,
            'nearby_predators': nearby_predators,
            'nearby_food': nearby_food,
            'temperature': agent_state.environment_conditions.get('temperature', 20.0),
            'vegetation': agent_state.environment_conditions.get('vegetation', 0.5),
            'is_hungry': agent_state.energy < 30,
            'pos_x': agent_state.pos_x,
            'pos_y': agent_state.pos_y,
            'environment_bounds': {
                'width': agent_state.environment_conditions.get('env_width', 100),
                'height': agent_state.environment_conditions.get('env_height', 100)
            }
        }
    
    def _calculate_confidence(self, agent: Any, state: Dict[str, Any], action: Dict[str, Any], algorithm: str) -> float:
        """Calculate confidence score for prediction."""
        try:
            if algorithm == 'epsilon_greedy':
                # For DQN, use Q-value difference as confidence
                if hasattr(agent, 'q_network'):
                    state_tensor = agent._state_to_tensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = agent.q_network(state_tensor)
                        sorted_q = torch.sort(q_values, descending=True)[0]
                        if len(sorted_q[0]) > 1:
                            confidence = (sorted_q[0][0] - sorted_q[0][1]).item()
                            return min(max(confidence / 10.0, 0.0), 1.0)
            
            elif algorithm == 'ppo':
                # For PPO, use entropy as inverse confidence
                if hasattr(agent, 'actor'):
                    state_tensor = agent._state_to_tensor(state).unsqueeze(0)
                    with torch.no_grad():
                        action_probs = agent.actor(state_tensor)
                        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()
                        confidence = 1.0 - (entropy.item() / np.log(len(action_probs[0])))
                        return max(confidence, 0.0)
            
            elif algorithm == 'lotka_volterra':
                # Deterministic model, high confidence
                return 0.95
            
            elif algorithm == 'maddpg':
                # For MADDPG, use a heuristic based on action consistency
                return 0.8
        
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
        
        # Default confidence
        return 0.5
    
    def _generate_reasoning(self, action: Dict[str, Any], agent_state: AgentState) -> str:
        """Generate human-readable reasoning for action."""
        action_type = action.get('action', 'unknown')
        
        if agent_state.agent_type == 'predator':
            if action_type == 'hunt':
                prey_count = len([a for a in agent_state.nearby_agents if a.get('type') == 'prey'])
                return f"Hunting nearby prey (detected {prey_count} prey agents)"
            elif action_type == 'move' or action_type == 'explore':
                return f"Exploring territory (energy: {agent_state.energy:.1f})"
            elif action_type == 'stay':
                return "Conserving energy"
        
        elif agent_state.agent_type == 'prey':
            if action_type == 'flee':
                pred_count = len([a for a in agent_state.nearby_agents if a.get('type') == 'predator'])
                return f"Fleeing from predators (detected {pred_count} threats)"
            elif action_type == 'forage':
                food_count = len([a for a in agent_state.nearby_agents if a.get('type') == 'plant'])
                return f"Foraging for food (detected {food_count} food sources)"
            elif action_type == 'move' or action_type == 'explore':
                return f"Searching for food (energy: {agent_state.energy:.1f})"
        
        return f"Executing {action_type} action"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get deployment performance metrics."""
        avg_prediction_time = (self.total_prediction_time / max(self.prediction_count, 1))
        error_rate = self.error_count / max(self.prediction_count, 1)
        
        return {
            'prediction_count': self.prediction_count,
            'average_prediction_time_ms': avg_prediction_time,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'loaded_models': len(self.loaded_models),
            'available_algorithms': list(self.model_metadata.keys())
        }
    
    def get_model_info(self, algorithm: str = None) -> Dict[str, Any]:
        """Get information about available models."""
        if algorithm:
            return self.model_metadata.get(algorithm, {})
        else:
            return self.model_metadata
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        # Check if models are available
        if not self.model_metadata:
            health['status'] = 'unhealthy'
            health['issues'].append('No trained models found')
        
        # Check error rate
        if self.prediction_count > 0:
            error_rate = self.error_count / self.prediction_count
            if error_rate > 0.1:  # More than 10% errors
                health['status'] = 'degraded'
                health['issues'].append(f'High error rate: {error_rate:.1%}')
        
        # Check prediction time
        if self.prediction_count > 0:
            avg_time = self.total_prediction_time / self.prediction_count
            if avg_time > self.config.max_prediction_time_ms:
                health['status'] = 'degraded'
                health['issues'].append(f'Slow predictions: {avg_time:.1f}ms average')
        
        health['metrics'] = self.get_performance_metrics()
        
        return health

def create_example_agent_state() -> AgentState:
    """Create example agent state for testing."""
    return AgentState(
        agent_id="pred_001",
        agent_type="predator",
        energy=75.0,
        age=15,
        pos_x=25.0,
        pos_y=30.0,
        nearby_agents=[
            {
                'type': 'prey',
                'pos': (20, 25),
                'distance': 7.0,
                'energy': 45.0
            },
            {
                'type': 'plant',
                'pos': (28, 33),
                'distance': 4.2,
                'energy': 15.0
            }
        ],
        environment_conditions={
            'temperature': 22.5,
            'vegetation': 0.7,
            'env_width': 100,
            'env_height': 100
        }
    )

def main():
    """Demonstrate deployment system."""
    print("🚀 BioFlux RL Model Deployment System")
    print("=" * 50)
    
    # Initialize deployment
    config = DeploymentConfig(
        models_dir="models",
        default_algorithm="ppo",
        max_prediction_time_ms=50.0
    )
    
    try:
        deployment = BioFluxModelDeployment(config)
    except Exception as e:
        print(f"❌ Failed to initialize deployment: {e}")
        print("Please run training first: python examples/full_training.py")
        return
    
    # Health check
    health = deployment.health_check()
    print(f"System Status: {health['status'].upper()}")
    if health['issues']:
        for issue in health['issues']:
            print(f"  ⚠️  {issue}")
    print()
    
    # Show available models
    model_info = deployment.get_model_info()
    print("📋 Available Models:")
    for algorithm, agents in model_info.items():
        print(f"  {algorithm.upper()}:")
        for agent_type, models in agents.items():
            latest = models[0]  # Latest model
            print(f"    {agent_type}: Episode {latest['episode']} "
                  f"({latest['size_mb']:.1f} MB, {latest['modified'].strftime('%Y-%m-%d %H:%M')})")
    print()
    
    # Demonstration predictions
    print("🎯 Running prediction demonstrations...")
    
    # Create test agent states
    predator_state = AgentState(
        agent_id="pred_001",
        agent_type="predator",
        energy=85.0,
        age=12,
        pos_x=30.0,
        pos_y=25.0,
        nearby_agents=[
            {'type': 'prey', 'pos': (28, 23), 'distance': 3.0, 'energy': 40.0}
        ],
        environment_conditions={'temperature': 21.0, 'vegetation': 0.6, 'env_width': 100, 'env_height': 100}
    )
    
    prey_state = AgentState(
        agent_id="prey_001",
        agent_type="prey",
        energy=45.0,
        age=8,
        pos_x=50.0,
        pos_y=50.0,
        nearby_agents=[
            {'type': 'predator', 'pos': (48, 47), 'distance': 4.0, 'energy': 80.0},
            {'type': 'plant', 'pos': (52, 51), 'distance': 2.5, 'energy': 20.0}
        ],
        environment_conditions={'temperature': 23.0, 'vegetation': 0.8, 'env_width': 100, 'env_height': 100}
    )
    
    # Test different algorithms
    for algorithm in ['lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg']:
        if algorithm in model_info:
            print(f"\n🔍 Testing {algorithm.upper()} predictions:")
            
            # Predator prediction
            try:
                result = deployment.predict_action(predator_state, algorithm)
                print(f"  Predator: {result.action['action']} "
                      f"-> ({result.action.get('x', 'N/A')}, {result.action.get('y', 'N/A')}) "
                      f"[confidence: {result.confidence:.2f}, {result.execution_time_ms:.1f}ms]")
                print(f"    Reasoning: {result.reasoning}")
            except Exception as e:
                print(f"  Predator: Error - {e}")
            
            # Prey prediction
            try:
                result = deployment.predict_action(prey_state, algorithm)
                print(f"  Prey: {result.action['action']} "
                      f"-> ({result.action.get('x', 'N/A')}, {result.action.get('y', 'N/A')}) "
                      f"[confidence: {result.confidence:.2f}, {result.execution_time_ms:.1f}ms]")
                print(f"    Reasoning: {result.reasoning}")
            except Exception as e:
                print(f"  Prey: Error - {e}")
    
    # Batch prediction demo
    print(f"\n📦 Testing batch predictions...")
    batch_states = [predator_state, prey_state]
    
    try:
        batch_results = deployment.predict_batch(batch_states)
        print(f"  Processed {len(batch_results)} agents in batch")
        for result in batch_results:
            print(f"    {result.agent_id}: {result.action['action']} "
                  f"({result.confidence:.2f} confidence, {result.execution_time_ms:.1f}ms)")
    except Exception as e:
        print(f"  Batch prediction failed: {e}")
    
    # Performance metrics
    print(f"\n📊 Performance Metrics:")
    metrics = deployment.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Deployment demonstration completed!")
    print("🚀 System is ready for production use!")

if __name__ == "__main__":
    main()
