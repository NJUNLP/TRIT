#!/usr/bin/env python3
"""
Self-improving via Self-translation (StS) Training Example

This script demonstrates how to use the StS trainer for improving model's multilingual reasoning capabilities.
The training process consists of three stages:
1. English reasoning and filtering
2. Translation to target language  
3. Multilingual reasoning and translation verification
"""

import os
import hydra
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="../configs", config_name="sts_config_example", version_base=None)
def main(config):
    """
    Main function to run StS training
    
    Args:
        config: Hydra configuration object loaded from sts_config_example.yaml
    """
    print("=" * 80)
    print("Starting Self-improving via Self-translation (StS) Training")
    print("=" * 80)
    
    # Print key StS parameters
    print(f"Target Language: {config.data.target_language}")
    print(f"Translation Source Accuracy Range: {config.data.translation_source_acc_lower} - {config.data.translation_source_acc_upper}")
    print(f"QT Training Ratio: {config.data.qt_training_ratio}")
    print("=" * 80)
    
    # Validate StS-specific configuration
    assert config.trainer.task == "sts", "Task must be set to 'sts' for StS training"
    assert hasattr(config.data, "target_language"), "target_language must be specified in data config"
    assert hasattr(config.data, "translation_prompt"), "translation_prompt must be specified in data config"
    
    # Run the training
    run_ppo(config)
    
    print("=" * 80)
    print("StS Training Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()