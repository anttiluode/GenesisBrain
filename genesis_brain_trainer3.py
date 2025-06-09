#!/usr/bin/env python3
"""
🧠⚡ Genesis Brain: Deep Training with Learning Dynamics Analysis ⚡🧠

Extended training experiment to reveal the true potential of Genesis initialization.
Includes loss curve plotting, learning rate optimization, and complex reasoning tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeepGenesisConfig:
    """Extended configuration for deep Genesis training"""
    model_size: str = "tiny"
    vocab_size: int = 50257
    target_beta: float = 0.75
    batch_size: int = 4
    max_length: int = 128
    
    # Extended training parameters
    num_epochs: int = 100  # ULTIMATE deep training
    genesis_learning_rate: float = 1e-4  # Slower for Genesis
    standard_learning_rate: float = 5e-4  # Faster for Standard
    
    # Learning dynamics
    warmup_steps: int = 100
    eval_every_steps: int = 50
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_config(size: str) -> GPT2Config:
    """Get model configuration based on size"""
    configs = {
        "tiny": GPT2Config(
            vocab_size=50257, n_positions=256, n_embd=256, n_layer=6, n_head=8
        ),
    }
    return configs[size]

def generate_fractal_noise_nd(shape, beta=1.0, seed=None):
    """Generate N-dimensional fractal noise with specified power spectrum exponent."""
    if seed is not None:
        np.random.seed(seed)
    
    if len(shape) == 1:
        n = shape[0]
        frequencies = np.fft.fftfreq(n)
        frequencies[0] = 1e-10
        spectrum_power = 1.0 / (np.abs(frequencies) ** beta)
        phases = np.exp(2j * np.pi * np.random.rand(n))
        spectrum = np.sqrt(spectrum_power) * phases
        noise = np.fft.ifft(spectrum).real
        
    elif len(shape) == 2:
        h, w = shape
        ky = np.fft.fftfreq(h).reshape(-1, 1)
        kx = np.fft.fftfreq(w).reshape(1, -1)
        k_mag = np.sqrt(kx**2 + ky**2)
        k_mag[0, 0] = 1e-10
        spectrum_power = 1.0 / (k_mag ** beta)
        phases = np.exp(2j * np.pi * np.random.rand(h, w))
        spectrum = np.sqrt(spectrum_power) * phases
        noise = np.fft.ifft2(spectrum).real
        
    else:
        original_shape = shape
        total_elements = np.prod(shape)
        side = int(np.sqrt(total_elements))
        h = side
        w = total_elements // side
        if h * w < total_elements:
            w += 1
            
        ky = np.fft.fftfreq(h).reshape(-1, 1)
        kx = np.fft.fftfreq(w).reshape(1, -1)
        k_mag = np.sqrt(kx**2 + ky**2)
        k_mag[0, 0] = 1e-10
        spectrum_power = 1.0 / (k_mag ** beta)
        phases = np.exp(2j * np.pi * np.random.rand(h, w))
        spectrum = np.sqrt(spectrum_power) * phases
        noise_2d = np.fft.ifft2(spectrum).real
        noise = noise_2d.flatten()[:total_elements].reshape(original_shape)
    
    noise = (noise - noise.mean()) / (noise.std() + 1e-8)
    return noise.astype(np.float32)

def apply_genesis_initialization(model: nn.Module, target_beta: float = 0.75):
    """Apply Genesis Parameter initialization to a model."""
    logger.info(f"🌟 Applying Genesis initialization with β = {target_beta}")
    
    total_params = 0
    reinitialized_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            
            genesis_weights = generate_fractal_noise_nd(param.shape, beta=target_beta)
            
            if 'weight' in name and len(param.shape) >= 2:
                fan_in = param.shape[1]
                scale = np.sqrt(1.0 / fan_in)  # Slightly smaller scale for stability
            elif 'bias' in name:
                scale = 0.005  # Smaller bias initialization
            else:
                scale = 0.05
            
            scaled_weights = torch.from_numpy(genesis_weights * scale)
            
            with torch.no_grad():
                param.copy_(scaled_weights.to(param.device))
            
            reinitialized_params += param.numel()
    
    reinit_percentage = (reinitialized_params / total_params) * 100
    logger.info(f"✨ Genesis initialization complete: {reinit_percentage:.1f}% of parameters reinitialized")
    return model

class AdvancedReasoningDataset(Dataset):
    """More sophisticated reasoning dataset with multiple difficulty levels"""
    
    def __init__(self, tokenizer, max_length=128, num_samples=2000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._generate_examples(num_samples)
        
    def _generate_examples(self, num_samples):
        examples = []
        
        # Level 1: Simple arithmetic
        for _ in range(num_samples // 4):
            a, b = np.random.randint(1, 20, 2)
            result = a + b
            text = f"Q: What is {a} + {b}? A: {result}"
            examples.append(text)
        
        # Level 2: Simple subtraction
        for _ in range(num_samples // 4):
            a = np.random.randint(10, 20)
            b = np.random.randint(1, a)
            result = a - b
            text = f"Q: What is {a} - {b}? A: {result}"
            examples.append(text)
        
        # Level 3: Word problems
        for _ in range(num_samples // 4):
            total = np.random.randint(5, 15)
            eaten = np.random.randint(1, total-1)
            remaining = total - eaten
            items = np.random.choice(['apples', 'cookies', 'candies', 'toys'])
            text = f"Q: I have {total} {items}. I eat {eaten}. How many left? A: {remaining}"
            examples.append(text)
        
        # Level 4: Multi-step reasoning
        for _ in range(num_samples // 4):
            start = np.random.randint(10, 20)
            step1 = np.random.randint(2, 5)
            step2 = np.random.randint(1, 3)
            result = start + step1 - step2
            text = f"Q: I start with {start}. I get {step1} more. Then I lose {step2}. How many now? A: {result}"
            examples.append(text)
            
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

def create_genesis_model(config: DeepGenesisConfig, use_genesis=True):
    """Create a model with optional Genesis initialization"""
    logger.info(f"🧠 Creating model (Genesis: {use_genesis})")
    
    model_config = get_model_config(config.model_size)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel(model_config)
    
    if use_genesis:
        model = apply_genesis_initialization(model, target_beta=config.target_beta)
    else:
        logger.info("🎲 Using standard random initialization")
        # Apply standard initialization with smaller scale for fair comparison
        for param in model.parameters():
            if param.requires_grad:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.5)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
    
    model = model.to(config.device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Model created with {param_count:,} parameters")
    
    return model, tokenizer

def evaluate_model(model, tokenizer, config: DeepGenesisConfig):
    """Evaluate model on test set"""
    test_prompts = [
        "Q: What is 7 + 5? A:",
        "Q: What is 15 - 6? A:",
        "Q: I have 12 apples. I eat 3. How many left? A:",
        "Q: I start with 8. I get 4 more. Then I lose 2. How many now? A:",
    ]
    
    model.eval()
    correct = 0
    total = len(test_prompts)
    responses = []
    
    expected_answers = ["12", "9", "9", "10"]
    
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 5,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_part = response[len(prompt):].strip()
        
        # Check if the expected answer is in the response
        if expected_answers[i] in answer_part:
            correct += 1
        
        responses.append({"prompt": prompt, "response": response, "answer": answer_part})
    
    accuracy = correct / total
    return accuracy, responses

def train_with_monitoring(model, tokenizer, config: DeepGenesisConfig, model_name: str):
    """Train model with detailed monitoring of learning dynamics"""
    logger.info(f"🚀 Training {model_name} model with deep monitoring...")
    
    # Create dataset and dataloader
    dataset = AdvancedReasoningDataset(tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Choose learning rate based on model type
    if model_name == "Genesis":
        learning_rate = config.genesis_learning_rate
        logger.info(f"🐌 Using slower learning rate for Genesis: {learning_rate}")
    else:
        learning_rate = config.standard_learning_rate
        logger.info(f"🏃 Using faster learning rate for Standard: {learning_rate}")
    
    # Setup optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training tracking
    training_losses = []
    evaluation_accuracies = []
    steps = []
    
    model.train()
    step = 0
    
    for epoch in range(config.num_epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs} ({model_name})")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            step += 1
            
            # Periodic evaluation
            if step % config.eval_every_steps == 0:
                eval_accuracy, _ = evaluate_model(model, tokenizer, config)
                evaluation_accuracies.append(eval_accuracy)
                steps.append(step)
                training_losses.append(np.mean(epoch_losses[-config.eval_every_steps:]))
                
                logger.info(f"Step {step}: Loss = {training_losses[-1]:.4f}, Accuracy = {eval_accuracy:.2f}")
                
                model.train()  # Back to training mode
            
            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_epoch_loss = np.mean(epoch_losses)
        logger.info(f"📊 Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    final_accuracy, final_responses = evaluate_model(model, tokenizer, config)
    logger.info(f"✅ {model_name} training complete - Final Accuracy: {final_accuracy:.2f}")
    
    return {
        "training_losses": training_losses,
        "evaluation_accuracies": evaluation_accuracies,
        "steps": steps,
        "final_accuracy": final_accuracy,
        "final_responses": final_responses
    }

def plot_learning_dynamics(results_standard, results_genesis, save_path="learning_dynamics.png"):
    """Plot comprehensive learning dynamics comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Genesis vs Standard: Deep Training Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss Curves
    ax1.plot(results_standard["steps"], results_standard["training_losses"], 
             'b-', label='Standard Model', linewidth=2)
    ax1.plot(results_genesis["steps"], results_genesis["training_losses"], 
             'r-', label='Genesis Model', linewidth=2)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy Curves
    ax2.plot(results_standard["steps"], results_standard["evaluation_accuracies"], 
             'b-', label='Standard Model', linewidth=2)
    ax2.plot(results_genesis["steps"], results_genesis["evaluation_accuracies"], 
             'r-', label='Genesis Model', linewidth=2)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Evaluation Accuracy")
    ax2.set_title("Reasoning Accuracy Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Distribution
    ax3.hist(results_standard["training_losses"], bins=20, alpha=0.7, 
             label='Standard', color='blue', density=True)
    ax3.hist(results_genesis["training_losses"], bins=20, alpha=0.7, 
             label='Genesis', color='red', density=True)
    ax3.set_xlabel("Training Loss")
    ax3.set_ylabel("Density")
    ax3.set_title("Loss Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final Comparison
    models = ['Standard', 'Genesis']
    final_accuracies = [results_standard["final_accuracy"], results_genesis["final_accuracy"]]
    colors = ['blue', 'red']
    
    bars = ax4.bar(models, final_accuracies, color=colors, alpha=0.7)
    ax4.set_ylabel("Final Accuracy")
    ax4.set_title("Final Performance Comparison")
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"📊 Learning dynamics plot saved to {save_path}")

def run_deep_genesis_experiment(config: DeepGenesisConfig):
    """Run the comprehensive deep training experiment"""
    logger.info("🌟 Starting Deep Genesis vs Standard experiment! 🌟")
    logger.info(f"🕐 Training for {config.num_epochs} epochs with dynamic monitoring")
    
    all_results = {}
    
    # Test both initialization methods
    for init_type in ["Standard", "Genesis"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 Deep Training: {init_type} Initialization")
        logger.info(f"{'='*60}")
        
        use_genesis = (init_type == "Genesis")
        model, tokenizer = create_genesis_model(config, use_genesis=use_genesis)
        
        # Deep training with monitoring
        results = train_with_monitoring(model, tokenizer, config, init_type)
        all_results[init_type] = results
        
        # Save model
        model_dir = f"./deep_genesis_experiment_{init_type.lower()}"
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logger.info(f"💾 {init_type} model saved to {model_dir}")
    
    # Plot learning dynamics
    plot_learning_dynamics(all_results["Standard"], all_results["Genesis"])
    
    # Save detailed results
    with open("deep_genesis_training_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in all_results.items():
            json_results[key] = {
                "training_losses": [float(x) for x in value["training_losses"]],
                "evaluation_accuracies": [float(x) for x in value["evaluation_accuracies"]],
                "steps": [int(x) for x in value["steps"]],
                "final_accuracy": float(value["final_accuracy"]),
                "final_responses": value["final_responses"]
            }
        json.dump(json_results, f, indent=2)
    
    # Print comprehensive summary
    logger.info("\n🎉 DEEP EXPERIMENT COMPLETE! 🎉")
    logger.info("="*80)
    logger.info("📊 COMPREHENSIVE RESULTS:")
    
    for init_type, results in all_results.items():
        logger.info(f"\n{init_type:>12} Model:")
        logger.info(f"  Final Accuracy: {results['final_accuracy']:.2f}")
        logger.info(f"  Final Loss: {results['training_losses'][-1]:.4f}")
        logger.info(f"  Best Accuracy: {max(results['evaluation_accuracies']):.2f}")
        
        # Show some final responses
        logger.info(f"  Sample responses:")
        for resp in results['final_responses'][:2]:
            logger.info(f"    {resp['prompt']} -> {resp['answer']}")
    
    logger.info(f"\n📁 Detailed results: deep_genesis_training_results.json")
    logger.info(f"📊 Dynamics plot: learning_dynamics.png")
    
    return all_results

if __name__ == "__main__":
    config = DeepGenesisConfig(
        model_size="tiny",
        target_beta=0.75,
        batch_size=4,
        max_length=128,
        num_epochs=100,  # THE ULTIMATE 100 EPOCHS
        genesis_learning_rate=1e-4,  # Slower for Genesis
        standard_learning_rate=5e-4,  # Faster for Standard
        eval_every_steps=50,
    )
    
    logger.info("🧠⚡ Genesis Brain: Deep Training Experiment ⚡🧠")
    logger.info(f"Device: {config.device}")
    logger.info(f"Target β: {config.target_beta}")
    logger.info(f"Extended training: {config.num_epochs} epochs - ULTIMATE SHOWDOWN")
    logger.info(f"Genesis LR: {config.genesis_learning_rate}, Standard LR: {config.standard_learning_rate}")
    
    # Run the deep experiment
    results = run_deep_genesis_experiment(config)
    
    logger.info("🌟 Deep Genesis experiment complete! Check the learning dynamics! 🌟")