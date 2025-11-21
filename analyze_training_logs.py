#!/usr/bin/env python3
"""
Parse WAN2.2 training logs and generate loss analysis
Extracts step and loss values from run_high.log and run_low.log
Creates CSV files and matplotlib visualization
"""

import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def parse_log_file(log_path):
    """
    Parse a training log file and extract step numbers and loss values.
    Returns list of (step, loss) tuples.
    """
    step_loss_pairs = []
    
    # Pattern to match lines like: steps:   3%|███ | 109/3600 [..., avr_loss=0.0926]
    pattern = r'steps:\s*\d+%.*?\|\s*(\d+)/\d+.*?avr_loss=([\d.]+)'
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    step_loss_pairs.append((step, loss))
    except FileNotFoundError:
        print(f"Warning: {log_path} not found", file=sys.stderr)
        return []
    
    return step_loss_pairs


def save_csv(data, output_path, header="step,loss"):
    """Save step/loss data to CSV file."""
    with open(output_path, 'w') as f:
        f.write(header + '\n')
        for step, loss in data:
            f.write(f"{step},{loss}\n")
    print(f"Saved CSV: {output_path}")


def smooth_data(data, window=20):
    """Apply moving average smoothing to loss data."""
    if len(data) < window:
        return data
    
    steps, losses = zip(*data)
    smoothed = []
    
    for i in range(len(losses)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(losses), i + window // 2 + 1)
        avg_loss = sum(losses[start_idx:end_idx]) / (end_idx - start_idx)
        smoothed.append((steps[i], avg_loss))
    
    return smoothed


def create_plot(high_data, low_data, combined_data, output_path):
    """Create matplotlib plot comparing high and low noise training."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    if high_data:
        high_steps, high_losses = zip(*high_data)
        # Raw data with transparency
        ax.plot(high_steps, high_losses, label='High Noise (raw)', 
                alpha=0.3, linewidth=0.8, color='#1f77b4')
        # Smoothed trend line
        high_smooth = smooth_data(high_data, window=20)
        smooth_steps, smooth_losses = zip(*high_smooth)
        ax.plot(smooth_steps, smooth_losses, label='High Noise (smoothed)', 
                alpha=0.9, linewidth=2, color='#1f77b4')
    
    if low_data:
        low_steps, low_losses = zip(*low_data)
        # Raw data with transparency
        ax.plot(low_steps, low_losses, label='Low Noise (raw)', 
                alpha=0.3, linewidth=0.8, color='#ff7f0e')
        # Smoothed trend line
        low_smooth = smooth_data(low_data, window=20)
        smooth_steps, smooth_losses = zip(*low_smooth)
        ax.plot(smooth_steps, smooth_losses, label='Low Noise (smoothed)', 
                alpha=0.9, linewidth=2, color='#ff7f0e')
    
    if combined_data:
        combined_steps, combined_losses = zip(*combined_data)
        # Raw data with transparency
        ax.plot(combined_steps, combined_losses, label='Combined Noise (raw)', 
                alpha=0.3, linewidth=0.8, color='#2ca02c')
        # Smoothed trend line
        combined_smooth = smooth_data(combined_data, window=20)
        smooth_steps, smooth_losses = zip(*combined_smooth)
        ax.plot(smooth_steps, smooth_losses, label='Combined Noise (smoothed)', 
                alpha=0.9, linewidth=2, color='#2ca02c')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Loss', fontsize=12)  
    ax.set_title('WAN2.2 LoRA Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add minor gridlines
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    """Main analysis function."""
    # Get log directory from command line or use current directory
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    
    high_log = log_dir / "run_high.log"
    low_log = log_dir / "run_low.log"
    combined_log = log_dir / "run_combined.log"
    
    print("Analyzing training logs...")
    print(f"  High noise log: {high_log}")
    print(f"  Low noise log: {low_log}")
    print(f"  Combined noise log: {combined_log}")
    # Parse logs
    high_data = parse_log_file(high_log)
    low_data = parse_log_file(low_log)
    combined_data = parse_log_file(combined_log)
    if not high_data and not low_data and not combined_data:
        print("Error: No training data found in log files", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Found {len(high_data)} high noise steps")
    print(f"  Found {len(low_data)} low noise steps")
    print(f"  Found {len(combined_data)} combined noise steps")
    # Create output directory
    output_dir = log_dir / "training_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Save CSV files
    if high_data:
        save_csv(high_data, output_dir / "high_noise_loss.csv")
    if low_data:
        save_csv(low_data, output_dir / "low_noise_loss.csv")
    if combined_data:
        save_csv(combined_data, output_dir / "combined_noise_loss.csv")
    # Create plot
    create_plot(high_data, low_data, combined_data, output_dir / "training_loss_plot.png")
    
    # Generate summary statistics
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("WAN2.2 LoRA Training Summary\n")
        f.write("=" * 50 + "\n\n")
        
        if high_data:
            high_losses = [loss for _, loss in high_data]
            f.write(f"High Noise Training:\n")
            f.write(f"  Total steps: {len(high_data)}\n")
            f.write(f"  Initial loss: {high_losses[0]:.4f}\n")
            f.write(f"  Final loss: {high_losses[-1]:.4f}\n")
            f.write(f"  Min loss: {min(high_losses):.4f}\n")
            f.write(f"  Max loss: {max(high_losses):.4f}\n")
            f.write(f"  Mean loss: {sum(high_losses)/len(high_losses):.4f}\n")
            f.write("\n")
        
        if low_data:
            low_losses = [loss for _, loss in low_data]
            f.write(f"Low Noise Training:\n")
            f.write(f"  Total steps: {len(low_data)}\n")
            f.write(f"  Initial loss: {low_losses[0]:.4f}\n")
            f.write(f"  Final loss: {low_losses[-1]:.4f}\n")
            f.write(f"  Min loss: {min(low_losses):.4f}\n")
            f.write(f"  Max loss: {max(low_losses):.4f}\n")
            f.write(f"  Mean loss: {sum(low_losses)/len(low_losses):.4f}\n")
        if combined_data:
            combined_losses = [loss for _, loss in combined_data]
            f.write(f"Combined Noise Training:\n")
            f.write(f"  Total steps: {len(combined_data)}\n")
            f.write(f"  Initial loss: {combined_losses[0]:.4f}\n")
            f.write(f"  Final loss: {combined_losses[-1]:.4f}\n")
            f.write(f"  Min loss: {min(combined_losses):.4f}\n")
            f.write(f"  Max loss: {max(combined_losses):.4f}\n")
            f.write(f"  Mean loss: {sum(combined_losses)/len(combined_losses):.4f}\n")
    print(f"Saved summary: {summary_path}")
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

