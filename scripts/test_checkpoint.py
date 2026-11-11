import subprocess
import os
import re
import sys

def run_test():
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(workspace)
    
    # Clean checkpoints
    if os.path.exists('checkpoints'):
        import shutil
        shutil.rmtree('checkpoints')

    # Run for 10 steps, kill at step 4 (5th step, since 0-indexed)
    print("Starting Run 1 (initial run)...")
    p1 = subprocess.run([
        'py', 'scripts/train_pilot.py', 
        '--model_size', 'test',
        '--batch_size', '2',
        '--seq_len', '32',
        '--total_tokens', '640',
        '--checkpoint_every', '5',
        '--kill_at_step', '4'
    ], capture_output=True, text=True, env=dict(os.environ, WANDB_MODE='disabled'))
    
    out1 = p1.stdout
    if p1.returncode != 0:
        print("Run 1 Failed!")
        print(p1.stderr)
        sys.exit(1)
        
    print(out1)
    
    # Parse loss from run 1
    match = re.search(r'Killing at step 4 with loss: ([\d\.]+)', out1)
    if not match:
        print("FAIL: Did not find kill message in Run 1")
        sys.exit(1)
    loss1 = float(match.group(1))
    
    # Check if checkpoint exists
    if not os.path.exists('checkpoints/pilot_test_step5.pt'):
        print("FAIL: Checkpoint pilot_test_step5.pt not found")
        sys.exit(1)
        
    print(f"Run 1 completed. Saved loss at step 4: {loss1}")
    
    print("Starting Run 2 (Resume)...")
    p2 = subprocess.run([
        'py', 'scripts/train_pilot.py', 
        '--model_size', 'test',
        '--batch_size', '2',
        '--seq_len', '32',
        '--total_tokens', '640',
        '--resume', 'checkpoints/pilot_test_step5.pt'
    ], capture_output=True, text=True, env=dict(os.environ, WANDB_MODE='disabled'))
    
    out2 = p2.stdout
    if p2.returncode != 0:
        print("Run 2 Failed!")
        print(p2.stderr)
        sys.exit(1)
        
    print(out2)
    
    if "Resuming from step 5" not in out2:
        print("FAIL: Did not resume from step 5")
        sys.exit(1)
        
    # Optional: ensure it prints PASS
    print("PASS: Checkpointing test successful!")

if __name__ == '__main__':
    run_test()
