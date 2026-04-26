"""Test loading wav2lip checkpoint with different methods."""
import sys, os, struct
os.environ["OMP_NUM_THREADS"] = "1"

path = "models/wav2lip/checkpoints/wav2lip_gan.pth"

# Check file header
with open(path, 'rb') as f:
    header = f.read(20)
    print(f"First 20 bytes: {header.hex()}")
    print(f"First 20 bytes (repr): {header!r}")
    
    # Check if it's a ZIP file (new PyTorch format)
    f.seek(0)
    magic = f.read(4)
    print(f"Magic: {magic.hex()}")
    
    if magic[:2] == b'PK':
        print("File is ZIP format (new PyTorch)")
    elif magic[:2] == b'\x80\x02':
        print("File is pickle protocol 2 (old PyTorch)")
    else:
        print(f"Unknown format: {magic!r}")

# Try loading with pickle directly
print("\nTrying pickle.load directly...")
import pickle
try:
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    print(f"pickle.load OK, type: {type(data)}")
except Exception as e:
    print(f"pickle.load failed: {e}")

# Try with torch
print("\nTrying torch.load...")
import torch
try:
    data = torch.load(path, map_location='cpu', weights_only=False)
    print(f"torch.load OK, type: {type(data)}")
except Exception as e:
    print(f"torch.load failed: {e}")
    
    # Try with encoding='latin1'
    print("\nTrying torch.load with encoding='latin1'...")
    try:
        data = torch.load(path, map_location='cpu', weights_only=False, encoding='latin1')
        print(f"torch.load (latin1) OK, type: {type(data)}")
    except Exception as e2:
        print(f"torch.load (latin1) failed: {e2}")
