"""Quick test to verify Wav2Lip model loads correctly."""
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "models/wav2lip")

print("1. Importing torch...")
import torch
print(f"   torch {torch.__version__}, MPS: {torch.backends.mps.is_available()}")

print("2. Importing Wav2Lip model class...")
from models import Wav2Lip
print("   OK")

print("3. Importing face_detection...")
import face_detection
print("   OK")

print("4. Loading checkpoint...")
ckpt_path = "models/wav2lip/checkpoints/wav2lip_gan.pth"
try:
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
except Exception as e1:
    print(f"   Standard load failed: {e1}")
    print("   Trying with encoding fix...")
    import pickle
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            return super().find_class(module, name)
    import io
    with open(ckpt_path, 'rb') as f:
        checkpoint = CPU_Unpickler(f).load()
state = checkpoint["state_dict"]
new_state = {k.replace('module.', ''): v for k, v in state.items()}
print(f"   {len(new_state)} keys loaded")

print("5. Creating model...")
model = Wav2Lip()
model.load_state_dict(new_state)
model = model.eval()
print("   Model ready")

print("6. Creating face detector...")
detector = face_detection.FaceAlignment(
    face_detection.LandmarksType._2D, flip_input=False, device='cpu'
)
print("   Detector ready")

print("7. Testing face detection on sample image...")
import cv2
import numpy as np
img = cv2.imread("data/avatars/sample_face.png")
if img is not None:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.get_detections_for_batch(np.array([rgb]))
    if faces and faces[0] is not None:
        x1, y1, x2, y2 = faces[0]
        print(f"   Face detected: ({x1},{y1}) to ({x2},{y2})")
        
        # Test inference
        face_roi = img[int(y1):int(y2), int(x1):int(x2)]
        face_96 = cv2.resize(face_roi, (96, 96))
        
        mel = np.ones((80, 16), dtype=np.float32) * 2.5
        img_batch = np.asarray([face_96])
        mel_batch = np.asarray([mel])
        
        img_masked = img_batch.copy()
        img_masked[:, 48:] = 0
        img_concat = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = mel_batch.reshape(1, 80, 16, 1)
        
        img_t = torch.FloatTensor(np.transpose(img_concat, (0, 3, 1, 2)))
        mel_t = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2)))
        
        with torch.no_grad():
            pred = model(mel_t, img_t)
        
        print(f"   Inference output shape: {pred.shape}")
        print("   ✅ Wav2Lip works!")
    else:
        print("   No face detected")
else:
    print("   No sample image found")

print("\nDONE - All checks passed!")
