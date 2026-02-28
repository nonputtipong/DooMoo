import torch
import sys

def inspect_model(model_path):
    print(f"Loading '{model_path}'...")
    try:
        model = torch.jit.load(model_path)
        print("Model loaded successfully.\n")
        
        # Test input shapes that divide into 26 (from the error trace)
        # The error said shape '[2, 26, 2, 26, -1]' is invalid...
        # 26 * 14 (typical patch size) = 364
        # 26 * 16 (typical patch size) = 416
        shapes_to_test = [(1, 3, 364, 364), (1, 3, 416, 416), (1, 3, 1024, 1024), (1, 3, 518, 518)]
        print("--- TESTING INPUT SHAPES ---")
        
        success = False
        for shape in shapes_to_test:
            print(f"Testing input shape: {shape}...")
            dummy_input = torch.rand(*shape)
            try:
                outputs = model(dummy_input)
                print(f"SUCCESS! Model accepts {shape}")
                print("\n--- OUTPUT TENSORS ---")
                if isinstance(outputs, tuple):
                    for i, out in enumerate(outputs):
                        print(f"Output {i}: {out.shape} - dtype: {out.dtype}")
                else:
                    print(f"Output: {outputs.shape} - dtype: {outputs.dtype}")
                success = True
                break
            except Exception as e:
                pass
                
        if not success:
            print("\nCould not determine input shape automatically.")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_path>")
        sys.exit(1)
    inspect_model(sys.argv[1])
