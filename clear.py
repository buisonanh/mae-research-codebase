import torch
import gc

def clear_gpu_memory():
    # Move all models to CPU and delete
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(f"Clearing: {type(obj)}")
                obj.cpu()
        except:
            pass
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared")

if __name__ == "__main__":
    clear_gpu_memory()