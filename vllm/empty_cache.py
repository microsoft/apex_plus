import torch

def empty_cache():
    torch.cuda.empty_cache()
    print("PyTorch cache emptied.")

if __name__ == "__main__":
    empty_cache()
