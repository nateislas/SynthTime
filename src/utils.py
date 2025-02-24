def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single-GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (multi-GPU)

    # Ensure deterministic behavior in CUDA (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slows down training but ensures reproducibility

    print(f"Random seed set to: {seed}")