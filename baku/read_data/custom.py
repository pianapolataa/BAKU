from custom_suite import PKLDataset

def bc_dataset(demo_folder):
    """
    Returns iterable dataset for BAKU BC training.
    """
    return PKLDataset(demo_folder)
