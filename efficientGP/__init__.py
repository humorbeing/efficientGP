__version__ = "0.0.1"  # 20250804
__version__ = "0.0.2"  # 20250804



import os
ROOT = os.path.dirname(__file__)
ROOT = os.path.dirname(ROOT)

# Absolute Path (AP) function
def AP(relative_path):    
    return os.path.join(ROOT, relative_path)


# target_file = 'xxxxx'
# from xxx import AP
# target_file = AP(target_file)