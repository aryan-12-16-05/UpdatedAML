import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_engine import load_checkpoint, _get_device

def inspect():
    model, fe = load_checkpoint()
    
    print("Edge Scaler Mean:", fe.edge_scaler.mean_)
    print("Edge Scaler Scale:", fe.edge_scaler.scale_)
    print("Node Scaler Mean:", fe.node_scaler.mean_)
    print("Node Scaler Scale:", fe.node_scaler.scale_)
    
    # Check categorical maps
    print("Payment Format Map:", getattr(fe, 'payment_format_map', None))
    print("Payment Currency Map:", getattr(fe, 'payment_currency_map', None))

if __name__ == "__main__":
    inspect()
