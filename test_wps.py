#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/Users/jasperlinthorst/Documents/fragmentomics')

import numpy as np
import pysam

# Import the wps function directly
from cfstats.ft import wps

def test_wps():
    # Test parameters
    bam_file = "/Users/jasperlinthorst/Documents/fragmentomics/R206C.cram"
    reference = "/Users/jasperlinthorst/Documents/data/hg38flat.fa"
    chromosome = "chr1"
    start = 65419
    end = 75419  # 10,000 bp region
    
    print(f"Testing WPS function for {chromosome}:{start}-{end}")
    
    try:
        # Open BAM file
        bam = pysam.AlignmentFile(bam_file, "rc", reference_filename=reference)
        
        # Call WPS function
        signal = wps(bam, chromosome, start, end)
        
        print(f"WPS calculation completed")
        print(f"Signal shape: {signal.shape}")
        print(f"Signal dtype: {signal.dtype}")
        print(f"Signal min: {np.min(signal)}")
        print(f"Signal max: {np.max(signal)}")
        print(f"Signal mean: {np.mean(signal):.2f}")
        print(f"First 10 values: {signal[:10]}")
        print(f"Last 10 values: {signal[-10:]}")
        
        bam.close()
        
        return signal
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_wps()
