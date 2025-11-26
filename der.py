from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
import numpy as np

def calculate_der(reference_segments, hypothesis_segments):
    """
    Calculates the Diarization Error Rate (DER).

    Args:
        reference_segments (list of dict): Ground truth segments 
            [{'start': float, 'end': float, 'label': str}, ...]
        hypothesis_segments (list of dict): System output segments 
            [{'start': float, 'end': float, 'label': int}, ...]

    Returns:
        float: The Diarization Error Rate (DER).
    """
   
    reference = Annotation("reference")
    for seg in reference_segments:
        reference[Segment(seg['start'], seg['end'])] = seg['label']
        
   
    hypothesis = Annotation("hypothesis")
    
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'} 
    
    for seg in hypothesis_segments:
        label = label_map.get(seg['label'], f"Unknown_{seg['label']}")
        hypothesis[Segment(seg['start'], seg['end'])] = label

    print("Reference Annotation:")
    print(reference.get_timeline())
    print("\nHypothesis Annotation:")
    print(hypothesis.get_timeline())
    
    
    metric = DiarizationErrorRate()
    der_value = metric(reference, hypothesis, uem=reference.get_timeline())
    
    fa = metric['false alarm'] / metric['total']
    miss = metric['missed detection'] / metric['total']
    error = metric['confusion'] / metric['total']

    print(f"\n--- DER Components ---")
    print(f"Total reference time: {metric['total']:.2f} s")
    print(f"False Alarm (FA) Rate: {fa*100:.2f} %")
    print(f"Missed Detection (MISS) Rate: {miss*100:.2f} %")
    print(f"Speaker Confusion (ERROR) Rate: {error*100:.2f} %")
    
    return der_value


REFERENCE_SEGMENTS = [
    {'start': 0.00, 'end': 8.00, 'label': 'English'},
    {'start': 8.00, 'end': 14.00, 'label': 'Hindi'},
    {'start': 14.00, 'end': 19.00, 'label': 'English'},
]


HYPOTHESIS_SEGMENTS = [
    
    {'start': 0.00, 'end': 7.0, 'label': 0}, 
    {'start': 7.0, 'end': 13.50, 'label': 1}, 
    {'start': 13.50, 'end': 19.15, 'label': 0}, 
]

if len(HYPOTHESIS_SEGMENTS) < 1:
    print("\nERROR: Please replace the HYPOTHESIS_SEGMENTS placeholder with the actual output from ahc.py.")
else:
    final_der = calculate_der(REFERENCE_SEGMENTS, HYPOTHESIS_SEGMENTS)
    
    print(f"\n=======================================================")
    print(f"FINAL DIARIZATION ERROR RATE (DER): {final_der:.4f}")
    print(f"=======================================================")
    
    if final_der < 0.20:
        print("\nINFERENCE: This DER value is excellent for a baseline system.")
    else:
        print("\nINFERENCE: The DER is high. Check for severe boundary errors or label confusion.")