cls_dict = {
    0: 'Disturbed',
    1: 'Merging',
    2: 'Round Smooth',
    3: 'Smooth, Cigar shaped',
    4: 'Cigar Shaped Smooth',
    5: 'Barred Spiral',
    6: 'Unbarred Tight Spiral',
    7: 'Unbarred Loose Spiral',
    8: 'Edge-on without Bulge',
    9: 'Edge-on with Bulge'
}

def cls_lookup(class_idx: int) -> str:
    """
    Obtener el nombre de la clase a partir del índice
    """
    return cls_dict.get(class_idx, "Unknown")