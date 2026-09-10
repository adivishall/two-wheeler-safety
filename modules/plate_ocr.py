"""Minimal EasyOCR plate reader helper.

Kept as a small standalone helper (the video/image pipelines use their own
reader instances and the temporal stabilizer). The EasyOCR reader is created
lazily on first use so that merely importing this module doesn't pull in
easyocr/torch or download detector weights — important for tests and CI.
"""

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"])
    return _reader


def read_plate(image):
    """Return the first confident (>0.4) text read from ``image``, or None."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (_bbox, text, prob) in _get_reader().readtext(gray):
        if prob > 0.4:
            return text
    return None
