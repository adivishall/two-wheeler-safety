import easyocr
import cv2

reader = easyocr.Reader(['en'])

def read_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = reader.readtext(gray)

    for (bbox, text, prob) in results:
        if prob > 0.4:
            return text

    return None