import easyocr

reader = easyocr.Reader(['en'])

result = reader.readtext("plate3.jpeg")

print(result)