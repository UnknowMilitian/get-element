import easyocr

reader = easyocr.Reader(['en'])

# Read the processed image
result = reader.readtext('images/1.png',
                       width_ths=1,
                       decoder='beamsearch',
                       contrast_ths=0.3,
                       adjust_contrast=0.7)

for (bbox, text, prob) in result:
    print(text)