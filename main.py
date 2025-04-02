import easyocr

reader = easyocr.Reader(['en', 'uz'])

# Read the processed image
result = reader.readtext('images/2.png', 
                       paragraph=True,
                       detail=0,  # Get individual text boxes
                       decoder='beamsearch',  # Better for short text
                       beamWidth=10,
                       width_ths=0.3,  # Merge closer characters
                       text_threshold=0.7)

print(result)