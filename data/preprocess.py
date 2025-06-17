import re

def preprocess_text(text : str):

    to_be_replaced = {
        '|': '[BR]',
        'Generic_School': '[GEN_SCHOOL]',
        'Generic_school': '[GEN_SCHOOL]',
        'SCHOOL_NAME': '[SCHOOL_NAME]',
        'STUDENT_NAME': '[STU_NAME]',
        'Generic_Name': '[GEN_NAME]',   
        'Genric_Name': '[GEN_NAME]',
        'Generic_City': '[GEN_CITY]',
        'LOCATION_NAME': '[LOC_NAME]',
        'HOTEL_NAME': '[HOTEL_NAME]',
        'LANGUAGE_NAME': '[LANG_NAME]',
        'PROPER_NAME': '[PROPER_NAME]',
        'OTHER_NAME': '[OTHER_NAME]',
        'PROEPR_NAME': '[PROPER_NAME]',
        'RESTAURANT_NAME': '[RESTAURANT_NAME]',
        'STORE_NAME': '[STO_NAME]',
        'TEACHER_NAME': '[TEACHER_NAME]',
    }

    text = text.replace('\n\n', '|')

    for key, value in to_be_replaced.items():
        text = text.replace(key, value)

    #pad punctuation
    #text = re.sub('([.,!?()-])', r' \1 ', text)
    #text = re.sub('\s{2,}', ' ', text)

    return text
