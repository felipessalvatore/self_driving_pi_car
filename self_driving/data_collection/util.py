import time

def get_date():
    """
    gives you the date in form:
    day - month - year
    :rtype: str
    """
    return time.strftime('%d-%m-%Y')