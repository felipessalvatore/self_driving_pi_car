import time

def get_date():
    """
    Gives you the date in form:
    year-month-day-minutes-second 
    :rtype: str
    """
    return time.strftime('%Y-%m-%d-%-M-%S')