from datetime import date, datetime

def processTime(start = None):
    if start == None:
        return datetime.now()
    
    else:
        duration = datetime.now() - start
        duration = duration.total_seconds()
        
        if duration // 60 == 0:
            return '{0:.2f}s'.format(duration)
        
        else:
            minutes = duration // 60
            seconds = duration % 60
            return '{0:.0f}min {1:.0f}s'.format(minutes, seconds)