import time

# Function to format elapsed execution time
def format_time(elapsed_time) -> str:
    days = 0
    if elapsed_time >= 86400:
        days = int(elapsed_time / 86400)
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    return str(days) + ":" + elapsed_str
