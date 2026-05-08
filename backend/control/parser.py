import re

def parse_command(text):
    text = text.lower().strip()
    
    # Detect action
    action = None
    if "remove" in text:
        action = "remove"
    elif "replace" in text:
        action = "replace"
    elif "blur" in text:
        action = "blur"
    
    # Detect time range
    times = re.findall(r'(\d+)\s*s', text)
    start_time = int(times[0]) if len(times) > 0 else 0
    end_time = int(times[1]) if len(times) > 1 else None
    
    # Detect object description
    object_desc = None
    patterns = [
        r'(?:remove|replace|blur)\s+(.+?)\s+from',
        r'(?:remove|replace|blur)\s+(.+?)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            object_desc = match.group(1).strip()
            break
    
    return {
        "action": action,
        "object": object_desc,
        "start_time": start_time,
        "end_time": end_time
    }