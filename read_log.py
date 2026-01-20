import sys
content = open('debug_log.txt', 'r', encoding='utf-8').read()
# Find where traceback starts
if '=== ERROR ===' in content:
    idx = content.index('=== ERROR ===')
    print(content[idx:])
else:
    print("No error found, printing last 2000 chars:")
    print(content[-2000:])
