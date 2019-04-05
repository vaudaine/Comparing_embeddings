import time
import psutil

with open('memory_usage.txt', 'w') as f:
    while True:
        A = psutil.virtual_memory()[1]
        f.write(str(A//1024//1024))
        f.write(' '+str(time.asctime( time.localtime(time.time())))+'\n')
        time.sleep(1)
