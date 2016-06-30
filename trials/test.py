from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool as ThreadPool2
import time
import random
import sys



def mp(a):
    time.sleep(0.1*random.random())
    return a




def mul(a, b):
    time.sleep(2)
    pool = ThreadPool(1)
    my_list=range(a,b)
    results = pool.map(mp, my_list)
    pool.close()
    # print 'multilacation'
    print results
    return results

def plus(a, b):
    time.sleep(5)
    # print 'plus'
    return a + b


print 'Multi thread'
t1=time.time()
pool = ThreadPool(processes=5)
async_t = pool.apply_async(mul, (1,100))
async_b = pool.apply_async(plus, (1,100))
pool.close()
pool.join()
pl = async_b.get()
ml = async_t.get()  # get the return value from your function.
t2=time.time()
print (t2-t1)
# print ml
# print pl

print 'Single tread'
t1=time.time()
pl = mul(1,100)
ml = plus(1,100)
t2=time.time()
print (t2-t1)

