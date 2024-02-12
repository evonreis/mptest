import multiprocessing as mp
import numpy as np
import time


def run():
    num_tf = 48 * 48
    rate_hz = 2**14
    time_s = 10
    samples = 16
    print(f"creating random data num_tf={num_tf} rate={rate_hz} time_s={time_s}")
    td_data = [np.random.rand(time_s * rate_hz) for i in range(num_tf)]
    poolsizes = [1,2,4,8]
    block_size = 16
    for poolsize in poolsizes:
        print(f"starting run with poolsize={poolsize} block_size={block_size}")
        pool = mp.Pool(processes=poolsize)
        start_s = time.time()
        old_results = pool.map_async(np.fft.fft, td_data[0:block_size])
        total_calcs = 0
        for block in range(1, num_tf // block_size):
            start_index = block_size * block
            new_results = pool.map_async(np.fft.fft, td_data[start_index: start_index+block_size])
            v = old_results.get()
            total_calcs += len(v)
            new_results = old_results
        v = old_results.get()
        total_calcs += len(v)
        end_s = time.time()
        print(f"{poolsize}: {end_s - start_s} to calculate {total_calcs} ffts")
        pool.close()


if __name__ ==  "__main__":
    run()