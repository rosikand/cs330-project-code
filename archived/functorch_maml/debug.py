# test script for debugging functions 

import deeplake 
import pdb
import learn2learn as l2l



tasksets = l2l.vision.benchmarks.get_tasksets('omniglot',
                                                  train_ways=5,
                                                  train_samples=1,
                                                  test_ways=5,
                                                  test_samples=2,
                                                  num_tasks=20,
                                                  root='~/testdata')

pdb.set_trace()
