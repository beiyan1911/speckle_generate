from torch.utils.tensorboard import SummaryWriter
import random
import time
writer = SummaryWriter(flush_secs=1)
for i in range(10000):
    writer.add_scalar('loss', i + random.randint(0, 3))
    time.sleep(10)
    print(i)
    writer.flush()