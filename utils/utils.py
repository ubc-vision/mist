from datetime import datetime

def get_time_stamp():
	dateTimeObj = datetime.now()
	time_stamp = '{:02d}'.format(dateTimeObj.year%100) + \
	             '{:02d}'.format(dateTimeObj.month%100) + \
	             '{:02d}'.format(dateTimeObj.day%100) + \
	             '-' + \
	             '{:02d}'.format(dateTimeObj.hour%100) + \
	             '{:02d}'.format(dateTimeObj.minute%100) 
	return time_stamp

class tensorboard_scheduler():
    def __init__(self, eval_interval, save_interval, valid_interval,stop_time=-1):
    	self.eval_interval = eval_interval
    	self.save_interval = save_interval
    	self.valid_interval = valid_interval
    	self.start_time = datetime.now()
    	self.eval_counter = 0
    	self.save_counter = 0
    	self.valid_counter = 0
    	self.stop_time = stop_time
    def schedule(self):

    	delta_secs = (datetime.now() - self.start_time).total_seconds()
    	delta_mins = delta_secs/60

    	if self.stop_time!=-1 and delta_mins>self.stop_time:
    		return False, False, False

    	if delta_mins > self.eval_counter*self.eval_interval:
    		eval_flag = True
    		self.eval_counter = int(delta_mins/self.eval_interval) + 1
    	else: 
    		eval_flag =False

    	if delta_mins > self.save_counter*self.save_interval:
    		save_flag = True
    		self.save_counter = int(delta_mins/self.save_interval)  + 1
    	else:
    		save_flag = False

    	if delta_mins > self.valid_counter*self.valid_interval:
    		valid_flag = True
    		self.valid_counter = int(delta_mins/self.valid_interval)  + 1
    	else:
    		valid_flag = False

    	return eval_flag, save_flag, valid_flag
    def get_delta_time(self):
    	delta_secs = (datetime.now() - self.start_time).total_seconds()
    	return delta_secs


if __name__ == '__main__':
	# test scheduler
	import time
	import random
	scheduler = tensorboard_scheduler(2/60, 10/60, 20/60, 2)
	
	step = 0.4
	cur_time = 0
	for i in range(200):
		eval_flag, save_flag, valid_flag = scheduler.schedule()
		if eval_flag:
			print('eval at time {}'.format(scheduler.get_delta_time()))
			time.sleep(step/4)

		if save_flag:
			print('save at time {}'.format(scheduler.get_delta_time()))
			time.sleep(step/2)

		if valid_flag:
			print('valid at time {}'.format(scheduler.get_delta_time()))
			time.sleep(step*10)
		time.sleep(step+step*2*random.random())




