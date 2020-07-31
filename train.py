from model import RadarNet, Nowcast
from dataloader import DataSet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os
from loss import ComLoss
from dataloader import get_iter_dali
import time


use_gpu=True
EVENTS= ['20170604', '20170624', '20170709','20170826','20170922','20180704','20181207','20190507','20190522','20190823','20190919']

def num_params(net):
		num_params= 0
		for param in net.parameters():
				num_params+= param.numel()

		print('Total number of parameters: %d'%num_params)

def normalizer(x):
	'''input tensor size (b,tsize,c,m,n), Apply log transform to data
		See Casper et al. 2020 MetNet
	'''
	log_transform= torch.log10(x+0.01)/4
	tangent_transform= torch.tanh(log_transform)

	return tangent_transform 

def denormalizer(x):
	'''An inverse of normalizer'''

	return torch.exp(x*4)-0.01


def main():
	global use_gpu, EVENTS
	# set up some parameters
	batch_size=2
	lr= 1e-3
	logging_path= 'logging/'
	num_epoches= 100
	epoch_to_save= 10


	# print("# of training samples: %d\n" %int(len(dataset_train)))

	model= Nowcast(use_gpu=use_gpu)
	print(model)
	num_params(model)
	criterion= ComLoss()

	# model.load_state_dict(torch.load('../logging/newest-5_8.pth'))

	if use_gpu:
		model= model.cuda()
		criterion.cuda()

	#optimizer
	optimizer= torch.optim.Adam(model.parameters(), lr=lr)
	scheduler= MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)

	#record
	writer= SummaryWriter(logging_path)

	#start training
	step= 0
	for epoch in range(num_epoches):
		start= time.time()
		scheduler.step(epoch)

		for param_group in optimizer.param_groups:
			print('learning rate %f' %param_group['lr'])

		for e, event in enumerate([EVENTS[0]]):
			dataset_train= DataSet(event=event)
			# loader_train= DataLoader(dataset= dataset_train, num_workers=8, batch_size=batch_size, shuffle=True)
			loader_train = get_iter_dali(event=EVENTS[0], batch_size=2,
                                        num_threads=8)

			for i, data in enumerate(loader_train):
				# input size: (4,10,1,200,200)
				# target size: (4,10,1,200,200)
				# ====================normal===============#
				# input_train=data[0]
				# target_train=data[1]
				# ====================DALI===============#
				data= data[0]
				input_train=data['inputs']
				target_train=data['target']
				model.train()
				model.zero_grad()
				optimizer.zero_grad()

				input_train= normalizer(input_train)
				# target_train= normalizer(target_train)
				input_train, target_train= Variable(input_train), Variable(target_train)
				if use_gpu:
					input_train, target_train= input_train.cuda(), target_train.cuda()

				out_train= model(input_train)
				loss= -criterion(target_train, out_train)

				loss.backward()
				optimizer.step()

				# training track
				model.eval()
				out_train= model(input_train)
				# output_train= torch.clamp(out_train, 0, 1)
				# print("[epoch %d/%d][event %d/%d][step %d/%d]  obj: %.4f "%(epoch+1,num_epoches,e, len(EVENTS), i+1,len(loader_train),-loss.item()))
				print("[epoch %d/%d][event %d/%d][step %d]  obj: %.4f "%(epoch+1,num_epoches,e, len(EVENTS), i+1,-loss.item()))
				if step% 10 ==0:
					writer.add_scalar('loss', loss.item())

				step+=1

		#save model
		if epoch % epoch_to_save==0:
			torch.save(model.state_dict(), os.path.join(logging_path,'net_epoch%d.pth'%(epoch+1)))
		end= time.time()
		print('One epoch costs %.2f minutes!'%((end-start)/60.))

	torch.save(model.state_dict(), os.path.join(logging_path,'newest.pth'))


if __name__=='__main__':
	main()