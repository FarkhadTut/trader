import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from itertools import count
import math
import random
from collections import namedtuple
from operator import itemgetter
import sys


import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display



DATA_PATH = "/home/fara-ubuntu/.wine/dosdevices/c:/Program Files (x86)/Alpari MT4/MQL4/Files/Export_History"
HEADERS = ['date','time','open','high','low','close', 'volume']
PAIRS = ['EURUSD',  'GBPUSD', 'USDJPY', 'USDCHF', 'GBPJPY', 'EURAUD']
MAIN_PAIR = 'EURUSD'
TESTSET = 0.2
TRAINSET = 1 - TESTSET
WINDOW = 500
SLIDE_STEP = 10
FORECAST_STEP = 15
NUM_ACTIONS = 3
lr = 0.001


df = pd.read_csv(f'{DATA_PATH}/EURUSD/EURUSD60.csv', names=HEADERS)
df = df[1:]
df = np.array(df.values)

class Dataloader():
	def __init__(self):
		self.pairs = PAIRS
		self.step = 0
		self.current_state = None
		self.current_edge_close = 0

	def header_rename(self, header, pair):
		if header != 'date' and header != 'time':
			return f'{pair}' + '_' + header
		return header

	def get_pairs(self):  # creates a dict of pairs using the dataset
		pair_dict = dict.fromkeys(self.pairs)
		for pair in self.pairs:
			df = pd.read_csv(f'{DATA_PATH}/{pair}/{pair}60.csv', names=list(map(lambda x: self.header_rename(x, pair), HEADERS)))
			df = df[1:]
			self.merge_date_time(df)
			df['date'] = df['date'].astype(int)/((10**9)*3600) # the timesteps will be shown in hours
			# df = np.array(df.values)
			pair_dict[f'{pair}'] = df

		return pair_dict


	def merge_date_time(self, df): # converts str date and time to datetime and merges them together in one col
		df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
		df.drop('time', axis='columns', inplace=True)
		return df

	def set_targets(self, dataset):
		df = dataset[f'{MAIN_PAIR}'+'_close']
		df = df.shift(FORECAST_STEP)
		df.name = 'target'
		dataset = pd.concat([dataset, df], axis=1)
		dataset.dropna(inplace=True)

		pips = dataset[f'{MAIN_PAIR}'+'_close'] - dataset['target']
		pips = pips*100000
		profit_mean = abs(pips).mean()
		profit_border = profit_mean*0.4
		print('Average return:', profit_mean)
		print('Border:', profit_border)

		def match(x, profit_border):
			if x > 0 and x > profit_border:
				return 1
			else:
				if x < 0 and abs(x) > profit_border:
					return 2
			return 0

		targets = list(map(lambda x: match(x, profit_border), pips))
		dataset['target'] = targets

		return dataset




	def pct_change(self, df):
		df = pd.DataFrame(df)
		df.columns = [*df.columns[:-1], 'target']
		for col in df.columns:
			if col != 'target':
				column = df[col].shift(1)
				df[col] = (column - df[col])/df[col]
				if int(col)%4 == 0:
					df[col] = df[col]*100
					mask = abs(df[col].values) >= 1
					df[col][mask] = 1

		df.dropna(inplace=True)
		df = np.array(df)
		return df


	def match_time(self, pair_dict): # matches the currency pairs according to their timestamps
		main_pair = pair_dict[MAIN_PAIR]
		df = main_pair
		for pair in pair_dict:
			if not pair == MAIN_PAIR:
				df = pd.merge_asof(df, pair_dict[pair], on='date')

		return df

	def clean_nan(self, pair_dict, droptime=False):
		if droptime:
			pair_dict.drop('date', axis='columns', inplace=True)
		pair_dict = pair_dict.fillna(method='bfill')
		return pair_dict

	def droptime(self, pair_dict):
		pair_dict = pair_dict.drop('date', axis='columns')
		return pair_dict

	def batch_available(self):
		pass

	def state_available(self):
		return self.step*WINDOW < dataset.shape[0]

	def get_batch(self):
		pass

	def get_state(self):
		if not self.state_available():
			self.step = 0
		a = self.step*SLIDE_STEP
		b = a + WINDOW
		self.current_state = dataset[a:b]
		target = self.current_state[:,-1][-1]
		self.current_state = np.delete(self.current_state, -1, 1)
		self.step += 1

		a = self.step*SLIDE_STEP 
		b = a + WINDOW 
		self.next_state = dataset[a:b]
		self.next_state = np.delete(self.next_state, -1, 1)
		self.current_edge_close = self.current_state[-1][3]
		self.current_state = self.pct_change(self.current_state)
		self.current_state = self.current_state.reshape((1,1,self.current_state.shape[0],self.current_state.shape[1]))

		return torch.from_numpy(np.asarray(self.current_state)).to(device), torch.from_numpy(np.asarray(target)).to(device)

	def get_next_state(self):
		a = self.step*SLIDE_STEP 
		b = a + WINDOW 
		self.next_state = dataset[a:b]
		self.next_state = np.delete(self.next_state, -1, 1)
		self.next_edge_close = self.next_state[-1][3]
		self.next_state = self.pct_change(self.next_state)
		self.next_state = self.next_state.reshape((1,1,self.next_state.shape[0], self.next_state.shape[1]))
		
		return torch.from_numpy(np.asarray(self.next_state)).to(device)

	def take_action(self, action, target):
		if action == target.item():
			return torch.tensor([1.]).to(device)
		return torch.tensor([0.]).to(device)






class Model(nn.Module):
	def __init__(self, img_height, img_width, num_actions):
		super().__init__()
		self.conv1 = nn.Conv2d(1,16,8, stride=4, padding=1)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16,32,4, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,64,3)
		self.bn3 = nn.BatchNorm2d(64)

		self.fc1 = nn.Linear(in_features = 3840, out_features = 128)
		self.denseBn1 = nn.BatchNorm1d(256)
		self.fc2 = nn.Linear(in_features = 128, out_features = 64)
		self.denseBn2 = nn.BatchNorm1d(128)
		self.fc3 = nn.Linear(in_features = 64, out_features = 64)
		self.denseBn3 = nn.BatchNorm1d(64)
		self.out = nn.Linear(in_features = 64, out_features = num_actions)

		self.init_bias()


	def init_bias(self):
		nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
		nn.init.constant_(self.conv1.bias, 0)
		nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
		nn.init.constant_(self.conv2.bias, 0)
		nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
		nn.init.constant_(self.conv3.bias, 0)

	def forward(self, t):
		t = t.float().to(device)
		t = self.bn1(self.conv1(t))
		t = self.bn2(self.conv2(t))
		t = self.bn3(self.conv3(t))
		# t = F.relu(self.conv1(t))
		# t = F.relu(self.conv2(t))
		# t = F.relu(self.conv3(t))

		t = t.flatten(start_dim=1).float()
		t = F.relu(self.fc1(t))
		t = F.relu(self.fc2(t))
		t = F.relu(self.fc3(t))


		# t = F.relu(self.fc1(t))

		t = F.relu(self.out(t))
		return t




def plot(values, moving_avg_period, rate, init_episode, frames):
	plt.figure(2)
	plt.clf()
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(values)
	moving_avg = get_moving_average(moving_avg_period, values)
	plt.plot(moving_avg)
	plt.pause(0.01)
	print("Episode: ", init_episode + len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
	print("Explor. rate: ", rate)
	print("Frames: ", frames)
	print()
	plt.savefig("/home/fara-ubuntu/Documents/FARA/dqn/breakout_plot.png")
	if is_ipython: display.clear_output(wait=True)


def plot_loss(values, moving_avg_period):
	plt.figure(3)
	plt.clf()
	plt.title('LOSS')
	plt.xlabel('Episode')
	plt.ylabel('Loss')
	plt.plot(values)
	moving_avg = get_moving_average(moving_avg_period, values)
	plt.plot(moving_avg)
	plt.pause(0.01)

	if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
	values = torch.tensor(values, dtype=torch.float)
	if len(values) >= period:
		moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
		moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
		return moving_avg
	else:
		moving_avg = torch.zeros(len(values))
		return moving_avg



def calc_accuracy(output, labels):
	# pred = torch.tensor(list(map(lambda x: 1 if x>0.5 else 0, output))).to(device)
	_, pred = torch.max(output.data, 1)

	pred_correct = len(list(filter(lambda x: x == True, labels.eq(pred).type(torch.bool))))
	accuracy = 100*pred_correct/len(pred)

	w_mask_pred = pred == 0
	w_mask_corr = labels[w_mask_pred] == 0
	w_corr = len([x for x in w_mask_corr if x == True])
	w_pred = len(labels[w_mask_pred])
	w_ground = len(labels[labels == 0])

	b_mask_pred = pred == 1
	b_mask_corr = labels[b_mask_pred] == 1
	b_corr = len([x for x in b_mask_corr if x == True])
	b_pred = len(labels[b_mask_pred])
	b_ground = len(labels[labels == 1])

	s_mask_pred = pred == 2
	s_mask_corr = labels[s_mask_pred] == 2
	s_corr = len([x for x in s_mask_corr if x == True])
	s_pred = len(labels[s_mask_pred])
	s_ground = len(labels[labels == 2])


	if w_pred == 0:
		if w_ground == 0:
			w_precision = 1
			w_recall = 1
		else:
			w_precision = 1 - w_ground/len(labels)
			w_recall = 1 - w_ground/len(labels)
	else:
		if w_ground == 0:
			w_precision = 1 - w_pred/len(labels)
			w_recall = 1 - w_pred/len(labels)
		else:
			w_precision = w_corr/w_pred
			w_recall = w_corr/w_ground

	if b_pred == 0:
		if b_ground == 0:
			b_precision = 1
			b_recall = 1
		else:
			b_precision = 1 - b_ground/len(labels)
			b_recall = 1 - b_ground/len(labels)
	else:
		if b_ground == 0:
			b_precision = 1 - b_pred/len(labels)
			b_recall = 1 - b_pred/len(labels)
		else:
			b_precision = b_corr/b_pred
			b_recall = b_corr/b_ground

	if s_pred == 0:
		if s_ground == 0:
			s_precision = 1
			s_recall = 1
		else:
			s_precision = 1 - s_ground/len(labels)
			s_recall = 1 - s_ground/len(labels)
	else:
		if s_ground == 0:
			s_precision = 1 - s_pred/len(labels)
			s_recall = 1 - s_pred/len(labels)
		else:
			s_precision = s_corr/s_pred
			s_recall = s_corr/s_ground




	recall = (w_recall+b_recall+s_recall)/3
	precision = (w_precision+b_precision+s_precision)/3
	f1 = 2*recall*precision/(recall+precision)


	predictions = [w_pred, b_pred, s_pred]
	dist = [w_corr, b_corr, s_corr]
	ground = [w_ground, b_ground, s_ground]


	print("Precision:", precision)
	print("Recall:", recall,)
	print("F1:", f1)
	print("Dist:", dist)
	print("Pred:", predictions)
	print("Ground:", ground)



	return accuracy, precision, recall, f1, dist, predictions, ground






dataloader = Dataloader()
pair_dict = dataloader.get_pairs()
dataset = dataloader.match_time(pair_dict)
dataset = dataloader.clean_nan(dataset)
dataset = dataloader.droptime(dataset)
dataset = dataloader.set_targets(dataset)
# dataset = dataloader.pct_change(dataset)




trainset = dataset[:int(len(dataset)*TRAINSET)]
testset = dataset[int(len(dataset)*TRAINSET):]



trainset = pd.concat([trainset, trainset[-12:]])
train_labels = trainset['target'].values[::500]
trainset = np.array(trainset)
trainset = np.reshape(trainset, (96,1,500,31))	
trainset = torch.from_numpy(trainset).to(device)
train_labels = torch.tensor(train_labels).to(device)


testset = pd.concat([testset, testset[-3:]])
test_labels = testset['target'].values[::500]
testset = np.array(testset)
testset = np.reshape(testset, (24,1,500,31))
testset = torch.from_numpy(testset).to(device)
test_labels = torch.tensor(test_labels).to(device)





# model = Model(500, 30, 3).to(device)
# optimizer = optim.Adam(params=model.parameters(), lr=lr)
# cost_function = nn.CrossEntropyLoss()









# output = model(trainset)
# optimizer.zero_grad()
# loss = cost_function(output, train_labels)
# loss.backward()
# optimizer.step()

# calc_accuracy(output, train_labels)

print(trainset[1][0][:,1].shape)
print(trainset[1][0][:,1])

plt.figure(2)
plt.clf()
plt.title('EURUSD60')
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(trainset[1][0].shape)
plt.pause(0.01)
plt.savefig("/home/fara-ubuntu/Documents/Dev/trader/src/plot.png")
if is_ipython: display.clear_output(wait=True)













