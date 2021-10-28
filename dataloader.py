import numpy as np 
import pandas as pd 
from glob import glob
from scipy.stats import zscore
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import torch
import sys

BATCH_SIZE = 64
TRAIN_SPLIT = 0.8
PAIRS = ['EURUSD', 'GBPUSD']
TARGET_PAIR = 'EURUSD'
TIMEFRAME = '60'
FORECAST_DIST = 30
WINDOW_SIZE = 30
OVERLAP = WINDOW_SIZE/2
PROFIT_LABEL = 100
DATA_PATH = 'dataset/'
HEADERS_ALL = ['date','time','open','high','low','close', 'volume']

HEADERS = [HEADERS_ALL[i] for i in [2,6]]



def merge_date_time(df, pair): # converts str date and time to datetime and merges them together in one col
	df = df.rename(columns={f'date_{pair}': 'date', f'time_{pair}': 'time'})
	df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
	df['date'] = (df['date'].view(int)/((10**9)*3600)).view(int) # the timesteps will be shown in hours
	df = df.rename(columns={'time': f'time_{pair}'})
	df = drop_cols(df,pair)
	return df

def drop_cols(df, pair):
	cols = ['time', 'open', 'high', 'low']
	cols = [col + f'_{pair}' for col in cols]
	df.drop(cols, axis='columns', inplace=True)
	return df

def clean_nan(df, droptime=False):
	if droptime:
		df.drop('date', axis='columns', inplace=True)
	# df = df.fillna(method='bfill')
	df.dropna(inplace=True)
	return df

def get_dataset(PAIRS):
	for idx, pair in enumerate(PAIRS):
		file = glob(DATA_PATH + pair + '/' + pair + TIMEFRAME  +'.csv')
		df = pd.read_csv(file[0], usecols = [i for i in range(0,7)], names=[header + f'_{pair}' for header in HEADERS_ALL])
		df = merge_date_time(df, pair)
		if idx == 0:
			history = df
		else:
			history = pd.merge_asof(history, df, on='date')
	
	history = clean_nan(history, droptime=True)

	return history


def prepare_targets(df, target_pair):
	target = df.shift(FORECAST_DIST)[f'close_{target_pair}']
	target.name = 'labels'
	df = pd.concat([df, target], axis=1)
	df.dropna(inplace=True)
	df.reset_index(drop=True, inplace=True)
	labels = (df[f'close_{target_pair}'] - df['labels']) * 10000

	def match_label(x, PROFIT_LABEL):
		if x > 0 and x > PROFIT_LABEL:
			return 1
		else:
			if x < 0 and abs(x) > PROFIT_LABEL:
				return 2
		return 0

	labels = list(map(lambda x: match_label(x, PROFIT_LABEL), labels))
	df['labels'] = labels
	wait = int(len(df[df['labels'].values==0])/len(df['labels'].values)*100)/100
	buy = int(len(df[df['labels'].values==1])/len(df['labels'].values)*100)/100
	sell = int(len(df[df['labels'].values==2])/len(df['labels'].values)*100)/100
	print()
	print(wait, buy, sell)
	print()
	

	return df






history = get_dataset(PAIRS)
history = prepare_targets(history, TARGET_PAIR)


class Data(Dataset):
	
	def __init__(self, history, overlap):
		self.history = history
		self.overlap = overlap

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.item()

		window, label = self.get_window(idx)
		return window, label
		

	def __len__(self):
		num_win = (len(self.history) - OVERLAP)/(WINDOW_SIZE - OVERLAP)
		num_win = int(num_win) - 1
		return num_win

	def get_window(self, idx):
		start_idx = int(idx*OVERLAP)
		end_idx = int(start_idx+WINDOW_SIZE)
		
		window = self.history[start_idx:end_idx]
		label = window['labels'].values[-1]
		window.drop('labels', axis='columns', inplace=True)
		width = len(window.columns)
		height = WINDOW_SIZE
		
		window = np.array(window.values).reshape(width, height)
		window = self.normalization(window)
		window = torch.from_numpy(window)
		label = torch.tensor(label)
		return window, label


	def normalization(self, data):
		data = zscore(data, axis=1, ddof=1)
		return data





dataset = Data(history, OVERLAP)
indices = list(range(dataset.__len__()))

np.random.shuffle(indices)

train_idx = indices[:int(np.floor(TRAIN_SPLIT*len(indices)))]
val_idx = indices[int(np.floor(TRAIN_SPLIT*len(indices))):]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)


train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False,
							num_workers=1, sampler=train_sampler)
val_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False,
							num_workers=1, sampler=val_sampler)