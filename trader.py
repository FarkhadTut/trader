from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from sklearn import svm
from glob import glob
import torch.nn as nn 
import torch.optim as optim
import torch
from dataloader import train_dataloader, val_dataloader
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trader(LightningModule):

	hyper_parameters = {
		'num_classes': 3,
		'hidden_size': 128,
		'input_size': 30,
		'num_layers': 1,
		'lr': 0.001,
		'dropout': 0.1,

	}

	def __init__(self, num_classes, hidden_size, num_layers, input_size, dropout, lr = 0.001):
		super(Trader, self).__init__()

		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.num_classes = num_classes
		self.lr = lr
		self.input_size = input_size
		self.svm = svm.SVC()
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
							num_layers=num_layers, dropout=dropout, bidirectional=False, batch_first=True)

		weights = torch.tensor([0.8, 0.095, 0.095])
		self.criterion = nn.CrossEntropyLoss(weight=weights)


		self.layer_norm2 = nn.LayerNorm(hidden_size)
		self.dropout2 = nn.Dropout(dropout)
		self.final_fc = nn.Linear(512, num_classes)
		self.gelu = nn.GELU()


	def __init_hidden(self, batch_size):
		n, hs = self.num_layers, self.hidden_size
		return (torch.zeros(n*1, batch_size, hs),
				torch.zeros(n*1, batch_size, hs))


	def forward(self, t, hidden):
		t = t.float()
		out, (hn, cn) = self.lstm(t, hidden)
		t = self.dropout2(self.gelu(self.layer_norm2(out)))
		t = t.flatten(start_dim=1)
		# print(t.shape)
		t = self.final_fc(t)
		return t, (hn, cn)

	def step(self, batch):
		t, y = batch[0], batch[1]
		# t = t.reshape((t.shape[0],t.shape[2],t.shape[1]))
		batch_size = t.shape[0]
		hidden = self.__init_hidden(batch_size)
		hn, c0 = hidden[0].to(device), hidden[1].to(device)
		t, _ = self(t, (hn, c0))

		# t = F.log_softmax(t, dim=2).to(torch.float64)
		# print(t.shape)
		loss = self.criterion(t,y)
		return loss

	def training_step(self, batch, batch_idx):
		loss = self.step(batch)
		logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr'] }
		self.log('Training Loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {'loss': loss, 'log': logs}


	def validation_step(self, batch, batch_idx):
		loss = self.step(batch)
		logs = {'val_loss': loss}
		self.log('Validation Loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {'val_loss': loss, 'logs': logs}


	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		self.scheduler.step(avg_loss)
		tensorboard_logs = {'val_loss': avg_loss}
		return {'val_loss': avg_loss, 'log': tensorboard_logs}

	def configure_optimizers(self):
		self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
										self.optimizer, mode='min',
										factor=0.5, patience=6)
		return {'optimizer': self.optimizer, 'scheduler':self.scheduler}

	def train_dataloader(self):
		return train_dataloader

	def val_dataloader(self):
		return val_dataloader

	def checkpoint_callback(self, checkpoint_path):
		return ModelCheckpoint(
			save_top_k=2,
			auto_insert_metric_name=True,
			verbose=True,
			monitor='Validation Loss',
			mode='min',
			filename='trader-{epoch}-{val_loss}',
			dirpath=checkpoint_path
			)

	def get_checkpoint_file(self, checkpoint_path):
		checkpoint_path = glob(checkpoint_path + '*.ckpt')
		if checkpoint_path == []:
			return None
		return checkpoint_path[-1]




EPOCHS = 20
checkpoint_path = 'saved_models/'




def train():
	h_params = Trader.hyper_parameters
	model = Trader(**h_params).to(device)
	logger = TensorBoardLogger('logs', name='trader')
	trainer = Trainer(logger=logger)

	trainer = Trainer(
		callbacks=[model.checkpoint_callback(checkpoint_path)],
		max_epochs=EPOCHS, gpus=1, logger=logger,
		gradient_clip_val=1.0,	checkpoint_callback=True,
		resume_from_checkpoint=model.get_checkpoint_file(checkpoint_path),
		auto_select_gpus=True, num_nodes=1
		)

	trainer.fit(model)


train()









