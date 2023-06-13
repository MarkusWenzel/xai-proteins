"""
Instructions for running the code can be found in the go/README.md
You can train on multi-GPU, but need then to resubmit checkpoint on 1 GPU for testing to maintain test sample order.
Download the job result from the GPU-cluster and compute the GO-specific metrics on your local computer with evaluate_go.py

Attribution Notice:
Code modified from https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTuning-PyTorchLightning-MS.ipynb
published under the Academic Free License v3.0: https://github.com/agemagician/ProtTrans/blob/master/LICENSE.md
(contains code from https://github.com/minimalist-nlp/lightning-text-classification )
We cite the aforementioned license below the code.
"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin

import torchmetrics
from transformers import BertTokenizer, BertModel
from transformers import T5Tokenizer, T5EncoderModel

from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import collate_tensors

import pandas as pd
from test_tube import HyperOptArgumentParser
import os
import re
import requests
from tqdm.auto import tqdm
from datetime import datetime
from collections import OrderedDict
import logging as log
import numpy as np
import json
import shutil
from pathlib import Path

from transformer_pooling import *
from schedulers import *


class GO_dataset():
    """ Preprocess and load temporal split or CAFA3 dataset from *.npy files. """
            
    def preprocess_temporalsplit(self):        
        ids = np.load("/data/clas_go_deepgoplus_temporalsplit/ID.npy",allow_pickle=True)
        tok = np.load("/data/clas_go_deepgoplus_temporalsplit/tok.npy",allow_pickle=True)
        tok_itos = np.load("/data/clas_go_deepgoplus_temporalsplit/tok_itos.npy",allow_pickle=True)
        label = np.load("/data/clas_go_deepgoplus_temporalsplit/label.npy",allow_pickle=True)
        train = np.load("/data/clas_go_deepgoplus_temporalsplit/train_IDs.npy",allow_pickle=True)
        test = np.load("/data/clas_go_deepgoplus_temporalsplit/val_IDs.npy",allow_pickle=True) # test=val!
        #test = np.load("/data/clas_go_deepgoplus_temporalsplit/test_IDs.npy",allow_pickle=True) # Empty

        # Test split
        a = []
        for i in test:
            b = {'sequence': "".join(tok_itos[tok[i]][1:]),
                'name': ids[i],
                'label': "".join([str(s) for s in label[i]])}
            a.append(b)
        df_test = pd.DataFrame(a)
        df_test.to_json('/data/test.json', orient='records')

        # Random train and validation split. Like in DeepGOPlus: "In general, all parameters were tuned depending on their performance on a validation set which is a randomly split 10% of our training set." ( https://doi.org/10.1093/bioinformatics/btz595 )
        a = []
        for i in train:
            b = {'sequence': "".join(tok_itos[tok[i]][1:]),
                'name': ids[i],
                'label': "".join([str(s) for s in label[i]])}
            a.append(b)
        df_train = pd.DataFrame(a)

        df_valid_random = df_train.sample(frac=0.1, random_state=42)
        df_train_random = df_train.drop(df_valid_random.index)
        df_valid_random.to_json('/data/valid.json', orient='records') 
        df_train_random.to_json('/data/train.json', orient='records')  
               
               
    def preprocess_cafa3(self):        
        ids = np.load("/data/clas_go_deepgoplus_cafa/ID.npy",allow_pickle=True)
        tok = np.load("/data/clas_go_deepgoplus_cafa/tok.npy",allow_pickle=True)
        tok_itos = np.load("/data/clas_go_deepgoplus_cafa/tok_itos.npy",allow_pickle=True)
        label = np.load("/data/clas_go_deepgoplus_cafa/label.npy",allow_pickle=True)
        train = np.load("/data/clas_go_deepgoplus_cafa/train_IDs.npy",allow_pickle=True)
        val = np.load("/data/clas_go_deepgoplus_cafa/val_IDs.npy",allow_pickle=True)
        test = np.load("/data/clas_go_deepgoplus_cafa/test_IDs.npy",allow_pickle=True) # Here, we have three splits including test

        X = {"train":train, "valid":val, "test":test}
        for k, data in X.items():
            a = []
            for i in tqdm(data):
                b = {"sequence": "".join(tok_itos[tok[i]][1:]),
                    "name": ids[i], 
                    "label": "".join([str(s) for s in label[i]])}
                a.append(b)

            with open(f'/data/{k}.json', 'w') as f:
                json.dump(a,f)

    def load_dataset(self, path):
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        df=pd.DataFrame(data[0], columns=["sequence","name","label"])
                                
        seq = list(df["sequence"])
        label = list(df["label"])

        # Add space between every token, and map rare amino acids to "X"
        seq = [" ".join("".join(sample.split())) for sample in seq]
        seq = [re.sub(r"[UZOB]", "X", sample) for sample in seq]
        
        assert len(seq) == len(label)
        return Dataset(self.collate_lists(seq, label))

    def collate_lists(self, seq: list, label: list) -> dict:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seq)):
            collated_dataset.append({"seq": str(seq[i]), "label": torch.tensor( np.array(list(label[i])).astype(float))})
        return collated_dataset


class ProteinClassifier(pl.LightningModule):
    """ Adapted from https://github.com/minimalist-nlp/lightning-text-classification.git  """

    def __init__(self, hparams) -> None:
        super(ProteinClassifier, self).__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = self.hparams.batch_size

        self.model_name = "Rostlab/prot_" + self.hparams.model 
        self.dataset = GO_dataset()
         
        # Select either temporal split or CAFA3
        if self.hparams.datasetfile == "clas_go_deepgoplus_temporalsplit.tar.gz":
            #self.dataset.preprocess_temporalsplit()
            self.label_dims = 5101
        elif self.hparams.datasetfile == "clas_go_deepgoplus_cafa.tar.gz":
            #self.dataset.preprocess_cafa3()
            self.label_dims = 5220
        
        # Copy label_itos.npy in output folder for later evaluation
        from_file = Path("/data", self.hparams.datasetfile.split(sep=".")[0], "label_itos.npy")
        to_file = Path("/opt", "output", "label_itos.npy")
        #shutil.copy(str(from_file), str(to_file))
        
        
        # The final metrics are computed on a local machine later. Accuracy is not the final metric.
        self.valid_acc = torchmetrics.Accuracy(task="multilabel",num_labels=self.label_dims)
        self.test_acc = torchmetrics.Accuracy(task="multilabel",num_labels=self.label_dims)

        self.__build_model()
        self.__build_loss()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
            
        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        """ Init BERT/T5 model, tokenizer, pooling strategy and classification head."""
        
        # Pretrained model and tokenizer
        if(self.hparams.model=="bert_bfd"):
            self.ProtTrans = BertModel.from_pretrained(self.model_name)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        elif(self.hparams.model=="t5_xl_uniref50"):
            self.ProtTrans = T5EncoderModel.from_pretrained(self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
        
        self.encoder_features = 1024
        
        # Pooling strategy
        if(self.hparams.pool=="default_all"):
            self.pool = DefaultPool(self.encoder_features,pool_cls=True, pool_max=True, pool_mean=True, pool_mean_sqrt=True)
        elif(self.hparams.pool=="default_cls"):
            self.pool = DefaultPool(self.encoder_features,pool_cls=True, pool_max=False, pool_mean=False, pool_mean_sqrt=False)
            
        # Classification head
        self.classification_head = create_head(self.pool.output_dim, self.label_dims, lin_ftrs=[self.label_dims*2], dropout=0.1, norm=True, act="relu", layer_norm=True)

    def __build_loss(self):
        """ Initialize loss function. For Gene-Ontology-classification, we use binary cross entropy (BCE) loss instead of cross entropy loss, because of the multi-label-classification. The BCE loss version "BCEWithLogitsLoss" includes sigmoid.
        """
        self._loss = nn.BCEWithLogitsLoss()
        
    def unfreeze_encoder(self) -> None:
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.ProtTrans.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        for param in self.ProtTrans.parameters():
            param.requires_grad = False
        self._frozen = True
            
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        """ PyTorch forward function. Returns dictionary with model outputs. """
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask= torch.ones_like(input_ids) # MW: Very new.
        attention_mask = torch.tensor(attention_mask,device=self.device)

        if(self.hparams.model=="bert_bfd"):
            token_embeddings = self.ProtTrans(input_ids, attention_mask)[0]
        elif(self.hparams.model=="t5_xl_uniref50"):
            token_embeddings = self.ProtTrans(input_ids, attention_mask).last_hidden_state
        
        output = self.pool(token_embeddings,attention_mask)
        return {"logits": self.classification_head(output)}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """ Compute loss value according to previously defined loss function. """
        return self._loss(predictions["logits"], targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """        
        sample = collate_tensors(sample)
        inputs = self.tokenizer.batch_encode_plus(sample["seq"], add_special_tokens=True, padding=True, truncation=True, max_length=self.hparams.max_length, return_attention_mask=True)

        # Turn lists into numpy arrays already here (for upgrade to PyTorch Lightning 1.2.x)
        inputs = {k: np.array(v) for k, v in inputs.items()}

        if not prepare_target: return inputs, {}

        try:
            targets = {"labels": sample["label"]}
            return inputs, targets
        except RuntimeError:
            print(sample["label"])
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> torch.tensor:
        """ 
        Runs one training step, i.e. forward then loss function.        
        :param batch: output of dataloader. 
        :param batch_nb: integer displaying which batch this is
        Returns: loss (and adds the metrics to the lightning logger).
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)       
        loss_train = self.loss(model_out, targets)        
        self.log("train_loss", loss_train)
        return loss_train

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)
        y = targets["labels"]
        y = y.to(torch.int)
        y_hat = model_out["logits"]
        y_hat = torch.sigmoid(y_hat)
        val_acc = self.valid_acc(torch.round(y_hat), y)        
        self.log('val_loss', loss_val, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True, sync_dist=True)
        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc,})
        return output            

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_test = self.loss(model_out, targets)
        y = targets["labels"]
        y = y.to(torch.int)  
        # Rounding probabilities to binary multilabel is required to compute the accuracy
        y_hat = model_out["logits"]
        y_hat = torch.sigmoid(y_hat)
        test_acc = self.test_acc(torch.round(y_hat), y)        
        self.log('test_loss', loss_test, on_step=True, on_epoch=True, sync_dist=True)        
        self.log("test_acc", test_acc, on_step=True, on_epoch=True, sync_dist=True)
        output = OrderedDict({"y": y, "y_hat": y_hat,})
        return output

    def test_epoch_end(self, outputs: list):# -> dict:
        """ Function that takes as input a list of dictionaries returned by the test_step
        function and measures the model performance accross the entire test set.
        """
        y_all = torch.cat([x["y"] for x in outputs])
        y_hat_all = torch.cat([x["y_hat"] for x in outputs])

        y_all=self.all_gather(y_all) # Gather over all GPUs
        y_hat_all=self.all_gather(y_hat_all) # Gather over all GPUs
        
        # Attention: We use 1 GPU for testing. With 2,3,4 etc. GPUs the order of the test samples/predictions would be changed and the final (local) evaluation would not work!
        #torch.save(y_all, '/opt/output/y_all.pt')
        torch.save(y_all, 'y_all.pt')
        #torch.save(y_hat_all, '/opt/output/y_hat_all.pt')
        torch.save(y_hat_all, 'y_hat_all.pt')

    def configure_optimizers(self):
        """ Sets different learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters(), "name": "classification_head"},
            {"params": self.ProtTrans.parameters(), "lr": self.hparams.encoder_learning_rate, "name": "encoder_learning_rate"},
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        lr_dict = {"scheduler": get_cosine_schedule_with_warmup(optimizer, hparams.num_warmup_steps, hparams.num_training_steps),
            "interval": "step", "name": "cosine_schedule_with_warmup"}
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        if train: return self.dataset.load_dataset(hparams.train_json)
        elif val: return self.dataset.load_dataset(hparams.dev_json)
        elif test: return self.dataset.load_dataset(hparams.test_json)
        else: print('Incorrect dataset split')

    def train_dataloader(self) -> DataLoader:
        """ Loads training data set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(dataset=self._train_dataset, 
            #sampler=RandomSampler(self._train_dataset),
            sampler=DistributedSampler(self._train_dataset, shuffle=True),
            batch_size=self.hparams.batch_size, collate_fn=self.prepare_sample, num_workers=self.hparams.loader_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        """ Loads validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)
        return DataLoader(dataset=self._dev_dataset, 
            sampler=DistributedSampler(self._dev_dataset, shuffle=False), # MW: Very new: ", shuffle=False"
            batch_size=self.hparams.batch_size, collate_fn=self.prepare_sample, num_workers=self.hparams.loader_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        """ Loads test set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(dataset=self._test_dataset, 
            sampler=DistributedSampler(self._test_dataset, shuffle=False), # MW: Very new: ", shuffle=False"
            batch_size=self.hparams.batch_size, collate_fn=self.prepare_sample, num_workers=self.hparams.loader_workers, pin_memory=True)

    @classmethod
    def add_model_specific_args(cls, parser: HyperOptArgumentParser) -> HyperOptArgumentParser:
        """ Parser for estimator specific arguments/hyperparameters.""" 
        parser.opt_list("--model", default="bert_bfd", type=str, help="Choose pretrained model.", options=["bert_bfd", "t5_xl_uniref50"])
        parser.opt_list("--max_length", default=1000, type=int, help="Maximum sequence length.")
        parser.add_argument("--encoder_learning_rate", default=5e-06, type=float, help="Encoder specific learning rate.")
        parser.add_argument("--learning_rate", default=3e-05, type=float, help="Classification head learning rate.")
        parser.opt_list("--nr_frozen_epochs", default=1, type=int, help="Number of epochs we keep encoder model frozen.", tunable=True, options=[0, 1, 2, 3, 4, 5])
        parser.opt_list("--pool", default="default_all", type=str, help="Pooling type.", options=["default_all", "default_cls"])

        # Data arguments:      
        parser.add_argument("--train_json", default="/data/clas_go_deepgoplus_temporalsplit/train.json", type=str, help="Path to train data.")
        parser.add_argument("--dev_json", default="/data/clas_go_deepgoplus_temporalsplit/valid.json", type=str, help="Path to dev data.")
        parser.add_argument("--test_json", default="/data/clas_go_deepgoplus_temporalsplit/test.json", type=str, help="Path to test data.")
        parser.add_argument("--loader_workers", default=8, type=int, help="Subprocesses to use for data loading.")
        parser.add_argument("--gradient_checkpointing", default=True, type=bool,help="Enable or disable gradient checkpointing.")
        parser.opt_list("--datasetfile",  default="clas_go_deepgoplus_temporalsplit.tar.gz", type=str, help="Choose temporal or CAFA3 split", options=["clas_go_deepgoplus_temporalsplit.tar.gz","clas_go_deepgoplus_cafa.tar.gz"])

        return parser

# Setup the TensorBoardLogger # MW: VERY NEW
def setup_tensorboard_logger() -> TensorBoardLogger:
    #return TensorBoardLogger(save_dir="/opt/output", version="0", name="lightning_logs")
    return TensorBoardLogger(save_dir="", version="0", name="lightning_logs")
logger = setup_tensorboard_logger()


parser = HyperOptArgumentParser(strategy="random_search", description="Protein classifier", add_help=True)
parser.add_argument("--seed", type=int, default=3, help="Training seed.")

# Early stopping
parser.add_argument("--save_top_k", default=1, type=int, help="The best k models according to the quantity monitored will be saved.")
parser.add_argument("--save_last", type=bool, default=False, help="Safe last checkpoint.")
parser.add_argument("--monitor", default="val_loss_epoch", type=str, help="Quantity to monitor.") 
parser.add_argument("--metric_mode", default="min", type=str, help="If we want to min/max the monitored quantity.", choices=["auto", "min", "max"])
parser.add_argument("--patience", default=2, type=int, help=("Number of epochs with no improvement after which training will be stopped."))
parser.add_argument("--min_epochs", default=1, type=int, help="Limits training to a minimum number of epochs")
parser.add_argument("--max_epochs", default=100, type=int, help="Limits training to a maximum number of epochs")

# Batching
parser.add_argument("--batch_size", default=1, type=int, help="Batch size to be used.")
parser.add_argument("--accumulate_grad_batches", default=64, type=int, help=("Accumulated gradients runs K small batches of size N before doing a backwards pass."))

# Learning rate scheduling
parser.add_argument("--num_warmup_steps", default=500, type=int, help="The number of steps for the warmup phase.")
parser.add_argument("--num_training_steps", default=20000, type=int, help="The total number of training steps.")        
            
# GPU/TPU arguments (-1 means all)
parser.add_argument("--gpus", type=int, default=-1, help="How many gpus")
parser.add_argument("--tpu_cores", type=int, default=None, help="How many tpus")

# For rapid prototyping: use part of data only.
parser.add_argument("--limit_train_batches", default=1.0, type=float, help=("Ratio of training data to use."))
parser.add_argument("--limit_val_batches", default=1.0, type=float, help=("Ratio of validation data to use."))
parser.add_argument("--limit_test_batches", default=1.0, type=float, help=("Ratio of test data to use."))

# Resubmit job based on checkpoint from previous run.
parser.add_argument("--resume_from_checkpoint", default=None, type=str, help=("Path to checkpoint from previous run."))

# Precision
parser.add_argument("--precision", type=int, default="32", help="full or mixed precision mode")
parser.add_argument("--amp_level", type=str, default="O1", help="mixed precision type")
parser.add_argument("--amp_backend", type=str, default="apex", help="PyTorch AMP (native) or NVIDIA apex (apex).")

parser = ProteinClassifier.add_model_specific_args(parser)
hparams = parser.parse_known_args()[0]

# Main training routine
seed_everything(hparams.seed)

# Init model
model = ProteinClassifier(hparams)

# Early stopping
early_stop_callback = EarlyStopping(monitor=hparams.monitor, min_delta=0.0, patience=hparams.patience, verbose=True, mode=hparams.metric_mode)

# Checkpoints
ckpt_path = os.path.join(logger.save_dir, logger.name, f"version_{logger.version}", "checkpoints")
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_path + "/",
    filename = "{epoch}-{val_loss_epoch:.2f}-{val_acc_epoch:.2f}",
    save_top_k=hparams.save_top_k,
    verbose=True,
    monitor=hparams.monitor,
    every_n_epochs=1,
    mode=hparams.metric_mode,
)

# Log learning rate to tensorboard and stdout
lr_monitor = LearningRateMonitor(logging_interval="step")
lr_monitor2 = LRMonitorCallback(start=False,end=True, interval="step")

trainer = Trainer(
    gpus=hparams.gpus,
    tpu_cores=hparams.tpu_cores,
    logger=logger,
    strategy='ddp_sharded',
    max_epochs=hparams.max_epochs,
    min_epochs=hparams.min_epochs,
    accumulate_grad_batches=hparams.accumulate_grad_batches,
    limit_train_batches=hparams.limit_train_batches,
    limit_val_batches=hparams.limit_val_batches,
    limit_test_batches=hparams.limit_test_batches,
    callbacks=[lr_monitor, lr_monitor2, checkpoint_callback, early_stop_callback],
    precision=hparams.precision,
    amp_level=hparams.amp_level,
    amp_backend=hparams.amp_backend,
    deterministic=True,
    resume_from_checkpoint=hparams.resume_from_checkpoint,
)

#trainer.fit(model)
#trainer.test(ckpt_path='best')


"""
Academic Free License ("AFL") v. 3.0

This Academic Free License (the "License") applies to any original work of authorship (the "Original Work") whose owner (the "Licensor") has placed the following licensing notice adjacent to the copyright notice for the Original Work:

 Licensed under the Academic Free License version 3.0

    Grant of Copyright License. Licensor grants You a worldwide, royalty-free, non-exclusive, sublicensable license, for the duration of the copyright, to do the following:

    a) to reproduce the Original Work in copies, either alone or as part of a collective work;
    b) to translate, adapt, alter, transform, modify, or arrange the Original Work, thereby creating derivative works ("Derivative Works") based upon the Original Work;
    c) to distribute or communicate copies of the Original Work and Derivative Works to the public, under any license of your choice that does not contradict the terms and conditions, including Licensor's reserved rights and remedies, in this Academic Free License;
    d) to perform the Original Work publicly; and
    e) to display the Original Work publicly.

    Grant of Patent License. Licensor grants You a worldwide, royalty-free, non-exclusive, sublicensable license, under patent claims owned or controlled by the Licensor that are embodied in the Original Work as furnished by the Licensor, for the duration of the patents, to make, use, sell, offer for sale, have made, and import the Original Work and Derivative Works.

    Grant of Source Code License. The term "Source Code" means the preferred form of the Original Work for making modifications to it and all available documentation describing how to modify the Original Work. Licensor agrees to provide a machine-readable copy of the Source Code of the Original Work along with each copy of the Original Work that Licensor distributes. Licensor reserves the right to satisfy this obligation by placing a machine-readable copy of the Source Code in an information repository reasonably calculated to permit inexpensive and convenient access by You for as long as Licensor continues to distribute the Original Work.

    Exclusions From License Grant. Neither the names of Licensor, nor the names of any contributors to the Original Work, nor any of their trademarks or service marks, may be used to endorse or promote products derived from this Original Work without express prior permission of the Licensor. Except as expressly stated herein, nothing in this License grants any license to Licensor's trademarks, copyrights, patents, trade secrets or any other intellectual property. No patent license is granted to make, use, sell, offer for sale, have made, or import embodiments of any patent claims other than the licensed claims defined in Section 2. No license is granted to the trademarks of Licensor even if such marks are included in the Original Work. Nothing in this License shall be interpreted to prohibit Licensor from licensing under terms different from this License any Original Work that Licensor otherwise would have a right to license.

    External Deployment. The term "External Deployment" means the use, distribution, or communication of the Original Work or Derivative Works in any way such that the Original Work or Derivative Works may be used by anyone other than You, whether those works are distributed or communicated to those persons or made available as an application intended for use over a network. As an express condition for the grants of license hereunder, You must treat any External Deployment by You of the Original Work or a Derivative Work as a distribution under section 1(c).

    Attribution Rights. You must retain, in the Source Code of any Derivative Works that You create, all copyright, patent, or trademark notices from the Source Code of the Original Work, as well as any notices of licensing and any descriptive text identified therein as an "Attribution Notice." You must cause the Source Code for any Derivative Works that You create to carry a prominent Attribution Notice reasonably calculated to inform recipients that You have modified the Original Work.

    Warranty of Provenance and Disclaimer of Warranty. Licensor warrants that the copyright in and to the Original Work and the patent rights granted herein by Licensor are owned by the Licensor or are sublicensed to You under the terms of this License with the permission of the contributor(s) of those copyrights and patent rights. Except as expressly stated in the immediately preceding sentence, the Original Work is provided under this License on an "AS IS" BASIS and WITHOUT WARRANTY, either express or implied, including, without limitation, the warranties of non-infringement, merchantability or fitness for a particular purpose. THE ENTIRE RISK AS TO THE QUALITY OF THE ORIGINAL WORK IS WITH YOU. This DISCLAIMER OF WARRANTY constitutes an essential part of this License. No license to the Original Work is granted by this License except under this disclaimer.

    Limitation of Liability. Under no circumstances and under no legal theory, whether in tort (including negligence), contract, or otherwise, shall the Licensor be liable to anyone for any indirect, special, incidental, or consequential damages of any character arising as a result of this License or the use of the Original Work including, without limitation, damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses. This limitation of liability shall not apply to the extent applicable law prohibits such limitation.

    Acceptance and Termination. If, at any time, You expressly assented to this License, that assent indicates your clear and irrevocable acceptance of this License and all of its terms and conditions. If You distribute or communicate copies of the Original Work or a Derivative Work, You must make a reasonable effort under the circumstances to obtain the express assent of recipients to the terms of this License. This License conditions your rights to undertake the activities listed in Section 1, including your right to create Derivative Works based upon the Original Work, and doing so without honoring these terms and conditions is prohibited by copyright law and international treaty. Nothing in this License is intended to affect copyright exceptions and limitations (including "fair use" or "fair dealing"). This License shall terminate immediately and You may no longer exercise any of the rights granted to You by this License upon your failure to honor the conditions in Section 1(c).

    Termination for Patent Action. This License shall terminate automatically and You may no longer exercise any of the rights granted to You by this License as of the date You commence an action, including a cross-claim or counterclaim, against Licensor or any licensee alleging that the Original Work infringes a patent. This termination provision shall not apply for an action alleging patent infringement by combinations of the Original Work with other software or hardware.

    Jurisdiction, Venue and Governing Law. Any action or suit relating to this License may be brought only in the courts of a jurisdiction wherein the Licensor resides or in which Licensor conducts its primary business, and under the laws of that jurisdiction excluding its conflict-of-law provisions. The application of the United Nations Convention on Contracts for the International Sale of Goods is expressly excluded. Any use of the Original Work outside the scope of this License or after its termination shall be subject to the requirements and penalties of copyright or patent law in the appropriate jurisdiction. This section shall survive the termination of this License.

    Attorneys' Fees. In any action to enforce the terms of this License or seeking damages relating thereto, the prevailing party shall be entitled to recover its costs and expenses, including, without limitation, reasonable attorneys' fees and costs incurred in connection with such action, including any appeal of such action. This section shall survive the termination of this License.

    Miscellaneous. If any provision of this License is held to be unenforceable, such provision shall be reformed only to the extent necessary to make it enforceable.

    Definition of "You" in This License. "You" throughout this License, whether in upper or lower case, means an individual or a legal entity exercising rights under, and complying with all of the terms of, this License. For legal entities, "You" includes any entity that controls, is controlled by, or is under common control with you. For purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

    Right to Use. You may use the Original Work in all ways not otherwise restricted or conditioned by this License or by law, and Licensor promises not to interfere with or be responsible for such uses by You.

    Modification of This License. This License is Copyright © 2005 Lawrence Rosen. Permission is granted to copy, distribute, or communicate this License without modification. Nothing in this License permits You to modify this License as applied to the Original Work or to Derivative Works. However, You may modify the text of this License and copy, distribute or communicate your modified version (the "Modified License") and apply it to other original works of authorship subject to the following conditions: (i) You may not indicate in any way that your Modified License is the "Academic Free License" or "AFL" and you may not use those names in the name of your Modified License; (ii) You must replace the notice specified in the first paragraph above with the notice "Licensed under " or with a notice of your own that is not confusingly similar to the notice in this License; and (iii) You may not claim that your original works are open source software unless your Modified License has been approved by Open Source Initiative (OSI) and You comply with its license review and certification process.
"""
