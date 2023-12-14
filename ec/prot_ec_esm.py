"""
Instructions for running the code can be found in the ec/README.md

Attribution Notice:
The code in this file was adapted from https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTuning-PyTorchLightning-MS.ipynb
which was published under the Academic Free License v3.0: https://github.com/agemagician/ProtTrans/blob/master/LICENSE.md
(and which contains code from https://github.com/minimalist-nlp/lightning-text-classification )
We cite the aforementioned license below the code.
"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

import torchmetrics
from transformers import AutoTokenizer, EsmModel
from transformers import BertTokenizer, BertModel
from transformers import T5Tokenizer, T5EncoderModel
from transformer_pooling import DefaultPool,bn_drop_lin,create_head

from torchnlp.encoders import LabelEncoder
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import collate_tensors

from test_tube import HyperOptArgumentParser
from tqdm.auto import tqdm
from datetime import datetime
from collections import OrderedDict

import os,re,json, requests
import pandas as pd
import logging as log
import numpy as np

from pytorch_lightning.plugins import DDPPlugin

class EC_dataset():
    """ Load dataset from *.npy files. """
    def  __init__(self) -> None:
        self.preprocess_dataset()

    def preprocess_dataset(self):
        """ Convert *.npy files to *.json lines."""
        ids = np.load("/data/ID.npy",allow_pickle=True)
        tok = np.load("/data/tok.npy",allow_pickle=True)
        tok_itos = np.load("/data/tok_itos.npy",allow_pickle=True)
        label = np.load("/data/label.npy",allow_pickle=True)
        train = np.load("/data/train_IDs.npy",allow_pickle=True)
        val = np.load("/data/val_IDs.npy",allow_pickle=True)
        test = np.load("/data/test_IDs.npy",allow_pickle=True)

        X = {"train":train, "valid":val, "test":test}
        for k, data in X.items():
            c = []
            for i in tqdm(data):
                d = {"sequence": "".join(tok_itos[tok[i]][1:]),
                     "name": ids[i], 
                     "label": float(label[i])}
                c.append(d)

            with open(f'/data/{k}.json', 'w') as f:
                json.dump(c,f)
                
    def load_dataset(self, path):
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        df=pd.DataFrame(data[0], columns=['sequence','name','label'])        
                        
        seq = list(df['sequence'])
        label = list(df['label'])

        # Add space between every token, and map rare amino acids to "X"
        seq = [" ".join("".join(sample.split())) for sample in seq]
        seq = [re.sub(r"[UZOB]", "X", sample) for sample in seq]

        assert len(seq) == len(label)
        return Dataset(self.collate_lists(seq, label))

    def collate_lists(self, seq: list, label: list) -> dict:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seq)):
            collated_dataset.append({"seq": str(seq[i]), "label": str(label[i])})
        return collated_dataset


class ProteinClassifier(pl.LightningModule):
    """ Adapted from https://github.com/minimalist-nlp/lightning-text-classification.git  """

    def __init__(self, hparams) -> None:
        super(ProteinClassifier, self).__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = self.hparams.batch_size

        self.model_name = "Rostlab/prot_" + self.hparams.model                
        self.dataset = EC_dataset()
        
        max_i = [2,6,65][self.hparams.ec_level]
        self.hparams.label_set = ",".join([str(float(i)) for i in range(0, max_i)])
        # Enzyme Commission (EC) number level 0 (enzyme no/yes): "0.0,1.0"
        #EC level 1 (six main classes of enzymatic reactions): "0.0,1.0,2.0,3.0,4.0,5.0"
        #EC level 2 (sub-classes of enzymatic reactions): "0.0,1.0,2.0...,63.0,64.0"

        task="binary" if max_i==2 else "multiclass"
        self.valid_acc = torchmetrics.Accuracy(task=task, num_classes=max_i)
        self.test_acc = torchmetrics.Accuracy(task=task, num_classes=max_i)

        self.__build_model()
        self.__build_loss()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        """ Init BERT/T5Encoder or ESM2 model, tokenizer, pooling strategy and classification head."""

        if(self.hparams.model=="bert_bfd"):
            self.ProtTrans = BertModel.from_pretrained(self.model_name)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.encoder_features = 1024
        elif(self.hparams.model=="t5_xl_uniref50"):
            self.ProtTrans = T5EncoderModel.from_pretrained(self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.encoder_features = 1024
        elif(self.hparams.model=="esm2_t6_8M_UR50D"):
            self.ProtTrans = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D", problem_type="single_label_classification")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.encoder_features = 320
        elif(self.hparams.model=="esm2_t33_650M_UR50D"):
            self.ProtTrans = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", problem_type="single_label_classification")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.encoder_features = 1280
        

        # Label Encoder
        self.label_encoder = LabelEncoder(self.hparams.label_set.split(","), reserved_labels=[])
        self.label_encoder.unknown_index = None

        # Pooling strategy
        if(self.hparams.pool=="default_all"):
            self.pool = DefaultPool(self.encoder_features,pool_cls=True, pool_max=True, pool_mean=True, pool_mean_sqrt=True)
        elif(self.hparams.pool=="default_cls"):
            self.pool = DefaultPool(self.encoder_features,pool_cls=True, pool_max=False, pool_mean=False, pool_mean_sqrt=False)
            
        # Classification head
        self.classification_head = create_head(self.pool.output_dim, self.label_encoder.vocab_size, lin_ftrs=[50], dropout=0.1, norm=True, act="relu", layer_norm=True)
        

    def __build_loss(self):
        """ Initialize loss function. """
        self._loss = nn.CrossEntropyLoss()

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
        attention_mask = torch.tensor(attention_mask,device=self.device)

        if(self.hparams.model=="bert_bfd"):
            token_embeddings = self.ProtTrans(input_ids, attention_mask)[0]
        elif(self.hparams.model=="t5_xl_uniref50"):
            token_embeddings = self.ProtTrans(input_ids, attention_mask).last_hidden_state
        elif(self.hparams.model=="esm2_t6_8M_UR50D"):
            token_embeddings = self.ProtTrans(input_ids, attention_mask).last_hidden_state
        elif(self.hparams.model=="esm2_t33_650M_UR50D"):
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
        Returns: Dictionary with the expected model inputs, and dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)

        inputs = self.tokenizer.batch_encode_plus(sample["seq"], add_special_tokens=True, padding=True, truncation=True, max_length=self.hparams.max_length, return_attention_mask=True)

        # Turn lists into numpy arrays already here (for upgrade to PyTorch Lightning 1.2.x)
        inputs = {k: np.array(v) for k, v in inputs.items()}
        
        if not prepare_target:
            return inputs, {}

        # Prepare target:
        try:
            targets = {"labels": self.label_encoder.batch_encode(sample["label"])}
            return inputs, targets
        except RuntimeError:
            print(sample["label"])
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> torch.tensor:
        """ 
        Runs one training step, i.e. forward then loss function.        
        :param batch: output of dataloader. 
        :param batch_nb: integer displaying which batch this is
        Returns: Loss (and adds the metrics to the logger).
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_train = self.loss(model_out, targets)
        
        self.log("train_loss", loss_train)

        return loss_train        
        
    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns: Dictionary passed to the validation_epoch_end function.
        """

        inputs, targets = batch

        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]
        
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = self.valid_acc(labels_hat, y)
        
        self.log('val_loss', loss_val, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True, sync_dist=True)        
                
        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc,})

        return output
        
    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns: dictionary passed to the test_epoch_end function.
        """

        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_test = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]
        
        labels_hat = torch.argmax(y_hat, dim=1)

        test_acc = self.test_acc(labels_hat, y)
        self.log('test_acc', test_acc, on_step=True, on_epoch=True, sync_dist=True)
                
        output = OrderedDict({"test_loss": loss_test, "test_acc": test_acc,})

        return output                
        
    def configure_optimizers(self):
        """ Sets different learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.ProtTrans.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        if train:
            return self.dataset.load_dataset(hparams.train_json)
        elif val:
            return self.dataset.load_dataset(hparams.dev_json)
        elif test:
            return self.dataset.load_dataset(hparams.test_json)
        else:
            print('Incorrect dataset split')

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=DistributedSampler(self._train_dataset, shuffle=True),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)
        return DataLoader(
            dataset=self._dev_dataset,
            sampler=DistributedSampler(self._dev_dataset, shuffle=False),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the test set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(
            dataset=self._test_dataset,
            sampler=DistributedSampler(self._test_dataset, shuffle=False),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            pin_memory=True,
        )

    @classmethod
    def add_model_specific_args(
        cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for estimator specific arguments/hyperparameters. 
        :param parser: HyperOptArgumentParser obj
        Returns: updated parser
        """
        parser.opt_list("--model", default="bert_bfd", type=str, help="Name of pretrained ProtTrans model.", options=["bert_bfd", "t5_xl_uniref50"])
        parser.opt_list("--max_length", default=1000, type=int, help="Maximum sequence length.")
        parser.add_argument("--encoder_learning_rate", default=5e-06, type=float, help="Encoder specific learning rate.")
        parser.add_argument("--learning_rate", default=3e-05, type=float, help="Classification head learning rate.")
        parser.opt_list("--nr_frozen_epochs", default=1, type=int, help="Number of epochs encoder model kept frozen.", tunable=True, options=[0, 1, 2, 3, 4, 5])        
        parser.opt_list("--pool", default="default_all", type=str, help="Pooling type.", options=["default_all", "default_cls"])

        # Data arguments:            
        parser.opt_list("--ec_level", default=1, type=int, help="Enzyme Commission number level 0, 1, or 2.", options=[0, 1, 2])
        
        parser.add_argument("--label_set", default="", type=str, help="Classification labels set.") # The "label_set" is automatically determined depending on the EC level. Not necessary to set this manually as argument.

        parser.add_argument("--train_json", default="/data/train.json", type=str, help="Path to the file containing the train data.")
        parser.add_argument("--dev_json", default="/data/valid.json", type=str, help="Path to the file containing the dev data.")
        parser.add_argument("--test_json", default="/data/test.json", type=str, help="Path to the file containing the test data.")
        parser.add_argument("--loader_workers", default=8, type=int, help="Nr. subprocesses for data loading")
        
        return parser

# Setup the TensorBoardLogger
def setup_tensorboard_logger() -> TensorBoardLogger:
    return TensorBoardLogger(save_dir="/opt/output", version="0", name="lightning_logs")
logger = setup_tensorboard_logger()

# Project-wide arguments
parser = HyperOptArgumentParser(strategy="random_search", description="Protein classifier", add_help=True)
parser.add_argument("--seed", type=int, default=3, help="Training seed.")
parser.add_argument("--save_top_k", default=1, type=int, help="The best k models according to the quantity monitored will be saved.")

# Early stopping
parser.add_argument("--monitor", default="val_acc_epoch", type=str, help="Quantity to monitor.")
parser.add_argument("--metric_mode", default="max", type=str, help="If we want to min/max the monitored quantity.", choices=["auto", "min", "max"])
parser.add_argument("--patience", default=2, type=int, help=("Nr. epochs with no improvement after which training will be stopped."))
parser.add_argument("--min_epochs", default=1, type=int, help="Limits training to a minimum number of epochs")
parser.add_argument("--max_epochs", default=100, type=int, help="Limits training to a max number number of epochs")

# Batching
parser.add_argument("--batch_size", default=1, type=int, help="Batch size to be used.")
parser.add_argument("--accumulate_grad_batches", default=64, type=int, help=("Accumulated gradients runs K small batches of size N before backwards pass."))

# GPU/TPU arguments
parser.add_argument("--gpus", type=int, default=-1, help="How many gpus")
parser.add_argument("--tpu_cores", type=int, default=None, help="How many tpus")
parser.add_argument("--limit_train_batches", default=1.0, type=float, help=("For rapid prototyping. Ratio of training dataset to use."))
parser.add_argument("--limit_val_batches", default=1.0, type=float, help=("For rapid prototyping. Ratio of validation dataset to use."))
parser.add_argument("--limit_test_batches", default=1.0, type=float, help=("For rapid prototyping. Ratio of test dataset to use."))
parser.add_argument("--resume_from_checkpoint", default=None, type=str, help=("Path to checkpoint from previous run; for job resubmission."))

# Mixed precision
parser.add_argument("--precision", type=int, default="32", help="full precision or mixed precision mode")
parser.add_argument("--amp_level", type=str, default="O1", help="mixed precision type")
parser.add_argument("--amp_backend", type=str, default="apex", help="PyTorch AMP (native) or NVIDIA apex (apex).")

# Each LightningModule defines arguments relevant to it
parser = ProteinClassifier.add_model_specific_args(parser)
hparams = parser.parse_known_args()[0]

""" Main training routine specific for this project """
seed_everything(hparams.seed)

# Init lightning model
model = ProteinClassifier(hparams)

# Init early stopping
early_stop_callback = EarlyStopping(monitor=hparams.monitor, min_delta=0.0, patience=hparams.patience, verbose=True, mode=hparams.metric_mode)

# Init model checkpoint callback
ckpt_path = os.path.join(logger.save_dir, logger.name, f"version_{logger.version}", "checkpoints")

# Initialize model checkpoint saver
checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path + "/", filename = "{epoch}-{val_loss_epoch:.2f}-{val_acc_epoch:.2f}", save_top_k=hparams.save_top_k, verbose=True, monitor=hparams.monitor, every_n_epochs=1, mode=hparams.metric_mode)

# Init Trainer
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
    callbacks=[checkpoint_callback, early_stop_callback],
    precision=hparams.precision,
    amp_level=hparams.amp_level,
    amp_backend=hparams.amp_backend,
    deterministic=True,
    resume_from_checkpoint=hparams.resume_from_checkpoint,
)

# Start training
trainer.fit(model)
trainer.test(ckpt_path='best')


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
