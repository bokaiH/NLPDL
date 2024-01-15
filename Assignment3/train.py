import logging
from dataclasses import dataclass, field
import numpy as np
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score
from dataHelper import get_dataset
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from adapters.adapters import load_adapter_model

'''
	initialize logging, seed, argparse...
'''
# HfArgumentParser
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    add_adapter: int = field(
        default=0,
    )

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use"}
    )
    
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
logger.info("Training/evaluation parameters %s", training_args)

# seed
set_seed(training_args.seed)

# load datasets
dataset = get_dataset(data_args.dataset_name, '<sep>')


# load models
# AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
num_labels = len(set(dataset['train']['label']))
config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path)
if model_args.add_adapter:
    model = load_adapter_model(model)


# use datasets.map and tokenzier to process the dataset, process the text string into tokenized ids. 
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# use evaluate library (you need to compute micro_f1, macro_f1, accuracy)
wandb.init(project='Training Scripts')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = {
        'f1_micro' : f1_score(labels, predictions, average='micro'),
        'f1_macro' : f1_score(labels, predictions, average='macro'),
        'accuracy' : accuracy_score(labels, predictions)
    }
    wandb.log(metrics)
    return metrics

# DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# training!
# Use wandb to track your experiments. 
# You can use the following code to initialize wandb.
wandb.watch(model)

trainer.train()
trainer.evaluate()

