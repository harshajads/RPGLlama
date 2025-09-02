import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, DataCollatorForLanguageModeling
import json
from datasets import Dataset, load_from_disk
from sample_generator import Sample
from peft import LoraConfig, get_peft_model

# Load your JSON data
class Preprocessor:
    def __init__(self):
        self.training_dataset = "C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\sample_instance.json"
        # Data cluster collector. When tuner is called these data will be copied to temporary_dataset
        self.formatted_data_location = "C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\formatted_dataset"
        self.tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\tinyllama")
        self.tokenized_dataset_location = "C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\tokenized_dataset"
        self.main_dataset = "C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\dataset.json"
        # The collection of all the data gathered
        self.temporary_dataset = "C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\temporary_dataset.json"
        # File to be used while fine tuning the model
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_json_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data

    # Format for instruction tuning
    def format_example(self, example):
        return {
            "sample": f"Generate the map for level {example['prompt']}\n\n### Response:\n{example['map']}"
        }

    def save_formatted_data(self):
        data = self.load_json_data(self.temporary_dataset)
        formatted_data = [self.format_example(item) for item in data]
        dataset = Dataset.from_list(formatted_data)
        # Save the formatted dataset
        dataset.save_to_disk(self.formatted_data_location)
        print("Formatted dataset saved to 'formatted_dataset' directory")

    #-------------------------------------------------------------------------------------------------------------
    #Tokenizer Functions
    #-------------------------------------------------------------------------------------------------------------
    def tokenize_function(self, examples):
        # Tokenize the entire text
        tokenized = self.tokenizer(
            examples["sample"],
            truncation=True,
            max_length=1024,  # Adjust based on your longest map
            padding=True,
            return_tensors=None
        )

        # For causal LM, labels are the same as input_ids
        # The model learns to predict the next token in the sequence
        tokenized["labels"] = tokenized["input_ids"].copy()
        # Convert all lists to tensors
        for key in tokenized:
            tokenized[key] = torch.tensor(tokenized[key])  # Keep as lists, HuggingFace handles conversion

        return tokenized

    def tokenize(self):
        loaded_dataset = load_from_disk(self.formatted_data_location)
        print("Loaded successfully")
        tokenized_dataset = loaded_dataset.map(
            self.tokenize_function,
            remove_columns=loaded_dataset.column_names,
            batched=True
        )
        tokenized_dataset.save_to_disk(self.tokenized_dataset_location)
        print("Tokenized dataset saved to 'tokenized_dataset' directory")

        self.remove_temporary_dataset()
        print("Removed temporary dataset after tokenization")

    def test_tokenized_dataset(self):
        tokenized_dataset = load_from_disk(self.tokenized_dataset_location)
        print("Tokenized dataset loaded from 'tokenized_dataset' directory")
        # Verify a sample
        sample = tokenized_dataset[0]
        print("Sample input IDs:", sample['input_ids'][:20])
        print("Sample decoded:")
        print(self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True))

    def import_to_main_dataset(self):
        # Read from source file and append to destination file
        try:
            source_file = open(self.training_dataset, 'r')
            data = source_file.read()
            source_file.close()

            # Data in the training_dataset will be copied to 2 locations as following.
            # Then it is going to be deleted and reused from the beginning with g.new() call
            temporary_file = open(self.temporary_dataset, 'w')
            temporary_file.write(data)
            temporary_file.close()

            dest_file = open(self.main_dataset, 'a')
            dest_file.write(data)
            dest_file.close()

            os.remove(self.training_dataset)
        except FileNotFoundError:
            print("File has been removed an not created back yet.")
    def remove_temporary_dataset(self): # After tokenization, temporary data file will be deleted
        os.remove(self.temporary_dataset)




#Test Case
"""s = Sample("Hi", "How are you")
print("Model Training:",s.can_train_model())
p = Preprocessor()
p.import_to_main_dataset()

if s.can_train_model():
    p = Preprocessor()
    p.save_formatted_data()
    p.tokenize()
    p.test_tokenized_dataset()
"""






