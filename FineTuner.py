import dataset_preprocessor as dp
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
import json
from datasets import Dataset, load_from_disk
#from Level_generator import get_prompt, create_level

class FineTuner:
    def __init__(self):
        self.trained_model = 'C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\tuned_model'
        self.model_path = "C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\tinyllama"
        self.trained_model_mask = "C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\tuned_model\\checkpoint-200"
        self.Preprocessor = dp.Preprocessor()
        self.current_checkpoint = None # Tracks the latest checkpoint

    def raw_model_test(self):
        tokenizer= self.Preprocessor.tokenizer
        # ✅ Load model in float16 directly on GPU (RTX 4050 friendly)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=None
        ).to("cuda")

        # ✅ Your prompt
        prompt = "How many hours does a day have ?"

        # ✅ Encode input
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # ✅ Generate text
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # how long the poem can be
            do_sample=True,  # randomness for creativity
            temperature=0.8,  # creativity control
            top_p=0.9  # nucleus sampling
        )

        # ✅ Decode result
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Model response:\n")
        print(response)


    """def finetune_model(self):
        model_name = self.model_path
        tokenized_dataset = self.Preprocessor.tokenized_dataset_location

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Or load in 8bit -> Defines the bit config for matrix coefficients
            bnb_4bit_use_double_quant=True,  # Applies extra round of quantization to save memory
            bnb_4bit_quant_type='nf4',  # 4bit quantization algorithm (nf4 or pf4, nf4 better for LLMs)
            bnb_4bit_compute_dtype='float16'
        )

        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')

        # Loading the tokenizer -> TinyLLama itself to tokenize the wording
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

        # --------------------------------------------------------------------------------------------------------------
        # Freezing the Original weights -> Setting W constant
        # --------------------------------------------------------------------------------------------------------------

        for param in model.parameters():
            param.requires_grad = False  # Freezing the model -> grad(W) = 0
            if param.ndim == 1:
                # Casts the small parameters (layer norm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()  # Reduces number of stored activations
        model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)

        model.lm_head = CastOutputToFloat(model.lm_head)

        # ---------------------------------------------------------------------------------------------------------------------
        # Setting up LoRA adapters
        # ---------------------------------------------------------------------------------------------------------------------
        def print_trainable_parameters(model):
            trainable_params = 0
            all_params = 0
            for _, param in model.named_parameters():
                all_params += param.numel()
                trainable_params += param.numel()
            print(f"Trainable parameters: {trainable_params}"
                  f"|| Total parameters: {all_params} || Trainable %: {trainable_params / all_params * 100:.2f}%")

        print_trainable_parameters(model)

        config = LoraConfig(
            r=8,  # Attention Heads -> No. of rows in A and columns in B
            lora_alpha=16,  # Scaling Factor
            lora_dropout=0.05,  # 5% of input embeddings will be zeroed randomly
            target_modules=["q_proj", "v_proj"], # More expressive power to adapt, 0.1 - 1% is used to train
            bias='none',  # y = Wx + b -> b=0
            task_type="CAUSAL_LM"  # Model learns to predict the next token
            # ex: John wick killed 3 people in a bar with a f**** -> "Pencil"
        )

        # ---------------------------------------------------------------------------------------------------------------------
        # Loading the dataset and model
        # ---------------------------------------------------------------------------------------------------------------------

        model = get_peft_model(model, config)
        print("Model is ready for finetuning!")
        tokenized_data = load_from_disk(tokenized_dataset)

        # ---------------------------------------------------------------------------------------------------------------------
        # Setting up trainer
        # ---------------------------------------------------------------------------------------------------------------------

        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=2,  # Processes 16 samples per GPU at once
                gradient_accumulation_steps=1,  # Updates weights after each n steps, high -> smooth, low -> noisy
                warmup_steps=1,  # gradually increases learning rate until first 100 steps -> high learning stability
                #max_steps=20,  # High -> Better learning, tendency to overfit.
                num_train_epochs=50,
                learning_rate=5e-5,
                fp16=True,  # Cuts memory usage in half
                output_dir=self.trained_model,
                logging_steps=1,  # Prints progress each and every step
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,
                                                                       pad_to_multiple_of=8)
        )
        print("Trainer Ready.")

        # ---------------------------------------------------------------------------------------------------------------------
        # Model Training
        # ---------------------------------------------------------------------------------------------------------------------

        model.config.use_cache = False
        trainer.train()

        print("Fine tuning completed and model saved!")"""

    def finetune_model(self, resume_checkpoint):
        model_name = self.model_path
        tokenized_dataset = self.Preprocessor.tokenized_dataset_location

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16'
        )

        # ----------------------------------------------------------------------------------------
        # NEW: Find latest checkpoint for resuming
        # ----------------------------------------------------------------------------------------
        checkpoints = [f for f in os.listdir(self.trained_model)
                       if f.startswith('checkpoint-') and os.path.isdir(os.path.join(self.trained_model, f))]

        if checkpoints:
            # Find the latest checkpoint by step number
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            resume_checkpoint = os.path.join(self.trained_model, latest_checkpoint)
            print(f"▶️ Found previous checkpoint: {latest_checkpoint}. Resuming training...")

            # Load base model and then the checkpoint
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map='auto'
            )
            model = PeftModel.from_pretrained(base_model, resume_checkpoint)
        elif os.path.exists(os.path.join(self.trained_model, "adapter_model.bin")):
            # ----------------------------------------------------------------------------------------
            # Load the starting model from saved adapter (your existing code)
            # ----------------------------------------------------------------------------------------
            print("▶️ Found previous adapter. Loading for incremental fine-tuning...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map='auto'
            )
            model = PeftModel.from_pretrained(base_model, self.trained_model)
        else:
            # ----------------------------------------------------------------------------------------
            # No previous adapter -> start fresh from TinyLlama base (your existing code)
            # ----------------------------------------------------------------------------------------
            print("▶️ No previous adapter found. Starting fresh LoRA fine-tuning from TinyLlama...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map='auto'
            )
            model = get_peft_model(base_model, LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias='none',
                task_type="CAUSAL_LM"
            ))

        # ----------------------------------------------------------------------------------------
        # Tokenizer setup (your existing code)
        # ----------------------------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # ----------------------------------------------------------------------------------------
        # Freeze base model weights (your existing code)
        # ----------------------------------------------------------------------------------------
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)

        model.lm_head = CastOutputToFloat(model.lm_head)

        # ----------------------------------------------------------------------------------------
        # Print trainable parameters (your existing code)
        # ----------------------------------------------------------------------------------------
        def print_trainable_parameters(model):
            trainable_params = 0
            all_params = 0
            for _, param in model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(f"Trainable parameters: {trainable_params}"
                  f" || Total parameters: {all_params} || Trainable %: {trainable_params / all_params * 100:.2f}%")

        print_trainable_parameters(model)

        # ----------------------------------------------------------------------------------------
        # Load tokenized dataset (your existing code)
        # ----------------------------------------------------------------------------------------
        tokenized_data = load_from_disk(tokenized_dataset)

        # ----------------------------------------------------------------------------------------
        # NEW: Calculate steps for 50 epochs (FIXED VERSION)
        # ----------------------------------------------------------------------------------------
        total_samples = len(tokenized_data)
        batch_size = 2  # per_device_train_batch_size
        grad_accumulation = 1  # gradient_accumulation_steps

        # Calculate steps needed for 50 epochs
        steps_per_epoch = total_samples / (batch_size * grad_accumulation)
        steps_for_50_epochs = int(steps_per_epoch * 50)

        # If resuming from checkpoint, we want to train for 50 MORE epochs
        # but we need to handle the case where we've already trained some epochs
        if checkpoints:
            current_step = int(resume_checkpoint.split('-')[1])
            # Calculate how many steps we've already completed in terms of epochs
            epochs_completed = current_step / steps_per_epoch
            # Calculate steps needed to complete 50 total epochs
            steps_needed = max(0, int(steps_for_50_epochs - current_step))

            if steps_needed > 0:
                max_steps = current_step + steps_needed
                print(f"▶️ Resuming from step {current_step} ({epochs_completed:.1f} epochs completed)")
                print(f"▶️ Training for {steps_needed} more steps to complete 50 total epochs")
            else:
                print(f"⚠️ Already completed {epochs_completed:.1f} epochs (more than 50)")
                print(f"▶️ Training for additional 50 epochs worth of steps")
                max_steps = current_step + steps_for_50_epochs
        else:
            max_steps = steps_for_50_epochs
            print(f"▶️ Starting fresh, training for {max_steps} steps (50 epochs)")

        # ----------------------------------------------------------------------------------------
        # Trainer setup with checkpoint management
        # ----------------------------------------------------------------------------------------
        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                warmup_steps=1,
                max_steps=max_steps,  # ← Use calculated max_steps
                learning_rate=5e-5,
                fp16=True,
                output_dir=self.trained_model,
                logging_steps=1,
                save_strategy="steps",
                save_steps=50,
                save_total_limit=3,  # Keep only 3 latest checkpoints
                load_best_model_at_end=False,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
        )

        print("Trainer Ready.")

        # ----------------------------------------------------------------------------------------
        # Training - resume from checkpoint if available
        # ----------------------------------------------------------------------------------------
        model.config.use_cache = False

        if checkpoints:
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            trainer.train()

        print("Fine tuning completed.")

    def test_finetuned_model(self, difficulty):
        f = FineTuner()
        model_name = f.get_last_checkpoint()
        tokenizer = self.Preprocessor.tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(base_model, self.current_checkpoint)

        # Set to evaluation mode
        model.eval()


        difficulty = difficulty
        width = 20
        height = 20
        num_enemies = 8
        num_wall = 18
        prompt = f"""
            Generate a top-down 2D game map as a JSON array of strings.

            Specifications:
            - Size: {width}x{height} tiles
            - Number of barriers: {num_wall} inner 'B' tiles + border of 'B's
            - Number of enemies: {num_enemies} 'E' tiles
            - Player: 1 'P' tile, placed so it can move
            - Empty spaces: '.' tiles

        Rules:
            1. Map borders must be 'B' (top, bottom, left, right)
            2. Inner 'B' tiles: exactly {num_wall}, do not block player
            3. Enemies: exactly {num_enemies}
            4. Player: exactly 1 'P', can move
            5. Return only JSON array of strings, no extra text

            Example 5x5:
            ["BBBBB", "B...B", "B.E.B", "B.P.B", "BBBBB"]
            """

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)  # ← Add padding=True

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,  # ← CRITICAL: Pass attention mask
                max_length=512,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        map_output = response.split("### Response:")[-1].strip()
        return map_output


    def load_map_from_finetuned_model(self, prompt, model, tokenizer):
        """model_name = self.model_path
        tokenizer = self.Preprocessor.tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(base_model, self.trained_model_mask)
        prompt = prompt

        # Set to evaluation mode
        model.eval()"""

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)  # ← Add padding=True

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,  # ← CRITICAL: Pass attention mask
                max_length=512,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        map_output = response.split("### Response:")[-1].strip()
        return map_output

    def load_llama(self):
        model_name = self.model_path
        tokenizer = self.Preprocessor.tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(base_model, self.current_checkpoint)

        # Set to evaluation mode
        model.eval()
        return model, tokenizer


#f = FineTuner()

"""
prompt = get_prompt(create_level(4))
model, tokenizer = f.load_llama()
for i in range (5):
    map = f.load_map_from_finetuned_model(prompt, model, tokenizer)
    print(map)

k = f.test_finetuned_model(4)
print(k)

prompt = get_prompt(create_level(4))
print(prompt)
map = f.load_map_from_finetuned_model(prompt)
print(map)"""



