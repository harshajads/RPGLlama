def finetune_model(self):
    model_name = self.model_path
    tokenized_dataset = self.Preprocessor.tokenized_dataset_location

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='float16'
    )

    # ----------------------------------------------------------------------------------------
    # Load the starting model
    # ----------------------------------------------------------------------------------------
    if os.path.exists(self.trained_model) and os.listdir(self.trained_model):
        # Previous adapter exists -> incremental training
        print("▶️ Found previous adapter. Loading for incremental fine-tuning...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto'
        )
        model = PeftModel.from_pretrained(base_model, self.trained_model)
    else:
        # No previous adapter -> start fresh from TinyLlama base
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
    # Tokenizer setup
    # ----------------------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------------------------------------------------------------------
    # Freeze base model weights
    # ----------------------------------------------------------------------------------------
    """for param in base_model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)"""
    for name, param in model.named_parameters():
        # Freeze only base model parameters
        if "lora_" not in name:
            param.requires_grad = False
        else:
            # LoRA layers must be trainable
            param.requires_grad = True

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    # ----------------------------------------------------------------------------------------
    # Print trainable parameters
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
    # Load tokenized dataset
    # ----------------------------------------------------------------------------------------
    tokenized_data = load_from_disk(tokenized_dataset)

    # ----------------------------------------------------------------------------------------
    # Trainer setup
    # ----------------------------------------------------------------------------------------
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            warmup_steps=1,
            num_train_epochs=50,
            learning_rate=5e-5,
            fp16=True,
            output_dir=self.trained_model,
            logging_steps=1,
            save_strategy="steps",
            save_steps=50
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    )

    print("Trainer Ready.")

    # ----------------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------------
    model.config.use_cache = False
    trainer.train()
    print("Fine tuning completed.")

    # ----------------------------------------------------------------------------------------
    # Save updated adapter for incremental fine-tuning
    # ----------------------------------------------------------------------------------------
    print("Saving updated LoRA adapter...")
    model.save_pretrained(self.trained_model)
    tokenizer.save_pretrained(self.trained_model)
    print(f"✅ Adapter saved at {self.trained_model}")

    # ----------------------------------------------------------------------------------------
    # Optional: merge final adapter into base model (uncomment if needed)
    # ----------------------------------------------------------------------------------------
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(self.trained_model)
    # tokenizer.save_pretrained(self.trained_model)
    # print(f"✅ Merged model saved at {self.trained_model}")
