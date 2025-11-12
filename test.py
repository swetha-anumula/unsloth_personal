# In your test script (e.g., test.py)

import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
from PIL import Image

# =====================================================================================
# PART 1: LOAD THE MODEL COMPONENTS IN NATIVE HALF-PRECISION (BF16 / FP16)
# =====================================================================================

print("✅ Step 1: Loading model components in native bfloat16/float16...")

model_name = "Qwen/Qwen-Image"
# Determine the best available data type for your GPU
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"   Using dtype: {dtype}")

try:
    # Load each component individually, specifying the dtype.
    # We are NOT passing any quantization_config.
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name, subfolder="tokenizer", trust_remote_code=True)
    
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        subfolder="text_encoder",
        torch_dtype=dtype, # Load in half-precision
        trust_remote_code=True,
    )
    
    diffusion_transformer = QwenImageTransformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=dtype, # Load in half-precision
        trust_remote_code=True,
    )
    print("✅ All components loaded successfully in half-precision!")

except Exception as e:
    print(f"❌ CRITICAL FAILURE: Could not load model components.")
    print(f"   Error: {e}")
    exit()

# =====================================================================================
# PART 2: COMBINE THE COMPONENTS AND APPLY UNSLOTH'S LORA
# =====================================================================================

# NEW, CORRECTED CODE BLOCK
# NEW, CORRECTED CODE BLOCK
class QwenImageTrainable(torch.nn.Module):
    def __init__(self, text_encoder, diffusion_transformer):
        super().__init__()
        # Keep a reference to the original text_encoder for convenience
        self._text_encoder_ref = text_encoder

        # Expose the components PEFT needs to see
        self.model = text_encoder.model
        self.transformer = diffusion_transformer
        self.lm_head = text_encoder.lm_head
        self.config = { "model_type": "qwen2" }

    # --- THIS IS THE FIX ---
    # Add the methods Unsloth is looking for.
    # They simply point to the real methods inside the text_encoder.
    def get_input_embeddings(self):
        return self._text_encoder_ref.get_input_embeddings()

    def get_output_embeddings(self):
        return self._text_encoder_ref.get_output_embeddings()
    # -----------------------

    def forward(self, *args, **kwargs):
        # This is a dummy forward pass; it's not used by SFTTrainer for LoRA
        return None

print("✅ Step 2: Combining models and applying Unsloth's LoRA...")

model = QwenImageTrainable(text_encoder, diffusion_transformer)
model._saved_temp_tokenizer = tokenizer
# Use Unsloth's get_peft_model. This will use your modified qwen2.py
# and find all the target_modules in BOTH the text_encoder and the transformer.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

model.print_trainable_parameters()

# =====================================================================================
# PART 3: RUN A TEST TRAINING STEP
# =====================================================================================
print("\n✅ Step 3: Setting up a test training step...")

# Create a minimal, valid dummy dataset
dummy_image = Image.new('RGB', (64, 64), 'red')
dummy_data = [
    {"messages": [{"role": "user", "content": [{"text": "A picture of a red square."}, {"image": dummy_image}]}]}
]
dummy_dataset = Dataset.from_list(dummy_data)

# Set up the Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dummy_dataset,
    dataset_text_field="messages",
    max_seq_length=1024,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_torch", # Use the standard AdamW optimizer
    ),
)

# Run the training
print("\n🚀 Starting training... This is the final test.")
try:
    trainer.train()
    print("\n\n🎉🎉🎉 CONGRATULATIONS! IT WORKS! 🎉🎉🎉")
    print("A full training step completed successfully without using bitsandbytes.")
    
except Exception as e:
    print(f"\n❌ FAILED: The training step crashed.")
    print(f"   Error: {e}")