# Finetuning_LLM

Hippocratically: Medical Chatbot Fine-tuning Project

This repository contains code for fine-tuning the Llama 3.2-3B-Instruct model on medical conversation data to create a specialized medical chatbot assistant.

## Overview

This project demonstrates how to fine-tune Meta's Llama 3.2-3B-Instruct model on a dataset of doctor-patient conversations to create a medical assistant that can provide helpful, accurate medical information in a conversational format.

## Dataset

The project uses the [ruslanmv/ai-medical-chatbot](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot) dataset from Hugging Face, which contains structured conversations between patients and doctors. For efficient training, we randomly sample 1,000 conversations from the dataset.

## Fine-tuning Approach

We use Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) to fine-tune the model. This approach:

- Keeps most model parameters frozen
- Only updates a small set of adapter parameters
- Significantly reduces computational requirements
- Results in a much smaller model size (the adapter is just ~9MB)

### Key Technical Details

- **Base Model**: meta-llama/Llama-3.2-3B-Instruct
- **Quantization**: 4-bit quantization using bitsandbytes
- **PEFT Technique**: LoRA with rank=8, alpha=16
- **Target Modules**: q_proj, v_proj (attention components)
- **Training Framework**: Hugging Face's SFTTrainer from the TRL library

## Files and Their Functions

- `requirements.txt` (implied): Lists all necessary Python packages
- `fine_tuning_script.ipynb`: Main Jupyter notebook with the complete implementation

## Setup and Training

1. Install required packages:
   ```bash
   pip install -U transformers datasets peft trl bitsandbytes accelerate
   ```

2. Load and prepare the dataset:
   - Load the medical conversation dataset
   - Format the conversations using the model's chat template
   - Split into train and evaluation sets

3. Initialize the model with PEFT/LoRA configuration:
   - Load the base model with 4-bit quantization
   - Apply LoRA configuration to attention layers
   - Prepare the model for training

4. Train the model:
   - Use SFTTrainer for supervised fine-tuning
   - Train for 1 epoch with learning rate 2e-4
   - Use mixed precision (fp16) training for efficiency

5. Save and upload the resulting model to Hugging Face Hub

## Using the Fine-tuned Model

You can use the fine-tuned model directly from Hugging Face:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "thebnbrkr/hippocratically_llama3.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example conversation
messages = [
    {"role": "user", "content": "I've been experiencing severe headaches and sensitivity to light. What could this be?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

## Limitations

- The model is fine-tuned on a limited dataset (1,000 conversations)
- It should not be used for actual medical advice or diagnosis
- The model may sometimes generate incorrect or outdated medical information
- Always consult healthcare professionals for medical concerns

## Future Improvements

- Fine-tune on a larger and more diverse medical conversation dataset
- Implement evaluation metrics specific to medical accuracy
- Add retrieval augmentation to incorporate vetted medical knowledge
- Create a more robust evaluation framework with medical expert oversight

## License

This project is dependent on the license of the base Llama 3.2 model. Please refer to Meta's license terms for the Llama models before using this fine-tuned version.

