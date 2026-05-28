import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

fine_tuned_model = PeftModel.from_pretrained(base_model, "say89/Mistral-7B-Instruct-v0.2-Finetuned") # load the traind adapters and load base model and use them

# 3. Define a history test prompt using the training format
prompt = "<s>[INST] Question: What event marked the beginning of the French Revolution? [/INST] Answer:"

# 4. Run Inference
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = fine_tuned_model.generate(
        **inputs, 
        max_new_tokens=30, 
        do_sample=True, 
        temperature=0.3
    )

# 5. Clean and print the final answer
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = generated_text.split("Answer:")[-1].strip()
print(f"Generated Answer: {answer}")
