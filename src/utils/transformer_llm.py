from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def generate_response(prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
  # Load pre-trained model and tokenizer

  tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
  model = GPT2LMHeadModel.from_pretrained("gpt2-large")
  model.eval()
  # Tokenize input prompt
  tokenizer.pad_token = tokenizer.eos_token
  model.config.pad_token_id = tokenizer.eos_token_id

  inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

  with torch.no_grad():
    output_ids = model.generate(
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      max_new_tokens=max_length,
      temperature=temperature,
      do_sample=True,
      top_k=50,
      top_p=0.95,
      num_return_sequences=1,
      pad_token_id=tokenizer.pad_token_id
    )

  return tokenizer.decode(output_ids[0], skip_special_tokens=True)