import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Define your input text
input_text = "Translate this English text to German: Hello, how are you?"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Translate the input text to French
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, num_beams=4, early_stopping=True, no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95)

# Decode and print the translated text
translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Translated text:", translated_text)

