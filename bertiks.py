from transformers import BertTokenizer, BertModel
import pandas as pd
import torch


df = pd.read_parquet('train-00000-of-00001.parquet')
all_texts = list(set(df['question_1'].to_list() + df['question_2'].to_list()))
# all_texts = ' '.join(all_texts)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer(all_texts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
# outputs = model(**inputs)

# print(outputs)
# predictions = torch.argmax(outputs.logits, dim=1)
# print(predictions)
# input_ids = inputs['input_ids']
# attention_mask = inputs['attention_mask']
#
with torch.no_grad():
    # last_hidden_states = model(input_ids, attention_mask=attention_mask)
    outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
#
# print(last_hidden_states)