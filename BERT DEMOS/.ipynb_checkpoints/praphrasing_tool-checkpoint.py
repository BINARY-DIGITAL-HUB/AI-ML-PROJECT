#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install transformers


# In[5]:


# pip install --upgrade pip


# In[6]:


# pip install sentencepiece


# In[1]:


from json import load
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import streamlit as st 


st.header('Paraprashing Tool')



@st.cache
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = TFAutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    return tokenizer, model

tokenizer , model = load_model()
# In[12]:


# dir(tokenizer)


# In[3]: sample sentnce


sentence_1 = "Washing your hands Properly will keep you away from COVID-19."
sentence_2 = "Wikipedia was launched on January 15, 2001, and was created by Jimmy Wales and Larry Sanger."
sentence_3 = "NLP is one of the interesting fields for Data Scientists to focus on."

# sentences  = f'{sentence_1} {sentence_2} {sentence_3}'

def paraprahser(doc='' , para_num = 4):

    text =  "paraphrase: " + doc + " </s>"

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="tf")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]


    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=False,
        num_return_sequences=para_num
    )

   
    
    paraprahsered_document = []
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        print(line)
        paraprahsered_document.append(line)

    return doc , paraprahsered_document

sentence = st.input_area('Type text')
    
doc , modify_doc =  paraprahser(sentence)

print(modify_doc)
st.write(modify_doc)


# In[ ]:




