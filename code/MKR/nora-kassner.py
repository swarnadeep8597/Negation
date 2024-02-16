import warnings
warnings.filterwarnings("ignore")

import json
from transformers import AutoTokenizer, BertForMaskedLM,RobertaForMaskedLM
import torch
import torch.nn.functional as F

import re

nora_kassner_bert    = "/Users/sbhar/Riju/PhDCode/ACL/data/MKR/Squad.jsonl"
nora_kassner_roberta = "/Users/sbhar/Riju/PhDCode/ACL/data/MKR/Squad-roberta.jsonl"
stopwords_path       = "/Users/sbhar/Riju/PhDCode/ACL/data/MKR/stopwords.txt"

def clean_token(token):
    # Define a regular expression pattern to match special characters.
    special_chars_pattern = r'[^a-zA-Z0-9\s\n]'  # This pattern matches non-alphanumeric characters and spaces.

    # Use the re.sub() function to replace all special characters with an empty string.
    cleaned_token = re.sub(special_chars_pattern, '', token)

    return cleaned_token

#torch.Size([1, 2048])
def flip_ns_bert(mask_file_path,stopwords_path):
        with open(mask_file_path,'r') as f:
                data = json.load(f)

        #tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        #model = RobertaForMaskedLM.from_pretrained("roberta-base")

        #tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        #model = RobertaForMaskedLM.from_pretrained("roberta-large")

        stop_list = []
        with open(stopwords_path, "r") as stop:
                lines = stop.readlines()

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        match   = 0
        total   = 0
        r_match = 0

        coherent = 0
        n_coherent = 0

        for item in data:
                total = total + 1
                masked_pos = item['masked_sentences']
                true_lable = item['obj_label']
                masked_neg = item['masked_negations']

                pos_tok  = tokenizer(masked_pos, return_tensors="pt")
                neg_tok  = tokenizer(masked_neg, return_tensors="pt")

                pos_mask_token_index  = (pos_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                neg_mask_token_index  = (neg_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]


                with torch.no_grad():
                        logits_pos = model(**pos_tok).logits
                        logits_neg = model(**neg_tok).logits

                pos_pred_token_id  = logits_pos[0, pos_mask_token_index].argmax(axis=-1)
                neg_pred_token_id  = logits_neg[0, neg_mask_token_index].argmax(axis=-1)

                #print(pos_pred_token_id)

                logits2softmax_pos = F.softmax(logits_pos[0,pos_mask_token_index])
                logits2softmax_neg = F.softmax(logits_neg[0,neg_mask_token_index])
                #print(logits2softmax_neg[0])

                predicted_pos_token_ids, pos_indices = torch.topk(logits2softmax_pos, k=5)
                predicted_neg_token_ids, neg_indices = torch.topk(logits2softmax_neg, k=5)

                #print(pos_indices[1])
                #print(neg_indices[1])


                rev_index_prob = []
                for n_index in pos_indices[0]:
                        prob = logits2softmax_pos[0, n_index]
                        rev_index_prob.append((n_index, 1 - prob)) #flipping the probs here

                sorted_list_descending = sorted(rev_index_prob, key=lambda x: x[1], reverse=True)
                pred_neg_token_rev = tokenizer.decode(sorted_list_descending[0][0], skip_special_tokens=True)

                pred_pos_token_orig = tokenizer.decode(pos_pred_token_id, skip_special_tokens=True)
                pred_neg_token_orig = tokenizer.decode(neg_pred_token_id, skip_special_tokens=True)

                if (pred_pos_token_orig == pred_neg_token_orig):
                        match = match + 1
                        if pred_neg_token_rev in stop_list:
                                n_coherent = n_coherent + 1

                        else:
                                coherent = coherent + 1

                if (pred_neg_token_rev == pred_pos_token_orig):
                        r_match = r_match + 1


        return match,total,r_match,coherent,n_coherent

def flip_ns_roberta(mask_file_path,stopwords_path):
        with open(mask_file_path,'r') as f:
                data = json.load(f)

        #tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        #model = RobertaForMaskedLM.from_pretrained("roberta-base")

        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        model = RobertaForMaskedLM.from_pretrained("roberta-large")

        stop_list = []
        with open(stopwords_path, "r") as stop:
                lines = stop.readlines()

        #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        #model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        match   = 0
        total   = 0
        r_match = 0

        coherent = 0
        n_coherent = 0

        for item in data:
                total = total + 1
                masked_pos = item['masked_sentences']
                true_lable = item['obj_label']
                masked_neg = item['masked_negations']

                pos_tok  = tokenizer(masked_pos, return_tensors="pt")
                neg_tok  = tokenizer(masked_neg, return_tensors="pt")

                pos_mask_token_index  = (pos_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                neg_mask_token_index  = (neg_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]


                with torch.no_grad():
                        logits_pos = model(**pos_tok).logits
                        logits_neg = model(**neg_tok).logits

                pos_pred_token_id  = logits_pos[0, pos_mask_token_index].argmax(axis=-1)
                neg_pred_token_id  = logits_neg[0, neg_mask_token_index].argmax(axis=-1)

                #print(pos_pred_token_id)

                logits2softmax_pos = F.softmax(logits_pos[0,pos_mask_token_index])
                logits2softmax_neg = F.softmax(logits_neg[0,neg_mask_token_index])
                #print(logits2softmax_neg[0])

                predicted_pos_token_ids, pos_indices = torch.topk(logits2softmax_pos, k=5)
                predicted_neg_token_ids, neg_indices = torch.topk(logits2softmax_neg, k=5)

                #print(pos_indices[1])
                #print(neg_indices[1])


                rev_index_prob = []
                for n_index in pos_indices[0]:
                        prob = logits2softmax_pos[0, n_index]
                        rev_index_prob.append((n_index, 1 - prob)) #flipping the probs here

                sorted_list_descending = sorted(rev_index_prob, key=lambda x: x[1], reverse=True)
                pred_neg_token_rev = tokenizer.decode(sorted_list_descending[0][0], skip_special_tokens=True)

                pred_pos_token_orig = tokenizer.decode(pos_pred_token_id, skip_special_tokens=True)
                pred_neg_token_orig = tokenizer.decode(neg_pred_token_id, skip_special_tokens=True)

                if (pred_pos_token_orig == pred_neg_token_orig):
                        match = match + 1
                        if pred_neg_token_rev in stop_list:
                                n_coherent = n_coherent + 1

                        else:
                                coherent = coherent + 1

                if (pred_neg_token_rev == pred_pos_token_orig):
                        r_match = r_match + 1


        return match,total,r_match,coherent,n_coherent

if __name__ == "__main__":
        print("-------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------")
        print("The Flip Stats for BERT on Nora Kassner's data!!")
        match,total,r_match,coherent,n_coherent = flip_ns_bert(
                mask_file_path=nora_kassner_bert,
                stopwords_path=stopwords_path
        )

        #print((match,total,r_match,coherent,n_coherent))
        print(f"The total string looked at:{total}")
        print(f"The reverse matches is:{r_match}")
        #print(f"The fraction of inputs where it confused negation originally is: {(match/total)*100}")
        #print(f"The fraction of inputs where it confused negation after flip is: {(r_match/total)*100}")
        print(f"The coherent string: {coherent}")
        print(f"The non coherent string: {n_coherent}")
        print("-------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------")
        print("The Flip Stats for ROBERTA on Nora Kassner's data!!")
        ro_match, ro_total, ro_r_match, ro_coherent, ro_n_coherent = flip_ns_roberta(
                mask_file_path=nora_kassner_roberta,
                stopwords_path=stopwords_path
        )

        # print((match,total,r_match,coherent,n_coherent))
        print(f"The total string looked at:{ro_total}")
        print(f"The reverse matches is:{ro_r_match}")
        # print(f"The fraction of inputs where it confused negation originally is: {(match/total)*100}")
        # print(f"The fraction of inputs where it confused negation after flip is: {(r_match/total)*100}")
        print(f"The coherent string: {ro_coherent}")
        print(f"The non coherent string: {ro_n_coherent}")

