import warnings
warnings.filterwarnings("ignore")

import json
from transformers import AutoTokenizer, BertForMaskedLM,RobertaForMaskedLM
import torch
import torch.nn.functional as F

import re
from tqdm import tqdm


jang_bert       = "/Users/sbhar/Riju/PhDCode/ACL/data/MKR/dataset.jsonl"
jang_roberta    = "/Users/sbhar/Riju/PhDCode/ACL/data/MKR/dataset-roberta.jsonl"
stopwords       = "/Users/sbhar/Riju/PhDCode/ACL/data/MKR/stopwords.txt"

def clean_token(token):
    # Define a regular expression pattern to match special characters.
    special_chars_pattern = r'[^a-zA-Z0-9\s\n]'  # This pattern matches non-alphanumeric characters and spaces.

    # Use the re.sub() function to replace all special characters with an empty string.
    cleaned_token = re.sub(special_chars_pattern, '', token)

    return cleaned_token

def flip_ja_bert(mask_file_path,stopword_path):
    #f = open(json_file_path)
    #data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    #tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    #model = BertForMaskedLM.from_pretrained("bert-large-uncased")

    #tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    #model = RobertaForMaskedLM.from_pretrained("roberta-large")

    stop_list = []
    with open(stopword_path,"r") as stop:
        lines = stop.readlines()

    stop_list = [line.strip() for line in lines]
    #print(stop_list[0])
    #return 0,0,0,0,0
    #tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    #model = RobertaForMaskedLM.from_pretrained("roberta-base")

    match = 0
    total = 0
    r_match = 0
    coherent = 0
    n_coherent = 0
    with open (mask_file_path) as f:
        num_lines = sum(1 for line in f)

        # Reset the file pointer to the beginning
        f.seek(0)

        for line in tqdm(f,total=num_lines, desc='Reading and processing JSONL'):
            total = total + 1
            data = json.loads(line.strip())
            #print(f"The data is : {data}")
            #print(f"The type of the data is: {type(data)}")

            masked_pos = data['opposite_sent']
            true_lable = data['wrong_prediction']
            masked_neg = data['input_sent']

            #print(f"The masked pos is: {masked_pos}")
            #print(f"The masked neg is: {masked_neg}")
            #print(f"The masked label is: {true_lable}")

            pos_tok = tokenizer(masked_pos, return_tensors="pt")
            neg_tok = tokenizer(masked_neg, return_tensors="pt")

            pos_mask_token_index = (pos_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            neg_mask_token_index = (neg_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

            with torch.no_grad():
                logits_pos = model(**pos_tok).logits
                logits_neg = model(**neg_tok).logits

            pos_pred_token_id = logits_pos[0, pos_mask_token_index].argmax(axis=-1)
            neg_pred_token_id = logits_neg[0, neg_mask_token_index].argmax(axis=-1)

            logits2softmax_pos = F.softmax(logits_pos[0, pos_mask_token_index])
            logits2softmax_neg = F.softmax(logits_neg[0, neg_mask_token_index])

            predicted_pos_token_ids, pos_indices = torch.topk(logits2softmax_pos, k=5)
            predicted_neg_token_ids, neg_indices = torch.topk(logits2softmax_neg, k=5)

            rev_index_prob = []
            # for n_index in neg_indices[0]:
            for n_index in pos_indices[0]:
                prob = logits2softmax_pos[0, n_index]
                rev_index_prob.append((n_index, 1 - prob))

            sorted_list_descending = sorted(rev_index_prob, key=lambda x: x[1], reverse=True)
            # print(sorted_list_descending)
            # pred_neg_token_rev = tokenizer.decode(sorted_list_descending[0][0], skip_special_tokens=True)
            pred_neg_token_rev = tokenizer.decode(sorted_list_descending[0][0], skip_special_tokens=True)

            """
            
            if pred_neg_token_rev in stop_list:
                n_coherent = n_coherent + 1

            else:
                coherent = coherent + 1
            """

            pred_pos_token_orig = tokenizer.decode(pos_pred_token_id, skip_special_tokens=True)
            pred_neg_token_orig = tokenizer.decode(neg_pred_token_id, skip_special_tokens=True)

            if (pred_pos_token_orig == pred_neg_token_orig):
                #print("--------------------------------")
                #print(f"The positive string:{masked_pos}")
                #print(f"Positive token:{pred_pos_token_orig}")
                #print(f"The negative string:{masked_neg}")
                #print(f"Negative token:{pred_neg_token_orig}")
                #print(f"Negative token reversed:{pred_neg_token_rev}")
                #print("--------------------------------")
                # print(f"Negative token reversed:{pred_neg_token_rev}")
                match = match + 1

                if pred_neg_token_rev in stop_list:
                    n_coherent = n_coherent + 1

                else:
                    coherent = coherent + 1

            if (pred_neg_token_rev == pred_pos_token_orig):
                print(f"The rev token is: ",pred_neg_token_rev)
                r_match = r_match + 1


    return match,total,r_match,coherent,n_coherent


def flip_ja_roberta(mask_file_path, stopword_path):
    # f = open(json_file_path)
    # data = json.load(f)

    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    # tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    # model = BertForMaskedLM.from_pretrained("bert-large-uncased")

    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = RobertaForMaskedLM.from_pretrained("roberta-large")

    stop_list = []
    with open(stopword_path, "r") as stop:
        lines = stop.readlines()

    stop_list = [line.strip() for line in lines]
    # print(stop_list[0])
    # return 0,0,0,0,0
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # model = RobertaForMaskedLM.from_pretrained("roberta-base")

    match = 0
    total = 0
    r_match = 0
    coherent = 0
    n_coherent = 0
    with open(mask_file_path) as f:
        num_lines = sum(1 for line in f)

        # Reset the file pointer to the beginning
        f.seek(0)

        for line in tqdm(f, total=num_lines, desc='Reading and processing JSONL'):
            total = total + 1
            data = json.loads(line.strip())
            # print(f"The data is : {data}")
            # print(f"The type of the data is: {type(data)}")

            masked_pos = data['opposite_sent']
            true_lable = data['wrong_prediction']
            masked_neg = data['input_sent']

            # print(f"The masked pos is: {masked_pos}")
            # print(f"The masked neg is: {masked_neg}")
            # print(f"The masked label is: {true_lable}")

            pos_tok = tokenizer(masked_pos, return_tensors="pt")
            neg_tok = tokenizer(masked_neg, return_tensors="pt")

            pos_mask_token_index = (pos_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            neg_mask_token_index = (neg_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

            with torch.no_grad():
                logits_pos = model(**pos_tok).logits
                logits_neg = model(**neg_tok).logits

            pos_pred_token_id = logits_pos[0, pos_mask_token_index].argmax(axis=-1)
            neg_pred_token_id = logits_neg[0, neg_mask_token_index].argmax(axis=-1)

            logits2softmax_pos = F.softmax(logits_pos[0, pos_mask_token_index])
            logits2softmax_neg = F.softmax(logits_neg[0, neg_mask_token_index])

            predicted_pos_token_ids, pos_indices = torch.topk(logits2softmax_pos, k=5)
            predicted_neg_token_ids, neg_indices = torch.topk(logits2softmax_neg, k=5)

            rev_index_prob = []
            # for n_index in neg_indices[0]:
            for n_index in pos_indices[0]:
                prob = logits2softmax_pos[0, n_index]
                rev_index_prob.append((n_index, 1 - prob))

            sorted_list_descending = sorted(rev_index_prob, key=lambda x: x[1], reverse=True)
            # print(sorted_list_descending)
            # pred_neg_token_rev = tokenizer.decode(sorted_list_descending[0][0], skip_special_tokens=True)
            pred_neg_token_rev = tokenizer.decode(sorted_list_descending[0][0], skip_special_tokens=True)

            """

            if pred_neg_token_rev in stop_list:
                n_coherent = n_coherent + 1

            else:
                coherent = coherent + 1
            """

            pred_pos_token_orig = tokenizer.decode(pos_pred_token_id, skip_special_tokens=True)
            pred_neg_token_orig = tokenizer.decode(neg_pred_token_id, skip_special_tokens=True)

            if (pred_pos_token_orig == pred_neg_token_orig):
                # print("--------------------------------")
                # print(f"The positive string:{masked_pos}")
                # print(f"Positive token:{pred_pos_token_orig}")
                # print(f"The negative string:{masked_neg}")
                # print(f"Negative token:{pred_neg_token_orig}")
                # print(f"Negative token reversed:{pred_neg_token_rev}")
                # print("--------------------------------")
                # print(f"Negative token reversed:{pred_neg_token_rev}")
                match = match + 1

                if pred_neg_token_rev in stop_list:
                    n_coherent = n_coherent + 1

                else:
                    coherent = coherent + 1

            if (pred_neg_token_rev == pred_pos_token_orig):
                print(f"The rev token is: ", pred_neg_token_rev)
                r_match = r_match + 1

    return match, total, r_match, coherent, n_coherent


if __name__ == "__main__":
         print("-------------------------------------------------------------------------------------------")
         print("-------------------------------------------------------------------------------------------")
         print("The Flip Stats for BERT on Jang's data!!")
         match,total,r_match,coherent,n_coherent = flip_ja_bert(
             mask_file_path=jang_bert,
             stopword_path=stopwords
         )
         # print((match,total,r_match,coherent,n_coherent))
         print(f"The total string looked at:{total}")
         print(f"The reverse matches is:{r_match}")
         # print(f"The fraction of inputs where it confused negation originally is: {(match/total)*100}")
         # print(f"The fraction of inputs where it confused negation after flip is: {(r_match/total)*100}")
         print(f"The coherent string: {coherent}")
         print(f"The non coherent string: {n_coherent}")
         print("-------------------------------------------------------------------------------------------")
         print("-------------------------------------------------------------------------------------------")
         print("The Flip Stats for ROBERTA on Jang's data!!")
         ro_match, ro_total, ro_r_match, ro_coherent, ro_n_coherent = flip_ja_roberta(
             mask_file_path=jang_roberta,
             stopword_path=stopwords
         )

         # print((match,total,r_match,coherent,n_coherent))
         print(f"The total string looked at:{ro_total}")
         print(f"The reverse matches is:{ro_r_match}")
         # print(f"The fraction of inputs where it confused negation originally is: {(match/total)*100}")
         # print(f"The fraction of inputs where it confused negation after flip is: {(r_match/total)*100}")
         print(f"The coherent string: {ro_coherent}")
         print(f"The non coherent string: {ro_n_coherent}")
         print("-------------------------------------------------------------------------------------------")

