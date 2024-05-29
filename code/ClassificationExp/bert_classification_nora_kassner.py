import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, BertForMaskedLM, BertConfig,BertTokenizer,BertModel
import torch
import json
from tqdm import tqdm
import string

pre_trained_model = '/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/ClassificationExp/BigBert/bigbert_coqa.bin'
stop_words_list   = '/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/ClassificationExp/BigBert/jang-data/stopwords.txt'



def is_punctuation(s):
    return all(char in string.punctuation for char in s)
def load_plm_state_dict(file_name, plm_name):
    device = torch.device("cpu")
    aa = torch.load(file_name,map_location=device)
    new_dict = {}
    for key in aa.keys():
        if key.startswith(plm_name):
            if key.startswith(f'{plm_name}.pooler'):
                continue
            new_dict[key.replace(f"{plm_name}.", "")] = aa[key]
    return new_dict

def load_model(model_path):
        model     = BertForMaskedLM.from_pretrained("bert-large-uncased")
        model.bert.load_state_dict(load_plm_state_dict(model_path, 'bert'))
        return model

def masked_inference_vanilla():
        """
        Vanilla model for language modelling
        """
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertForMaskedLM.from_pretrained("bert-large-uncased")
        #text = "Miky way is a [MASK]."
        #text = "Goblet does not have fancy [MASK]."
        #text = "Goblet does not have fancy [MASK]."
        text = "A recorder is person. A recorder is not a [MASK]."

        encoded_input = tokenizer(text,return_tensors='pt')
        mask_token_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
        with torch.no_grad():
                logits = model(**encoded_input).logits


        mask_token_logits = logits[0,mask_token_index,:]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        for token in top_5_tokens:
                print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

def masked_inference_QA(pre_trained_model):
        """
        Vanilla model for language modelling
        """
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = load_model(pre_trained_model)

        text = "A recorder is person. A recorder is not a [MASK]."
        #text = "Goblet does not have fancy [MASK]."
        encoded_input = tokenizer(text,return_tensors='pt')
        mask_token_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
        with torch.no_grad():
                logits = model(**encoded_input).logits

        mask_token_logits = logits[0,mask_token_index,:]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        for token in top_5_tokens:
                print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

def batch_vanilla(json_file_path):
        stop_list = []
        with open(stop_words_list,'r') as file:
                lines = file.readlines()

        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        model = BertForMaskedLM.from_pretrained("bert-large-uncased")

        match = 0
        total = 0

        coherent = 0
        n_coherent = 0
        matches = {}
        with open(json_file_path) as f:
                num_lines = sum(1 for line in f)

                # Reset the file pointer to the beginning
                f.seek(0)

                for line in tqdm(f, total=num_lines, desc='Reading and processing JSONL'):
                        #total = total + 1
                        data = json.loads(line.strip())
                        # print(f"The data is : {data}")
                        # print(f"The type of the data is: {type(data)}")

                        masked_pos = data['masked_sentences']
                        masked_neg = data['masked_negations']

                        #masked_pos = data['opposite_sent']
                        #masked_neg = data['input_sent']

                        pos_tok = tokenizer(masked_pos, return_tensors="pt")
                        neg_tok = tokenizer(masked_neg, return_tensors="pt")

                        pos_mask_token_index = (pos_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                        neg_mask_token_index = (neg_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

                        with torch.no_grad():
                                logits_pos = model(**pos_tok).logits
                                logits_neg = model(**neg_tok).logits

                        pos_pred_token_id = logits_pos[0, pos_mask_token_index].argmax(axis=-1)
                        neg_pred_token_id = logits_neg[0, neg_mask_token_index].argmax(axis=-1)

                        pred_pos_token_orig = tokenizer.decode(pos_pred_token_id, skip_special_tokens=True)
                        pred_neg_token_orig = tokenizer.decode(neg_pred_token_id, skip_special_tokens=True)

                        """
                        if pred_pos_token_orig in stop_list or pred_neg_token_orig in stop_list:
                                continue

                        elif is_punctuation(pred_pos_token_orig) or is_punctuation(pred_neg_token_orig):
                                continue

                        else:
                                total = total + 1

                        """
                        total = total + 1
                        if (pred_pos_token_orig == pred_neg_token_orig):
                                match = match + 1
                                matches[match] = [
                                        [
                                                masked_pos,
                                                pred_pos_token_orig,
                                        ],
                                        [
                                                masked_neg,
                                                pred_neg_token_orig
                                        ]
                                ]

        """
        with open('/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/ClassificationExp/mismatches/vanilla_matches.json','w') as file:
                json.dump(matches,file,indent=4)
        """
        return match, total, coherent, n_coherent


def batch_finetuned(json_file_path):
        stop_list = []
        with open(stop_words_list,'r') as file:
                lines = file.readlines()

        stop_list = [line.strip() for line in lines]
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        model = load_model(pre_trained_model)

        match = 0
        total = 0

        coherent = 0
        n_coherent = 0

        matches = {}
        with open(json_file_path) as f:
                num_lines = sum(1 for line in f)

                # Reset the file pointer to the beginning
                f.seek(0)

                for line in tqdm(f, total=num_lines, desc='Reading and processing JSONL'):
                        #total = total + 1
                        data = json.loads(line.strip())
                        # print(f"The data is : {data}")
                        # print(f"The type of the data is: {type(data)}")

                        masked_pos = data['masked_sentences']
                        masked_neg = data['masked_negations']

                        #masked_pos = data['opposite_sent']
                        #masked_neg = data['input_sent']

                        pos_tok = tokenizer(masked_pos, return_tensors="pt")
                        neg_tok = tokenizer(masked_neg, return_tensors="pt")

                        pos_mask_token_index = (pos_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                        neg_mask_token_index = (neg_tok.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

                        with torch.no_grad():
                                logits_pos = model(**pos_tok).logits
                                logits_neg = model(**neg_tok).logits

                        pos_pred_token_id = logits_pos[0, pos_mask_token_index].argmax(axis=-1)
                        neg_pred_token_id = logits_neg[0, neg_mask_token_index].argmax(axis=-1)

                        pred_pos_token_orig = tokenizer.decode(pos_pred_token_id, skip_special_tokens=True)
                        pred_neg_token_orig = tokenizer.decode(neg_pred_token_id, skip_special_tokens=True)

                        #print(stop_list)

                        """
                        if pred_pos_token_orig in stop_list or pred_neg_token_orig in stop_list:
                                continue

                        elif is_punctuation(pred_pos_token_orig) or is_punctuation(pred_neg_token_orig):
                                continue

                        else:
                                total = total + 1
                        """
                        total = total + 1
                        if (pred_pos_token_orig == pred_neg_token_orig):
                                match = match + 1
                                matches[match] = [
                                        [
                                                masked_pos,
                                                pred_pos_token_orig,
                                        ],
                                        [
                                                masked_neg,
                                                pred_neg_token_orig
                                        ]
                                ]
        """
        
        with open('/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/ClassificationExp/mismatches/finetuned_matches.json','w') as file:
                json.dump(matches,file,indent=4)
        """
        return match, total, coherent, n_coherent


def calculate_percentsge(val1,val2):
        return (val1/val2)*100

if __name__ == "__main__":
        #load_model(pre_trained_model)
        #masked_inference_vanilla()
        #masked_inference_QA(pre_trained_model)

        """
        print("Jang Data Scores...")
        match, total, _, _ = batch_vanilla('/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/ClassificationExp/BigBert/jang-data/dataset.jsonl')
        print(f"The total samples are: {total}")
        print(f"The no. of matches between positive and negative by the vanilla encoder model: {match}")
        match, total, _, _ = batch_finetuned('/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/ClassificationExp/BigBert/jang-data/dataset.jsonl')
        print(f"The total samples are: {total}")
        print(f"The no. of matches between positive and negative by the QA encoder model: {match}")
        """


        print("Nora Kassner Data Scores...")
        match, total, _, _ = batch_vanilla(
                '/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/ClassificationExp/BigBert/nora/Squad.jsonl')
        print(f"The total samples are: {total}")
        print(f"The no. of matches between positive and negative by the vanilla encoder model: {match}")
        print("----------------------------------")
        match, total, _, _ = batch_finetuned(
                '/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/ClassificationExp/BigBert/nora/Squad.jsonl')
        print(f"The total samples are: {total}")
        print(f"The no. of matches between positive and negative by the finetuned encoder model: {match}")


        #print(calculate_percentsge(30,51))



