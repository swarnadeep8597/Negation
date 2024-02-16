import json
from sklearn.metrics import accuracy_score
import re


HC_path              = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_hc_split_all.json"
CH_path              = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_ch_split_all.json"
C_negH_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_c_negh_split_all.json"
scope_path           = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_bracket_split_all.json"
negC_H_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_negc_h_split_all.json"
negC_H_pred_path  = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/intermediate/golden_negc_h_pred.json"
scoped_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_bracket_split_all.json"
negC_negH_path       = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_negc_negh_split_all.json"
HC_prime_path        = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_hc_prime_split_all.json"

llama_HC_path        = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/llama_predictions/HC_pred.json"
llama_CH_path        = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/llama_predictions/CH_pred.json"
llama_scope_path     = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/llama_predictions/PH_pred.json"
llama_HC_prime_path  = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/llama_predictions/hc_prime_pred.json"
llama_negC_H_pred = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/intermediate/llama_negc_h_pred.json"


def read_prompt1_pred_file(op_file_path):
        with open(op_file_path,'r') as file:
                results = json.load(file)

        outputs = {}
        for i,key in enumerate(results):
                op = results[key]
                relevant_parts = op.split()[-3:]
                relevant_parts = [item.lower() for item in relevant_parts]
                for rel in relevant_parts:
                        if 'entailment' in rel:
                                outputs[key] = 'entailment'
                                continue
                        elif 'contradiction' in rel:
                                outputs[key] = 'contradiction'
                                continue
                        elif 'neutral' in rel:
                                outputs[key] = 'neutral'
                                continue


        return outputs

def read_prompt2_pred_file(pred_file):
        with open(pred_file,'r') as file:
                results = json.load(file)

        outputs = {}
        for i, key in enumerate(results):
                op = results[key]

                relevant_parts = op.split()[-4:]
                relevant_parts = [item.lower() for item in relevant_parts]
                for rel in relevant_parts:
                        #print(rel)
                        if 'entailment' in rel:
                                outputs[key] = 'entailment'
                                continue
                        elif 'contradiction' in rel:
                                outputs[key] = 'contradiction'
                                continue
                        elif 'neutral' in rel:
                                outputs[key] = 'neutral'
                                continue

        return outputs

def extract_content(input_string):
        matches = re.findall(r'\[/INST\](.*)', input_string,re.DOTALL)
        if matches:
                #return match.group(1)
                return matches[-1].strip()
        else:
                return None

def read_prompt3_pred_file(llama_op_path):
        with open(llama_op_path,'r') as file:
                llama_op = json.load(file)

        pred_labels = {}
        for key in llama_op:
                labels = []
                content = llama_op[key]
                extracted = extract_content(content)
                prediction = extract_content(extracted)

                tokens = prediction.strip()
                if '[entailment]' in tokens:
                        labels.append('entailment')
                if '[contradiction]' in tokens:
                        labels.append('contradiction')
                if '[neutral]' in tokens:
                        labels.append('neutral')


                pred_labels[key] = labels
                if len(labels) == 0:
                        if 'entailment' in tokens:
                                labels.append('enatilment')
                        if 'contradiction' in tokens:
                                labels.append('contradiction')
                        if 'neutral' in tokens:
                                labels.append('neutral')

                        pred_labels[key] = labels


        #print(len(pred_labels))
        """
        28 ['enatilment', 'contradiction', 'neutral']
        86 ['contradiction', 'neutral']
        101 ['contradiction', 'neutral']
        """

        for key in pred_labels:
                if len(pred_labels[key]) > 1:
                        pred_labels[key] = pred_labels[key][-1]

        return pred_labels

def calc_C_negH(pos_file_path,true_c_negH,op_from_llama=True):
        c_negH_pred = {}

        if op_from_llama:
                ch = read_prompt1_pred_file(pos_file_path)
        else:
                with open(pos_file_path,'r') as file:
                        ch = json.load(file)


        with open(true_c_negH,'r') as file:
                c_negH = json.load(file)

        ch_labels = {}
        for key in ch:
                if op_from_llama:
                        ch_labels[key] = ch[key]
                else:
                        label = ch[key][2].lower()
                        ch_labels[key] = label

        for key in ch_labels:
                label = ch_labels[key]
                if label == 'entailment':
                        c_negH_pred[key] = 'contradiction'
                if label == 'contradiction':
                        c_negH_pred[key] = 'entailment'
                if label == 'neutral':
                        c_negH_pred[key] = 'neutral'

        true_labels = []
        mismatch_ids = []


        for key in c_negH_pred:
                true_label = c_negH[key][2].lower()
                pred_label = c_negH_pred[key]
                true_labels.append(true_label)
                if true_label != pred_label:
                        mismatch_ids.append(key)

        #print(f"The no. of mismatches are: ",len(mismatch_ids))
        #print(f"The mismatch ids are: ",mismatch_ids)
        #print(set(true_labels))
        #print(len(true_labels))
        #print(len(list(c_negH_pred.values())))

        print(f"The accuracy score for (C neg H) calculated on {len(list(c_negH_pred.values()))} samples is: {accuracy_score(true_labels,list(c_negH_pred.values()))}")

def calc_negC_H(pos_file_path,rev_file_path,brk_file_path,star_file_path,true_negC_h,op_from_llama=True):
        if op_from_llama:
                ch = read_prompt1_pred_file(pos_file_path)
                ph = read_prompt2_pred_file(brk_file_path)
                hc_prime = read_prompt3_pred_file(star_file_path)
                hc = read_prompt3_pred_file(rev_file_path)
        else:
                with open(pos_file_path,'r') as file:
                        ch = json.load(file)

                with open(brk_file_path,'r') as file:
                        ph = json.load(file)

                with open(star_file_path,'r') as file:
                        hc_prime = json.load(file)

                with open(rev_file_path,'r') as file:
                        hc = json.load(file)


        with open(true_negC_h,'r') as file:
                negC_h = json.load(file)

        ch_labels         = {}
        hc_labels         = {}
        ph_labels         = {}
        hc_prime_labels   = {}
        negC_h_labels     = {}

        for key in ch:
                if op_from_llama:
                        ch_labels[key] = ch[key]
                else:
                        label = ch[key][2].lower()
                        ch_labels[key] = label


        for key in ph:
                if op_from_llama:
                        ph_labels[key] = ph[key]
                else:
                        label = ph[key][2].lower()
                        ph_labels[key] = label

        for key in hc_prime:
                if op_from_llama:
                        hc_prime_labels[key] = hc_prime[key][0]
                else:
                        label = hc_prime[key][2].lower()
                        hc_prime_labels[key] = label

        for key in hc:
                if op_from_llama:
                        hc_labels[key] = hc[key][0]
                else:
                        label = hc[key][2].lower()
                        hc_labels[key] = label

        for key in negC_h:
                label = negC_h[key][2].lower()
                negC_h_labels[key] = label

        negC_h_pred = {}
        for key in ch_labels:
                label = ch_labels[key].lower()
                if label == 'entailment':
                        if key in list(ph_labels.keys()):
                                if ph_labels[key].lower() == 'entailment':
                                        negC_h_pred[key] = 'entailment'

                                elif ph_labels[key].lower() == 'neutral':
                                        if key in list(hc_prime_labels.keys()):
                                                if hc_prime_labels[key] == 'entailment':
                                                        negC_h_pred[key] = 'contradiction'
                                                else:
                                                        negC_h_pred[key] = 'neutral'
                                        else:
                                                negC_h_pred[key] = 'neutral'
                                else:
                                        negC_h_pred[key] = 'neutral'

                if label == 'contradiction':
                        if key in list(ph_labels.keys()):
                                if ph_labels[key] == 'contradiction':
                                        negC_h_pred[key] = 'contradiction'
                                elif ph_labels[key] == 'neutral':
                                        if key in list(hc_prime_labels.keys()):
                                                if hc_prime_labels[key] == 'entailment':
                                                        negC_h_pred[key] = 'contradiction'
                                                else:
                                                        negC_h_pred[key] = 'neutral'
                                        else:
                                                negC_h_pred[key] = 'neutral'

                                else:
                                        negC_h_pred[key] = 'neutral'


                if label == 'neutral':
                        if hc_labels[key] == 'entailment':
                                negC_h_pred[key] = 'contradiction'
                        else:
                                negC_h_pred[key] = 'neutral'

        #print(len(negC_h_pred))
        true_labels = []
        mismatch_ids = []
        for key in negC_h_pred:
                true_label = negC_h_labels[key].lower()
                pred_label = negC_h_pred[key]
                true_labels.append(true_label)
                if true_label != pred_label:
                        mismatch_ids.append(key)

        #print(len(true_labels))
        #print(mismatch_ids)

        if op_from_llama:
                with open(llama_negC_H_pred,'w') as file:
                        json.dump(negC_h_pred,file)
        else:
                with open(negC_H_pred_path,'w') as file:
                        json.dump(negC_h_pred,file)

        #print("The pred file has been dumped!!")
        print(f"The accuracy for (negC H) calculated on {len(list(negC_h_pred.values()))} samples is: {accuracy_score(true_labels,list(negC_h_pred.values()))}")

def calc_negC_negH(negC_h_pred,negC_negH_true,op_from_llam=True):
        with open(negC_h_pred,'r') as file:
                negC_h = json.load(file)
        with open(negC_negH_true,'r') as file:
                negC_negH_true = json.load(file)

        negC_negH_true_labels = {}
        for key in negC_negH_true:
                label = negC_negH_true[key][2].lower()
                negC_negH_true_labels[key] = label


        #print(len(negC_negH_true_labels))


        negC_negH_pred = {}
        for key in negC_h:
                label = negC_h[key]
                if label == 'contradiction':
                        negC_negH_pred[key] = 'entailment'
                if label == 'entailment':
                        negC_negH_pred[key] = 'contradiction'
                if label == 'neutral':
                        negC_negH_pred[key] = 'neutral'

        #print(len(negC_negH_pred))
        mismatch_ids = []
        pred_labels = []
        true_labels = []
        for key in negC_negH_pred:
                pred_label = negC_negH_pred[key]
                true_label = negC_negH_true_labels[key]
                pred_labels.append(pred_label)
                true_labels.append(true_label)
                if pred_label != true_label:
                        mismatch_ids.append(key)

        #print(mismatch_ids)
        #print(accuracy_score(list(negC_negH_true_labels.values()),list(negC_negH_pred.values())))
        #print(accuracy_score(true_labels,pred_labels))
        print(f"The accuracy for (negC negH) calculated on {len(list(negC_negH_pred.values()))} samples is: {accuracy_score(true_labels, pred_labels)}")

if __name__ == "__main__":
        print("-----------------------------------------")
        print("The Scores for SNLI")
        print("-----------------------------------------")
        print("The Algorithm Scores using the Gold Annotations")
        print("-----------------------------------------")
        calc_C_negH(
                pos_file_path=CH_path,
                true_c_negH=C_negH_path,
                op_from_llama=False
        )
        print("-----------------------------------------")
        calc_negC_H(
                pos_file_path=CH_path,
                rev_file_path=HC_path,
                brk_file_path=scope_path,
                star_file_path=HC_prime_path,
                true_negC_h=negC_H_path,
                op_from_llama=False
        )
        print("-----------------------------------------")

        calc_negC_negH(
                negC_h_pred=negC_H_pred_path,
                negC_negH_true=negC_negH_path
        )

        print("-----------------------------------------")
        print("-----------------------------------------")
        print("The Algorithm Scores using the LLAMA2 predictions")
        print("-----------------------------------------")
        calc_C_negH(
                pos_file_path=llama_CH_path,
                true_c_negH=C_negH_path,
                op_from_llama=True
        )
        print("-----------------------------------------")
        calc_negC_H(
                pos_file_path=llama_CH_path,
                rev_file_path=llama_HC_path,
                brk_file_path=llama_scope_path,
                star_file_path=llama_HC_prime_path,
                true_negC_h=negC_H_path,
                op_from_llama=True
        )
        print("-----------------------------------------")
        calc_negC_negH(
                negC_h_pred=llama_negC_H_pred,
                negC_negH_true=negC_negH_path,
                op_from_llam=True
        )
        print("-----------------------------------------")