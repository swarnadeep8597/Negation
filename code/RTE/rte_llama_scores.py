import json
import re
from sklearn.metrics import accuracy_score

C_negH_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_C_negH_split.json"
negC_H_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_negC_H_split.json"
negC_negH_path       = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_negC_negH_split.json"

llama_C_negH         = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/llama_predictions/rte_llama_C_negH_pred.json"
llama_negC_H         = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/llama_predictions/rte_llama_negC_H_pred.json"
llama_negC_negH      = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/llama_predictions/rte_llama_negC_negH_pred.json"

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
                if '[notentailment]' in tokens:
                        labels.append('notentailment')


                pred_labels[key] = labels
                if len(labels) == 0:
                        if 'entailment' in tokens:
                                labels.append('enatilment')
                        if 'notentailment' in tokens:
                                labels.append('notentailment')

                        pred_labels[key] = labels


        for key in pred_labels:
                if len(pred_labels[key]) > 1:
                        pred_labels[key] = pred_labels[key][-1]

        #print(pred_labels)
        return pred_labels

def calc_scores(true_file_path,pred_file_path,id=""):
        pred_list = read_prompt3_pred_file(pred_file_path)
        pred_labels = []
        for pred in pred_list:
                if isinstance(pred_list[pred], list):
                        if pred_list[pred][0] == 'enatilment':
                                pred_labels.append('entailment')
                        else:
                                pred_labels.append(pred_list[pred][0])
                else:
                        if pred_list[pred] == 'enatilment':
                                pred_labels.append('entailment')
                        else:
                                pred_labels.append(pred_list[pred])

        #print(pred_labels)
        #print(list(set(pred_labels)))


        with open(true_file_path,'r') as file:
                ch_true = json.load(file)

        true_labels = []
        for key in ch_true:
                label = ch_true[key][2].strip()
                true_labels.append(label.lower())
        #print(true_labels)
        overlap  = sum(1 for x, y in zip(true_labels, pred_labels) if x == y)
        mismatch = sum(1 for x, y in zip(true_labels, pred_labels) if x != y)

        #print("Overlap between true and pred:", overlap)
        #print("Mismatch between true and pred: ",mismatch)

        from collections import Counter
        pred_freq = Counter(pred_labels)
        true_freq = Counter(true_labels)

        #print(f"The pred freq is:{pred_freq}")
        #print(f"The true freq is:{true_freq}")

        #print(len(true_labels))
        #print(len(pred_labels))
        if id == "C_negH":
                print(f"The accuracy for RTE (C negH) is:{accuracy_score(true_labels,pred_labels)}")
        elif id == "negC_H":
                print(f"The accuracy for RTE (negC H) is:{accuracy_score(true_labels, pred_labels)}")
        elif id == "negC_negH":
                print(f"The accuracy for RTE (negC negH) is:{accuracy_score(true_labels, pred_labels)}")

if __name__ == "__main__":
        print("-----------------------------------------")
        print("LLAMA2 SOLO Scores for RTE")
        print("-----------------------------------------")
        calc_scores(
                true_file_path=C_negH_path,
                pred_file_path=llama_C_negH,
                id='C_negH'
        )
        print("-----------------------------------------")
        calc_scores(
                true_file_path=negC_H_path,
                pred_file_path=llama_negC_H,
                id='negC_H'
        )
        print("-----------------------------------------")
        calc_scores(
                true_file_path=negC_negH_path,
                pred_file_path=llama_negC_negH,
                id='negC_negH'
        )
        print("-----------------------------------------")
        print("-----------------------------------------")

