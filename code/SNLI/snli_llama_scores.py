import json
import re
from sklearn.metrics import accuracy_score

C_negH_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_c_negh_split_all.json"
negC_H_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_negc_h_split_all.json"
negC_negH_path       = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/SplittedData/snli_negc_negh_split_all.json"

llama_C_negH         = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/llama_predictions/C_negH_pred.json"
llama_negC_H         = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/llama_predictions/negC_H_pred.json"
llama_negC_negH      = "/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/llama_predictions/negC_negH_pred.json"

def read_result_file(op_file_path):
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
def calc_scores(true_file_path,pred_file_path,id=""):
        pred_ops  = read_result_file(pred_file_path)
        pred_keys = list(pred_ops.keys())
        #print(len(pred_keys))
        for i in range(len(pred_keys)):
                pred_keys[i] = str(pred_keys[i])

        with open(true_file_path,'r') as file:
                true_ops = json.load(file)

        true_labels  = []
        for key in true_ops:
                if key in pred_keys:
                        true_labels.append(true_ops[key][2])

        #print(len(true_labels))
        #print(len(pred_keys))

        pred_labels = []
        for key in pred_ops:
                pred_labels.append(pred_ops[key])

        #print(len(pred_labels))
        if id == "C_negH":
                print(f"The accuracy score for (C negH) is:{accuracy_score(true_labels,pred_labels)}")
        elif id == "negC_H":
                print(f"The accuracy score for (negC H) is:{accuracy_score(true_labels,pred_labels)}")
        elif id == "negC_negH":
                print(f"The accuracy score for (negC negH) is:{accuracy_score(true_labels,pred_labels)}")


if __name__ == "__main__":
        print("-----------------------------------------")
        print("LLAMA2 SOLO Scores for SNLI")
        print("-----------------------------------------")
        calc_scores(
                true_file_path=C_negH_path,
                pred_file_path=llama_C_negH,
                id="C_negH"
        )
        print("-----------------------------------------")
        calc_scores(
                true_file_path=negC_H_path,
                pred_file_path=llama_negC_H,
                id="negC_H"
        )
        print("-----------------------------------------")
        calc_scores(
                true_file_path=negC_negH_path,
                pred_file_path=llama_negC_negH,
                id="negC_negH"
        )
        print("-----------------------------------------")
        print("-----------------------------------------")