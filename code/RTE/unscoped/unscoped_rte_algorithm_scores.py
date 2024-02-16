import json
from sklearn.metrics import accuracy_score
import re

HC_path              = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_HC_split.json"
CH_path              = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_CH_split.json"
C_negH_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_C_negH_split.json"
negC_negH_path       = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_negC_negH_split.json"
negC_H_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_negC_H_split.json"

llama_HC_path        = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/llama_predictions/rte_llama_HC.json"
llama_CH_path        = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/llama_predictions/rte_llama_CH.json"

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

        return pred_labels
def extract_content(input_string):
        matches = re.findall(r'\[/INST\](.*)', input_string,re.DOTALL)
        if matches:
                #return match.group(1)
                return matches[-1].strip()
        else:
                return None

def remove_special_characters(input_string):
        # Remove special characters using regex
        return re.sub(r'[^a-zA-Z0-9\s]', '', input_string)

def calc_c_noth(pos_path, true_labels,rev_pos_split,llama_op=True):
        """
        :param revised_pos_path: contains the path to positive sentences containing (CH) type samples
        :param revised_neg_path: contains the path to neg sentences containing (C neg H) type samples
        :return: output the predictions for (C neg H) using our algorithm
        """

        if llama_op:
                ch_true = read_prompt3_pred_file(pos_path)
        else:
                with open(pos_path,'r') as file:
                        ch_true = json.load(file)

        with open(true_labels, 'r') as file:
                c_negh_true = json.load(file)

        if llama_op:
                hc_true = read_prompt3_pred_file(rev_pos_split)
        else:
                with open(rev_pos_split,'r') as file:
                        hc_true = json.load(file)

        ch_labels = {}
        for j, ch in enumerate(ch_true):
                if llama_op:
                        if isinstance(ch_true[ch], list):
                                ch_labels[j] = ch_true[ch][0].strip().lower()
                        else:
                                ch_labels[j] = ch_true[ch].strip().lower()
                else:
                        label = remove_special_characters(ch_true[ch][2]).strip('\t').strip().lower()
                        ch_labels[j] = label

        c_negh_pred = {}

        not_ent_index = []
        ent_index = []
        for key in ch_labels:
                if llama_op:
                        if isinstance(ch_labels[int(key)],list):
                                pos_label = ch_labels[int(key)][0].strip().lower()
                        else:
                                pos_label = ch_labels[int(key)].strip().lower()
                else:
                        pos_label = ch_labels[key]

                if llama_op:
                        if isinstance(hc_true[str(key)],list):
                                rev_pos_label = hc_true[str(key)][0].strip().lower()
                        else:
                                rev_pos_label = hc_true[str(key)].strip().lower()

                else:
                        rev_pos_label = hc_true[str(key)][2].strip().lower()
                if pos_label == 'entailment':
                        c_negh_pred[key] = 'notentailment'
                elif pos_label == 'notentailment' and rev_pos_label == 'entailment':
                        c_negh_pred[key] = 'notentailment'
                elif pos_label == 'notentailment' and rev_pos_label == 'notentailment':
                        c_negh_pred[key] = 'notentailment'

        pred_labels = []
        for label in c_negh_pred:
                pred_neg = c_negh_pred[label]
                pred_labels.append(pred_neg)

        true_labels = []
        ent_count = 0
        noent_count = 0

        for key in c_negh_true:
                true_label = c_negh_true[key][2]
                if int(key) in list(c_negh_pred.keys()):
                        true_labels.append(true_label)

                """
                if int(key) not in not_ent_index:
                        true_label = c_negh_true[key][2]
                        true_labels.append(true_label)
                        if true_label == 'entailment':
                                ent_count = ent_count + 1
                        else:
                                noent_count = noent_count + 1

                """

        print(f"Predicting for {len(pred_labels)} samples!")

        if llama_op:
                print(f"The accuracy for (C, negH) using the llama predictions is :{accuracy_score(true_labels, pred_labels)}")
        else:
                print(f"The accuracy for (C, negH) using gold annotations is :{accuracy_score(true_labels, pred_labels)}")


def calc_notc_noth(pos_path, true_labels,rev_pos_split,llama_op=True,flag='score'):
        if llama_op:
                ch_true = read_prompt3_pred_file(pos_path)
        else:
                with open(pos_path,'r') as file:
                        ch_true = json.load(file)

        if llama_op:
                hc_true = read_prompt3_pred_file(rev_pos_split)
        else:
                with open(rev_pos_split,'r') as file:
                        hc_true = json.load(file)

        if flag == 'score':
                with open(true_labels, 'r') as file:
                        negc_negh_true = json.load(file)
        else:
                negc_negh_true = {}

        negc_negh_pred = {}
        no_ent_index = []

        ch_labels = {}
        for j, ch in enumerate(ch_true):
                if llama_op:
                        if isinstance(ch_true[ch], list):
                                ch_labels[j] = ch_true[ch][0].strip().lower()
                        else:
                                ch_labels[j] = ch_true[ch].strip().lower()
                else:
                        label = remove_special_characters(ch_true[ch][2]).strip('\t').strip().lower()
                        ch_labels[j] = label

        for key in ch_true:
                if llama_op:
                        if isinstance(ch_labels[int(key)], list):
                                pos_label = ch_labels[int(key)][0].strip().lower()
                        else:
                                pos_label = ch_labels[int(key)].strip().lower()
                else:
                        pos_label = ch_labels[int(key)]

                if llama_op:
                        if isinstance(hc_true[str(key)], list):
                                rev_pos_label = hc_true[str(key)][0].strip().lower()
                        else:
                                rev_pos_label = hc_true[str(key)].strip().lower()

                else:
                        rev_pos_label = hc_true[str(key)][2].strip().lower()

                assert pos_label == 'entailment' or pos_label == 'notentailment'
                assert rev_pos_label == 'entailment' or rev_pos_label == 'notentailment'

                if rev_pos_label == 'notentailment':
                        negc_negh_pred[key] = 'notentailment'
                elif rev_pos_label == 'entailment':
                        negc_negh_pred[key] = 'entailment'
                elif (pos_label == 'notentailment' and rev_pos_label == 'entailment') or (pos_label == 'entailment' and rev_pos_label == 'notentailment'):
                        continue
                else:
                        negc_negh_pred[key] = 'notentailment'

        true_labels = []

        for key in negc_negh_true:
                true_label = negc_negh_true[key][2]

                if key in list(negc_negh_pred.keys()):
                        true_labels.append(true_label)


        pred_labels = list(negc_negh_pred.values())
        if flag == 'score':
                print(f"Predicting for {len(pred_labels)} samples!")
                if llama_op:
                        print(f"The accuracy for (negC, negH) using the llama predictions is :{accuracy_score(true_labels, pred_labels)}")
                else:
                        print(f"The accuracy for (negC, negH) using gold annotations is :{accuracy_score(true_labels, pred_labels)}")

        elif flag == 'use':
                return negc_negh_pred
def calc_notc_h(pos_path, true_labels,rev_pos_split,llama_op=True):
        negc_negh_pred = calc_notc_noth(
                pos_path=pos_path,
                true_labels="",
                rev_pos_split=rev_pos_split,
                llama_op=llama_op,
                flag='use'
        )

        if llama_op:
                ch_true = read_prompt3_pred_file(pos_path)
        else:
                with open(pos_path,'r') as file:
                        ch_true = json.load(file)

        if llama_op:
                hc_true = read_prompt3_pred_file(rev_pos_split)
        else:
                with open(rev_pos_split,'r') as file:
                        hc_true = json.load(file)

        with open(true_labels, 'r') as file:
                negc_h_true = json.load(file)

        negc_h_pred = {}
        no_ent_index = []

        ch_labels = {}
        for j, ch in enumerate(ch_true):
                if llama_op:
                        if isinstance(ch_true[ch], list):
                                ch_labels[j] = ch_true[ch][0].strip().lower()
                        else:
                                ch_labels[j] = ch_true[ch].strip().lower()
                else:
                        label = remove_special_characters(ch_true[ch][2]).strip('\t').strip().lower()
                        ch_labels[j] = label

        for key in ch_true:
                if llama_op:
                        if isinstance(ch_labels[int(key)], list):
                                pos_label = ch_labels[int(key)][0].strip().lower()
                        else:
                                pos_label = ch_labels[int(key)].strip().lower()
                else:
                        pos_label = ch_labels[int(key)]

                if llama_op:
                        if isinstance(hc_true[str(key)], list):
                                rev_pos_label = hc_true[str(key)][0].strip().lower()
                        else:
                                rev_pos_label = hc_true[str(key)].strip().lower()

                else:
                        rev_pos_label = hc_true[str(key)][2].strip().lower()

                negc_negh_pred_label = negc_negh_pred[key]

                assert negc_negh_pred_label == 'entailment' or negc_negh_pred_label == 'notentailment'
                assert pos_label == 'entailment' or pos_label == 'notentailment'
                assert rev_pos_label == 'entailment' or rev_pos_label == 'notentailment'

                if negc_negh_pred_label == 'entailment':
                        negc_h_pred[key] = 'notentailment'

                if (pos_label == 'notentailment' and rev_pos_label == 'entailment') or (pos_label == 'entailment' and rev_pos_label == 'notentailment'):
                        negc_h_pred[key] = 'notentailment'
                else:
                        negc_h_pred[key] = 'notentailment'
                        #continue


        true_labels = []

        for key in negc_h_true:
                true_label = negc_h_true[key][2]

                if key in list(negc_h_pred.keys()):
                        true_labels.append(true_label)


        pred_labels = list(negc_h_pred.values())
        print(f"Predicting for {len(pred_labels)} samples!")
        if llama_op:
                print(f"The accuracy for (negC, H) using the llama predictions is :{accuracy_score(true_labels, pred_labels)}")
        else:
                print(f"The accuracy for (negC, H) using gold annotations is :{accuracy_score(true_labels, pred_labels)}")


if __name__ == "__main__":
        print("-----------------------------------------")
        print("The Unscoped Algorithm Scores using the Gold Annotations")
        print("-----------------------------------------")
        calc_c_noth(
                pos_path=CH_path,
                true_labels=C_negH_path,
                rev_pos_split=HC_path,
                llama_op=False
        )
        print("-----------------------------------------")
        calc_notc_noth(
                pos_path=CH_path,
                true_labels=negC_negH_path,
                rev_pos_split=HC_path,
                llama_op=False
        )

        print("-----------------------------------------")
        calc_notc_h(
                pos_path=CH_path,
                true_labels=negC_H_path,
                rev_pos_split=HC_path,
                llama_op=False
        )
        print("-----------------------------------------")
        print("-----------------------------------------")
        print("The Unscoped Algorithm Scores using the LLAMA2 predictions")
        print("-----------------------------------------")
        calc_c_noth(
                pos_path=llama_CH_path,
                true_labels=C_negH_path,
                rev_pos_split=llama_HC_path,
                llama_op=True
        )
        print("-----------------------------------------")
        calc_notc_noth(
                pos_path=llama_CH_path,
                true_labels=negC_negH_path,
                rev_pos_split=llama_HC_path,
                llama_op=True
        )
        print("-----------------------------------------")
        calc_notc_h(
                pos_path=llama_CH_path,
                true_labels=negC_H_path,
                rev_pos_split=llama_HC_path,
                llama_op=True
        )
        print("-----------------------------------------")

