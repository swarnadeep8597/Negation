import json
import re
from sklearn.metrics import accuracy_score


from spacy.lang.en import English
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer


"""
The file paths 
"""
input_file_path      = "/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/RTE-Hossain/rev-data/RTE-neg-na-filtered.txt"
HC_path              = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_HC_split.json"
CH_path              = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_CH_split.json"
C_negH_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_C_negH_split.json"
scope_path           = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_negC_h_split_scope.json"
negC_H_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_negC_H_split.json"
negC_negH_pred_path  = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/intermediate/golden_negc_negh_pred.json"
scoped_path          = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_negC_h_split_scope.json"
negC_negH_path       = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/rte_negC_negH_split.json"
HC_prime_path        = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/HC_prime.json"


llama_HC_path        = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/llama_predictions/rte_llama_HC.json"
llama_CH_path        = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/llama_predictions/rte_llama_CH.json"
llama_scope_path     = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/llama_predictions/rte_llama_scope.json"
llama_HC_prime_path  = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/llama_predictions/llama_rte_prime.json"
llama_negC_negH_pred = "/Users/sbhar/Riju/PhDCode/ACL/data/RTE/SplittedData/intermediate/llama_negc_negh_pred.json"

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

        return pred_labels

def remove_special_characters(input_string):
        # Remove special characters using regex
        return re.sub(r'[^a-zA-Z0-9\s]', '', input_string)

def calc_negc_h_scope(pos_split,rev_pos_split,scoped_file,true_labels,negc_negh_pred_path,llama_op=True):
        if llama_op:
                ch_true = read_prompt3_pred_file(pos_split)
        else:
                with open(pos_split,'r') as file:
                        ch_true = json.load(file)

        if llama_op:
                hc_true = read_prompt3_pred_file(rev_pos_split)
        else:
                with open(rev_pos_split,'r') as file:
                        hc_true = json.load(file)



        if llama_op:
                scoped_negc_h = read_prompt3_pred_file(scoped_file)
        else:
                with open(scoped_file,'r') as file:
                        scoped_negc_h = json.load(file)


        with open(negc_negh_pred_path,'r') as file:
                negc_negh = json.load(file)

        with open(true_labels,'r') as file:
                negc_h_true = json.load(file)

        hc_labels = {}
        for i, rev in enumerate(hc_true):
                if llama_op:
                        if isinstance(hc_true[rev], list):
                                hc_labels[i] = hc_true[rev][0]
                        else:
                                hc_labels[i] = hc_true[rev]
                else:
                        label = remove_special_characters(hc_true[rev][2]).strip('\t').strip().lower()
                        hc_labels[i] = label

        ch_labels = {}
        for j, ch in enumerate(ch_true):
                if llama_op:
                        if isinstance(ch_true[ch],list):
                                ch_labels[j] = ch_true[ch][0]
                        else:
                                ch_labels[j] = ch_true[ch]
                else:
                        label = remove_special_characters(ch_true[ch][2]).strip('\t').strip().lower()
                        ch_labels[j] = label

        scoped_labels = {}
        for k,scp in enumerate(scoped_negc_h):
                if llama_op:
                        if isinstance(scoped_negc_h[scp],list):
                                scoped_labels[k] = scoped_negc_h[scp][0]
                        else:
                                scoped_labels[k] = scoped_negc_h[scp]
                else:
                        label = remove_special_characters(scoped_negc_h[scp][2]).strip('\t').strip().lower()
                        scoped_labels[k] = label

        negc_negh_pred = {}
        for l,nc_nh in enumerate(negc_negh):
                label = negc_negh[nc_nh]
                negc_negh_pred[int(l)] = label

        #print(len(negc_negh_pred.values()))
        #return
        negc_h_pred = {}
        ent_count = 0
        for rev in ch_labels:
                ch_label = ch_labels[rev]
                if ch_label == 'entailment':
                        ent_count = ent_count + 1

        not_ent_count = 0
        for rev in hc_labels:
                hc_label    = hc_labels[rev]
                ch_label    = ch_labels[rev]
                sc_label    = scoped_labels[rev]
                nc_nh_label = negc_negh_pred[int(rev)]

                if hc_label == 'entailment':
                        negc_h_pred[rev] = 'notentailment'
                if hc_label == 'notentailment':
                        if ch_label == 'entailment':
                                if sc_label == 'entailment':
                                        negc_h_pred[rev] = 'entailment'
                                else:
                                        negc_h_pred[rev] = 'notentailment'

                        if ch_label == 'notentailment':
                                not_ent_count = not_ent_count + 1
                                #negc_h_pred[rev] = 'notentailment'
                                if nc_nh_label == 'notentailment':
                                        negc_h_pred[rev] = 'notentailment'

        #print(len(negc_h_pred))
        from collections import Counter
        frequency = Counter(list(negc_h_pred.values()))
        #print(frequency)

        nn_not = 0
        for label in negc_negh_pred:
                pred_l = negc_negh_pred[label]
                if pred_l == 'notentailment':
                        nn_not = nn_not + 1

        #print("Neg Neg not entailment",nn_not)
        #print("ch not entailment: ",not_ent_count)

        true_labels = []
        for true in negc_h_true:
                if int(true) in list(negc_h_pred.keys()):
                        true_label = remove_special_characters(negc_h_true[true][2]).strip('\t').strip().lower()
                        true_labels.append(true_label)

        print(f"Predicting for {len(true_labels)} samples!")
        if llama_op:
                print(f"The accuracy scores for (neg C,h) using llama outputs is: {accuracy_score(true_labels,list(negc_h_pred.values()))}")
        else:
                print(f"The accuracy scores for (neg C,h) using gold annotations is: {accuracy_score(true_labels,list(negc_h_pred.values()))}")

def calc_negc_negh_scope(rev_pos_split,scoped_file,true_labels,star_anot,llama_op=True):
        if llama_op:
                hc_true = read_prompt3_pred_file(rev_pos_split)
        else:
                with open(rev_pos_split,'r') as file:
                        hc_true = json.load(file)



        with open(true_labels,'r') as file:
                negc_negh_true = json.load(file)

        if llama_op:
                scoped_negc_h = read_prompt3_pred_file(scoped_file)
        else:
                with open(scoped_file,'r') as file:
                        scoped_negc_h = json.load(file)

        if llama_op:
                hc_pr = read_prompt3_pred_file(star_anot)
        else:
                with open(star_anot,'r') as file:
                        hc_pr = json.load(file)



        hc_labels = {}
        for i, rev in enumerate(hc_true):
                #label = remove_special_characters(hc_true[rev][2]).strip('\t').strip().lower()
                #hc_labels[i] = label

                if llama_op:
                        if isinstance(hc_true[rev],list):
                                hc_labels[i] = hc_true[rev][0]
                        else:
                                hc_labels[i] = hc_true[rev]
                else:
                        label = remove_special_characters(hc_true[rev][2]).strip('\t').strip().lower()
                        hc_labels[i] = label

        hc_prime_labels = {}
        annotated_labels = [39,49,55,97,108,121,123,133,153]
        for i, rev in enumerate(hc_pr):
                if int(rev) in annotated_labels:
                        if llama_op:
                                if isinstance(hc_pr[rev], list):
                                        hc_prime_labels[rev] = hc_pr[rev][0]
                                else:
                                        hc_prime_labels[rev] = hc_pr[rev]
                        else:
                                label = remove_special_characters(hc_pr[rev][-1]).strip('\t').strip().lower()
                                hc_prime_labels[rev] = label

        #print("The prime labels are: ",len(hc_prime_labels))
        #print(hc_prime_labels['39'])
        scoped_labels = {}
        for k, scp in enumerate(scoped_negc_h):
                if llama_op:
                        if isinstance(scoped_negc_h[scp], list):
                                scoped_labels[k] = scoped_negc_h[scp][0]
                        else:
                                scoped_labels[k] = scoped_negc_h[scp]
                else:
                        label = remove_special_characters(scoped_negc_h[scp][2]).strip('\t').strip().lower()
                        scoped_labels[k] = label

        negc_negh_pred = {}
        for rev in hc_labels:
                hc_label  = hc_labels[rev]
                scp_label = scoped_labels[rev]

                if hc_label == 'entailment':
                        if scp_label == 'entailment':
                                negc_negh_pred[rev] = 'notentailment'
                        else:
                                negc_negh_pred[rev] = 'entailment'
                if hc_label == 'notentailment' :
                        if int(rev) in annotated_labels:
                                if hc_prime_labels[str(rev)] == 'entailment':
                                        #print(f"The prime labels as entailments: ",rev)
                                        negc_negh_pred[rev] = 'entailment'
                                else:
                                        negc_negh_pred[rev] = 'notentailment'
                        else:
                                #negc_negh_pred[rev] = 'notentailment'
                                negc_negh_pred[rev] = 'notentailment'


                """
                if hc_label == 'entailment':
                        if scp_label == 'entailment':
                                negc_negh_pred[rev] = 'notentailment'
                        else:
                                negc_negh_pred[rev] = 'entailment'
                if hc_label == 'notentailment' :
                        negc_negh_pred[rev] = 'notentailment'

                """

        #print("The hc label length is: ",len(hc_labels))
        #print("The neg c neg h pred samples length: ",len(negc_negh_pred))

        true_labels = []
        mis_match_labels = []
        for true in negc_negh_true:
                if int(true) in list(negc_negh_pred.keys()):
                        true_label = remove_special_characters(negc_negh_true[true][2]).strip('\t').strip().lower()
                        true_labels.append(true_label)
                        if true_label != negc_negh_pred[int(true)]:
                                mis_match_labels.append(true)


        negc_negh_mismatch = {}
        for label in mis_match_labels:
                negc_negh_mismatch[label] = negc_negh_true[label]
                negc_negh_mismatch[label][2] = negc_negh_pred[int(label)]

        #print(len(negc_negh_mismatch))

        """
        with open('/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/Llama_Code/source_data/rte/intermediate/negC_negH_mismatch.json','w') as file:
                json.dump(negc_negh_mismatch,file)
        """

        #print(len(mis_match_labels))
        mis_match_samples = {}
        for true in negc_negh_true:
                if true in mis_match_labels:
                        mis_match_samples[true] = negc_negh_true[true]


        if llama_op:
                with open(llama_negC_negH_pred,'w') as file:
                        json.dump(negc_negh_pred,file)
        else:
                with open(negC_negH_pred_path,'w') as file:
                        json.dump(negc_negh_pred,file)

        """
        #print(len(negc_negh_pred))
        with open('/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/RTE-Hossain/rev-data/negc_negh_mismatch.json','w') as file:
                json.dump(mis_match_samples,file)
        """
        #print(list(negc_negh_pred.values()))
        #print(true_labels)

        print(f"Predicting for {len(true_labels)} samples!")
        if llama_op:
                print(f"The accuracy scores for (neg C,neg H) using llama outputs is: {accuracy_score(true_labels,list(negc_negh_pred.values()))}")
        else:
                print(f"The accuracy scores for (neg C,neg H) using gold annotations is: {accuracy_score(true_labels,list(negc_negh_pred.values()))}")

        #overlap = sum(1 for x, y in zip(true_labels, list(negc_negh_pred.values())) if x != y)

        #print(negc_negh_pred[18])
        #print("Mismatches between true and pred:", overlap)
def calc_c_negh_scope(rev_pos_split,pos_split,true_labels,llama_op=True):
        if llama_op:
                ch_true = read_prompt3_pred_file(pos_split)
        else:
                with open(pos_split,'r') as file:
                        ch_true = json.load(file)


        if llama_op:
                hc_true = read_prompt3_pred_file(rev_pos_split)
        else:
                with open(rev_pos_split,'r') as file:
                        hc_true = json.load(file)

        with open(true_labels,'r') as file:
                c_negh_true = json.load(file)

        hc_labels = {}
        for i, rev in enumerate(hc_true):
                if llama_op:
                        if isinstance(hc_true[rev],list):
                                hc_labels[i] = hc_true[rev][0].strip().lower()
                        else:
                                hc_labels[i] = hc_true[rev].strip().lower()
                else:
                        label = remove_special_characters(hc_true[rev][2]).strip('\t').strip().lower()
                        hc_labels[i] = label

        ch_labels = {}
        for j, ch in enumerate(ch_true):
                if llama_op:
                        if isinstance(ch_true[ch],list):
                                ch_labels[j] = ch_true[ch][0].strip().lower()
                        else:
                                ch_labels[j] = ch_true[ch].strip().lower()
                else:
                        label = remove_special_characters(ch_true[ch][2]).strip('\t').strip().lower()
                        ch_labels[j] = label

        """
        Delete later this counter part
        """
        from collections import Counter
        frequency = Counter(list(ch_labels.values()))
        #print(frequency)

        c_negh_pred = {}
        for pos in ch_labels:
                ch_label = ch_labels[pos]
                if ch_label == 'entailment':
                        c_negh_pred[pos] = 'notentailment'
                elif ch_label == 'notentailment':
                        hc_label = hc_labels[pos]
                        if hc_label == 'notentailment':
                                c_negh_pred[pos] = 'notentailment' #---(1)
                        if hc_label == 'entailment':
                                c_negh_pred[pos] = 'notentailment'

        #print(len(c_negh_pred))
        pred_freq = Counter(list(c_negh_pred.values()))
        true_freq = Counter(c_negh_true)
        #print(f"The predicted labels frequency is: ",pred_freq)


        true_labels = []
        mismatches  = []
        for true in c_negh_true:
                if int(true) in list(c_negh_pred.keys()):
                        true_label = remove_special_characters(c_negh_true[true][2]).strip('\t').strip().lower()
                        true_labels.append(true_label)
                        if true_label != c_negh_pred[int(true)]:
                                mismatches.append(true)

        true_freq = Counter(true_labels)
        #print(f"The true labels frequency is: ", true_freq)
        #print("The mismatches are: ",len(mismatches))
        c_negh_mismatch = {}
        for j, ch in enumerate(c_negh_true):
                assert str(j) == ch
                if str(j) in mismatches:
                        c_negh_mismatch[j] = c_negh_true[ch]

        #print("The mismatches are: ",len(c_negh_mismatch))
        """
        with open('/Users/sbhar/Riju/PhDCode/Naacl_Neg/code/RTE-Hossain/rev-data/c_negh_mismatch.json','w') as file:
                json.dump(c_negh_mismatch,file)
        """
        #print(len(true_labels))
        print(f"Predicting for {len(true_labels)} samples!")
        if llama_op:
                print(f"The accuracy for (C, negH) using the llama predictions is : {accuracy_score(true_labels,list(c_negh_pred.values()))}")
        else:
                print(f"The accuracy for (C, negH) using gold annotations is : {accuracy_score(true_labels,list(c_negh_pred.values()))}")


if __name__ == "__main__":
        print("-----------------------------------------")
        print("The Scores for RTE")
        print("-----------------------------------------")
        print("The Scoped Algorithm Scores using the Gold Annotations")
        print("-----------------------------------------")
        calc_c_negh_scope(
                rev_pos_split=HC_path,
                pos_split=CH_path,
                true_labels=C_negH_path,
                llama_op=False
        )
        print("-----------------------------------------")
        calc_negc_negh_scope(
                rev_pos_split=HC_path,
                scoped_file=scoped_path,
                true_labels=negC_negH_path,
                star_anot=HC_prime_path,
                llama_op=False
        )
        print("-----------------------------------------")
        calc_negc_h_scope(
                pos_split=CH_path,
                rev_pos_split=HC_path,
                scoped_file=scope_path,
                true_labels=negC_H_path,
                negc_negh_pred_path=negC_negH_pred_path,
                llama_op=False
        )
        print("-----------------------------------------")
        print("-----------------------------------------")
        print("The Scoped Algorithm Scores using the LLAMA2 predictions")
        print("-----------------------------------------")
        calc_c_negh_scope(
                rev_pos_split=llama_HC_path,
                pos_split=llama_CH_path,
                true_labels=C_negH_path,
                llama_op=True
        )
        print("-----------------------------------------")
        calc_negc_negh_scope(
                rev_pos_split=llama_HC_path,
                scoped_file=llama_scope_path,
                true_labels=negC_negH_path,
                star_anot=llama_HC_prime_path,
                llama_op=True
        )
        print("-----------------------------------------")
        calc_negc_h_scope(
                pos_split=llama_CH_path,
                rev_pos_split=llama_HC_path,
                scoped_file=llama_scope_path,
                true_labels=negC_H_path,
                negc_negh_pred_path=llama_negC_negH_pred,
                llama_op=True
        )
        print("-----------------------------------------")
