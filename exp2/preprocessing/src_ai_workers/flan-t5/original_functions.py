# original functions from Alizadeh et al. (2025)

import re
import numpy as np
from utils_src import task_to_display_labels

def process_output(completion: str) -> str:
    # Clean the string a bit
    # Replace with regex the substrics 'Answer', 'folks', 'Plain', 'River' regardless of whether they are cased or not
    completion = re.sub(r'(?i)Answer|folks|Plain|River|IN', '', completion)    
    answers = completion.strip().split(' ')
    
    # Return the first letter of the first string in the list
    return answers[0][0] if len(answers[0]) > 0 else ""

def process_zero_shotoutput(completion: str, task: int) -> str:
    # Clean the string a bit
    mapping_dict = task_to_display_labels[task]
    full_values = mapping_dict.get('full_name')
    short_values = mapping_dict.get('short_name')
    
    val_out = completion[0]
    for full_val in full_values:
        if full_val in completion:
            index_full = full_values.index(full_val)
            val_out = short_values[index_full]
            next
    return val_out

def map_outputs_task_1(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|(a(\.|:|\)))|(\s|^|\')relev(a|e)nt|aelevant', output.lower().strip()):
        return 'A'
    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|not relevant|irrelevant|ielevant|\s+b$|brrelevant', output.lower().strip()):
        return 'B'
    elif output == np.nan or output == 'nan':
        return ""
    else:
        print(f'Weird value: {output.lower().strip()}')
        return ""


def map_outputs_task_2(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|challnge|problem|\bpro\b|blem', output.lower().strip()):
        return 'A'
    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|solution|\blution\b', output.lower().strip()):
        return 'B'
    elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|neither|neutral|(\s)+c$', output.lower().strip()):
        return 'C'
    elif output == np.nan or output == 'nan':
        return ""
    else:
        print(f'Weird value: {output.lower().strip()}')
        return "np.nan"


def map_outputs_task_3(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|economic|economy|aconomy', output.lower().strip()):
        return 'A'

    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|morality|rality', output.lower().strip()):
        return 'B'

    elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|fairness and equality|irness and equality', output.lower().strip()):
        return 'C'

    elif re.search(r'^(answer:){0,1}(\s)*d(\s)*$|d(\.|:|\))|policy prescription and evaluation|prescription and evaluation|licy prescription',
                   output.lower().strip()):
        return 'D'

    elif re.search(r'^(answer:){0,1}(\s)*e(\s)*$|e(\.|:|\))|law and order|crime and justice|law enforcement|w and order', output.lower().strip()):
        return 'E'

    elif re.search(r'^(answer:){0,1}(\s)*f(\s)*$|f(\.|:|\))|security and defense|curity and defense', output.lower().strip()):
        return 'F'

    elif re.search(r'^(answer:){0,1}(\s)*g(\s)*$|g(\.|:|\))|health and safety|alth and safety', output.lower().strip()):
        return 'G'

    elif re.search(r'^(answer:){0,1}(\s)*h(\s)*$|h(\.|:|\))|quality of life|ality of life', output.lower().strip()):
        return 'H'

    elif re.search(r'^(answer:){0,1}(\s)*i(\s)*$|i(\.|:|\))|political|litical', output.lower().strip()):
        return 'I'

    elif re.search(r'^(answer:){0,1}(\s)*j(\s)*$|j(\.|:|\))|external (regulation|region) and reputation|external regulation|regulation and reputation', output.lower().strip()):
        return 'J'

    elif re.search(
            r'^(answer:){0,1}(\s)*k(\s)*$|(k|n|w)(\.|:|\))|other|climate change|leadership and executive responsibility|'
            r'expansion of service opportunities|access to higher ed|potential',
            output.lower().strip()):
        return 'K'

    elif output == np.nan or output == 'nan':
        return ""

    else:
        print(f'Weird value: {output.lower().strip()}')
        return ""


def map_outputs_task_4(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|positive|postive stance|in favor|in advantage of|aast|a favor of a', output.lower().strip()):
        return 'A'

    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|negative|negative stance|against|aggainst|bast', output.lower().strip()):
        return 'B'

    elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|neutral|neutral stance|cast', output.lower().strip()):
        return 'C'

    elif output == np.nan or output == 'nan':
        return ""

    else:
        print(f'Weird value: {output.lower().strip()}')
        return ""


def map_outputs_task_5(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|section 230|230', output.lower().strip()):
        return 'A'

    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|trump ban|ban donald trump|ban(ning){0,1} trump|tr ban', output.lower().strip()):
        return 'B'

    elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|twitter support', output.lower().strip()):
        return 'C'

    elif re.search(r'^(answer:){0,1}(\s)*d(\s)*$|d(\.|:|\))|platform policies|policies', output.lower().strip()):
        return 'D'

    elif re.search(r'^(answer:){0,1}(\s)*e(\s)*$|e(\.|:|\))|complaint(s)+', output.lower().strip()):
        return 'E'

    elif re.search('^(answer:){0,1}(\s)*f(\s)*$|f(\.|:|\))|other',
                   output.lower().strip()):
        return 'F'

    elif output == np.nan or output == 'nan':
        return ""

    else:
        print(f'Weird value: {output.lower().strip()}')
        return  ""
    
def map_outputs_task_6(output):
     if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|policy prescription|policy prescription and regulation|licy and regulation|alicy',
                   output.lower().strip()):
        return 'A'
     
     elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|morality|rality', output.lower().strip()):
        return 'B'
     
     elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|economics|econom|onomics', output.lower().strip()):
        return 'C'

     elif re.search(r'^(answer:){0,1}(\s)*d(\s)*$|d(\.|:|\))|other', output.lower().strip()):
        return 'D'

     elif output == np.nan or output == 'nan':
        return ""

     else:
        print(f'Weird value: {output.lower().strip()}')
        return ""

def process_output_completed(completion: str, task:int) -> str:
    completion = re.sub(r'(?i)Answer|folks|Plain|River|IN', '', completion)    
    answers = completion.strip().split(' ')
    if task == 1:
        return map_outputs_task_1(completion)
    if task == 2:
        return map_outputs_task_2(completion)
    if task == 3:
        return map_outputs_task_3(completion)
    if task == 4:
        return map_outputs_task_4(completion)
    if task == 5:
        return map_outputs_task_5(completion)
    if task == 6:
        return map_outputs_task_6(completion)

def process_output_new(completion: str, task:int) -> str:
    completion = re.sub(r'(?i)Answer|folks|Plain|River|IN', '', completion)    
    answers = completion.strip().split(' ')
    mapping_dict = task_to_display_labels[task]
    full_values = [val.lower() for val in mapping_dict.get('full_name')]
    short_values = [val.lower() for val in mapping_dict.get('short_name')]
    val_out = answers[0][0]
    if val_out.lower() in short_values:
        return val_out
    else:
        for full_val in full_values:
            if full_val in completion:
                index_full = full_values.index(full_val)
                val_out = short_values[index_full]
                break
        val_out = val_out if val_out is not None else ""
        if (val_out not in ["A", "B", "C" , "D"]) or (task == 3 and val_out not in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]):
            print(f"unrpocessed answer at {completion}")
        return val_out

