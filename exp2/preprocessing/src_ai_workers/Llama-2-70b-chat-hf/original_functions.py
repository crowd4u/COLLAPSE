# original functions from Alizadeh et al. (2025)

import re
import numpy as np

def map_outputs_task_1(output):
    # Check first if it matches with the start of the sentence
    if re.search(r'^(\s)*RELEVANT', output.strip()):
        return 'RELEVANT'
    elif re.search(r'^(\s)*IRRELEVANT', output.strip()):
        return 'IRRELEVANT'
    elif re.search(r'\s(RELEVANT|RE|RELEVENT|RELEVAL|relevant|RELEV)', output.strip()):
        return 'RELEVANT'
    elif re.search(r'IRRELEVANT|IRRELE|does not mention|does not seem to refer ', output.strip()):
        return 'IRRELEVANT'
    elif output == np.nan or output == 'nan':
        return "NAN"
    else:
        print(f'Weird value: {output.strip()}')
        return "NAN"


def map_outputs_task_2(output):
    if re.search(r'^(\s)*PROBLEM', output.strip()):
        return 'PROBLEM'
    elif re.search(r'^(\s)*SOLUTION', output.strip()):
        return 'SOLUTION'
    elif re.search(r'^(\s)*(NEITHER|NEUTRAL)|Therefore, I would classify this tweet as NEUTRAL|I think it is NEUTRAL', output.strip()):
        return 'NEUTRAL'

    elif re.search(r'PROBLEM|(t|T)he tweet describes content moderation as a problem|'
                 r'described as a problem', output.strip()):
        return 'PROBLEM'
    elif re.search(r'SOLUTION', output.strip()):
        return 'SOLUTION'
    elif re.search(r'(NEITHER|NEUTRAL|NE|neutral)', output.strip()):
        return 'NEUTRAL'
    elif output == np.nan or output == 'nan':
        return "NAN"
    else:
        print(f'Weird value: {output.strip()}')
        return "NAN"

def map_outputs_task_3(output):
    if re.search(r'^(\s*A:){0,1}(\s)*ECONOMY', output.strip()):
        return 'ECONOMY'

    elif re.search(r'^(\s*B:){0,1}(\s)*MORALITY', output.strip()):
        return 'MORALITY'

    elif re.search(r'^(\s*C:){0,1}(\s)*FAIRNESS AND EQUALITY', output.strip()):
        return 'FAIRNESS AND EQUALITY'

    elif re.search(r'^(\s*D:){0,1}(\s)*POLICY PRESCRIPTION AND EVALUATION', output.strip()):
        return 'POLICY PRESCRIPTION AND EVALUATION'

    elif re.search(r'^(\s*E:){0,1}(\s)*LAW AND ORDER, CRIME AND JUSTICE', output.strip()):
        return 'LAW AND ORDER, CRIME AND JUSTICE'

    elif re.search(r'^(\s*F:){0,1}(\s)*SECURITY AND DEFENSE', output.strip()):
        return 'SECURITY AND DEFENSE'

    elif re.search(r'^(\s*G:){0,1}(\s)*HEALTH AND SAFETY', output.strip()):
        return 'HEALTH AND SAFETY'

    elif re.search(r'^(\s*H:){0,1}(\s)*QUALITY OF LIFE', output.strip()):
        return 'QUALITY OF LIFE'

    elif re.search(r'^(\s*I:){0,1}(\s)*POLITICAL', output.strip()):
        return 'POLITICAL'

    elif re.search(r'^(\s*J:){0,1}(\s)*EXTERNAL REGULATION AND REPUTATION', output.strip()):
        return 'EXTERNAL REGULATION AND REPUTATION'

    elif re.search(r'^(\s*K:){0,1}(\s)*OTHER', output.strip()):
        return 'OTHER'
    
    elif re.search(r'(\s*A:){0,1}(\s)*ECONOMY|^\s*A|ECONEY', output.strip()):
        return 'ECONOMY'

    elif re.search(r'(\s*B:){0,1}(\s)*MORALITY|^\s*B(\s+|$)', output.strip()):
        return 'MORALITY'

    elif re.search(r'(\s*C:){0,1}(\s)*FAIRNESS AND EQUALITY|EQUALITY AND FAIRNESS|FAIRNESS|It is concerned with the fairness and equality', output.strip()):
        return 'FAIRNESS AND EQUALITY'

    elif re.search(r'(\s*D:){0,1}(\s)*POLICY PRESCRIPTION AND EVALUATION', output.strip()):
        return 'POLICY PRESCRIPTION AND EVALUATION'

    elif re.search(r'(\s*E:){0,1}(\s)*LAW AND ORDER(, CRIME AND JUSTICE){0,1}', output.strip()):
        return 'LAW AND ORDER, CRIME AND JUSTICE'

    elif re.search(r'(\s*F:){0,1}(\s)*SECURITY AND DEFENSE', output.strip()):
        return 'SECURITY AND DEFENSE'

    elif re.search(r'(\s*G:){0,1}(\s)*HEALTH AND SAFETY', output.strip()):
        return 'HEALTH AND SAFETY'

    elif re.search(r'(\s*H:){0,1}(\s)*QUALITY OF LIFE|^\s*H.|quality of life|H]', output.strip()):
        return 'QUALITY OF LIFE'

    elif re.search(r'(\s*I:){0,1}(\s)*(POLITICAL|POLICITAL|POLIT)', output.strip()):
        return 'POLITICAL'

    elif re.search(r'(\s*J:){0,1}(\s)*EXTERNAL REGULATION AND REPUTATION', output.strip()):
        return 'EXTERNAL REGULATION AND REPUTATION'

    elif re.search(r'(\s*K:){0,1}(\s)*OTHER|EDUCATION|does not seem to fit into|does not fit neatly into|None of the above|^\s*Other\s*$', output.strip()):
        return 'OTHER'

    elif output == np.nan or output == 'nan':
        return "NAN"

    else:
        print(f'Weird value: {output.strip()}')
        return "NAN"


def map_outputs_task_4(output):
    if re.search(r'^(\s)*IN FAVOR OF', output.strip()):
        return 'IN FAVOR OF'

    elif re.search(r'^(\s)*AGAINST', output.strip()):
        return 'AGAINST'

    elif re.search(r'^(\s)*NEUTRAL', output.strip()):
        return 'NEUTRAL'
    
    elif re.search(r'IN FAVOR OF|IN FAV', output.strip()):
        return 'IN FAVOR OF'

    elif re.search(r'AGAINST', output.strip()):
        return 'AGAINST'

    elif re.search(r'NEUTRAL|(without|not) express(ing){0,1} approval or disapproval|^NEUT$', output.strip()):
        return 'NEUTRAL'

    elif output == np.nan or output == 'nan':
        return "NAN"

    else:
        print(f'Weird value: {output.strip()}')
        return "NAN"


def map_outputs_task_5(output):
    if re.search(r'^(the tweet is (about (the){0,1})){0,1}(\s)*section 230', output.lower().strip()):
        return 'Section 230'

    elif re.search(r'^(the tweet is about (the){0,1}){0,1}(\s)*trump ban', output.lower().strip()):
        return 'Trump ban'

    elif re.search(r'^(the tweet is about (the){0,1}){0,1}(\s)*twitter support', output.lower().strip()):
        return 'Twitter Support'

    elif re.search(r'^(the tweet is about (the){0,1}){0,1}(\s)*platform policies', output.lower().strip()):
        return 'Platform Policies'

    elif re.search(r'^(the tweet is about (the){0,1}){0,1}(\s)*complaint', output.lower().strip()):
        return 'Complaint'

    elif re.search('^(the tweet is about (the){0,1}){0,1}(\s)*other', output.lower().strip()):
        return 'Other'
    
    elif re.search('OTHER|Therefore, it (should|can) be classified as "Other.|Other"',
                   output.strip()):
        return 'Other'
    
    elif re.search(r'SECTION 230|classified as A: Section 230|"Section 230"|A: Section 2', output.strip()):
         return 'Section 230'

    elif re.search(r'TRUMP BAN|Trump Ban|B: Trump ban|B Trump ban|B(\s+|$)|B (Trump ban)', output.strip()):
        return 'Trump ban'

    elif re.search(r'TWITTER SUPPORT|classified as C \(Twitter Support\).', output.strip()):
        return 'Twitter Support'

    elif re.search(r'PLATFORM POLICIES|Platform Policies', output.strip()):
        return 'Platform Policies'

    elif re.search(r'COMPLAINT|E: Complaint|"Complaints"|E \(Complaint\)|E(\s+|$)|classified as \(E\) Complaint', output.strip()):
        return 'Complaint'


    # 
    # if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|section 230', output.strip()):
    #     return 'Section 230'
    # 
    # elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|trump ban|ban donald trump|ban(ning){0,1} trump', output.strip()):
    #     return 'Trump ban'
    # 
    # elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|twitter support', output.strip()):
    #     return 'Twitter Support'
    # 
    # elif re.search(r'^(answer:){0,1}(\s)*d(\s)*$|d(\.|:|\))|platform policies', output.strip()):
    #     return 'Platform Policies'
    # 
    # elif re.search(r'^(answer:){0,1}(\s)*e(\s)*$|e(\.|:|\))|complaint(s)+', output.strip()):
    #     return 'Complaint'
    # 
    # elif re.search('^(answer:){0,1}(\s)*f(\s)*$|f(\.|:|\))|other|censorship|banning free speech|'
    #                'blocking political opinions|state bankruptcy|republique|problems with tech platform|'
    #                'problem with the application',
    #                output.strip()):
    #     return 'Other'
    # 
    elif output == np.nan or output == 'nan':
        return "NAN"

    else:
        print(f'Weird value: {output.strip()}')
        return 'Other'
    
def map_outputs_task_6(output):
    if re.search(r'^(\s*A:){0,1}(\s)*POLICY AND REGULATION', output.strip()):
        return 'POLICY AND REGULATION'
    
    elif re.search(r'^(\s*B:){0,1}(\s)*MORALITY AND LAW', output.strip()):
        return 'MORALITY AND LAW'
    
    elif re.search(r'^(\s*C:){0,1}(\s)*ECONOMICS', output.strip()):
        return 'ECONOMICS'
    
    elif re.search(r'^(\sD:){0,1}(\s)*OTHER', output.strip()):
        return 'OTHER'

    elif re.search(r'(\s*C:){0,1}(\s)*ECONOM(Y|ICS)|^\s*A', output.strip()):
        return 'ECONOMICS'
    
    elif re.search(r'(\s*C:){0,1}(\s)*PUBLIC OPINION|^\s*A', output.strip()):
        return 'ECONOMICS'

    elif re.search(r'(\s*B:){0,1}(\s)*MORALITY|^\s*B(\s+|$)', output.strip()):
        return 'MORALITY AND LAW'

    elif re.search(r'(\s*B:){0,1}(\s)*FAIRNESS AND EQUALITY', output.strip()):
        return 'MORALITY AND LAW'

    elif re.search(r'(\s*A:){0,1}(\s)*POLICY PRESCRIPTION AND EVALUATION|POLICY AND REGULATION', output.strip()):
        return 'POLICY AND REGULATION'

    elif re.search(r'(\s*B:){0,1}(\s)*LAW AND ORDER, CRIME AND JUSTICE', output.strip()):
        return 'MORALITY AND LAW'
    
    elif re.search(r'(\s*B:){0,1}(\s)*CONSTITUTIONALITY AND JURISPRUDENCE', output.strip()):
        return 'MORALITY AND LAW'

    elif re.search(r'(\s*C:){0,1}(\s)*SECURITY AND DEFENSE', output.strip()):
        return 'CAPACITY AND RESOURCES'

    elif re.search(r'(\s*B:){0,1}(\s)*HEALTH AND SAFETY', output.strip()):
        return 'MORALITY AND LAW'

    elif re.search(r'(\sC:){0,1}(\s)*QUALITY OF LIFE', output.strip()):
        return 'ECONOMICS'
    
    elif re.search(r'(\sC:){0,1}(\s)*CAPACITY AND RESOURCES', output.strip()):
        return 'ECONOMICS'

    elif re.search(r'(\s*A:){0,1}(\s)*(POLITICAL|POLICITAL)', output.strip()):
        return 'POLICY AND REGULATION'

    elif re.search(r'(\s*A:){0,1}(\s)*EXTERNAL REGULATION AND REPUTATION', output.strip()):
        return 'POLICY AND REGULATION'

    elif re.search(r'(\s*D:){0,1}(\s)*OTHER|EDUCATION|^E: SOCIAL MEDIA', output.strip()):
        return 'OTHER'

    elif output == np.nan or output == 'nan':
        return "NAN"

    else:
        print(f'Weird value: {output.strip()}')
        return "NAN"
    
map_to_task_label_processing_fn = {
    1: map_outputs_task_1,
    2: map_outputs_task_2,
    3: map_outputs_task_3,
    4: map_outputs_task_4,
    5: map_outputs_task_5,
    6: map_outputs_task_6
}

def process_output(output_str, task_num):
    # Remove the prompt
    output_str = ' '.join(output_str.split('[/INST] ')[-1:])

    # process accordin to the task_num
    return map_to_task_label_processing_fn[task_num](output_str)

