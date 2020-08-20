import pickle
import argparse

result_dir = "/Users/fischer/UCSD/XPTLab/GPT2-commonsense/language-modeling/results/"

rel_map = {'hasproperty': 'HasProperty', 'capableof': 'CapableOf', 'haspainintensity': 'HasPainIntensity', 'causesdesire': 'CausesDesire', 
'notdesires': 'NotDesires', 'symbolof': 'SymbolOf', 'usedfor': 'UsedFor', 'atlocation': 'AtLocation', 'hasfirstsubevent': 'HasFirstSubevent', 
'causes': 'Causes', 'locatednear': 'LocatedNear', 'motivatedbygoal': 'MotivatedByGoal', 'locationofaction': 'LocationofAction', 'hassubevent': 'HasSubevent', 
'relatedto': 'RelatedTo', 'madeof': 'MadeOf', 'partof': 'PartOf', 'createdby': 'CreatedBy', 'desireof': 'DesireOf', 'inheritsfrom': 'InheritsFrom', 
'notisa': 'NotIsA', 'desires': 'Desires', 'nothasproperty': 'NotHasProperty', 'hasa': 'HasA', 'notcapableof': 'NotCapableOf', 'haslastsubevent': 'HasLastSubevent', 
'definedas': 'DefinedAs', 'instanceof': 'InstanceOf', 'nothasa': 'NotHasA', 'hasprerequisite': 'HasPrerequisite', 'receivesaction': 'ReceivesAction', 'haspaincharacter': 'HasPainCharacter', 
'notmadeof': 'NotMadeOf', 'isa': 'IsA', 'haveprerequisite':'HasPrerequisite'}


parser = argparse.ArgumentParser()
parser.add_argument("--gens_name", type=str, default="../language-modeling/results/test_conceptnet_model.txt")
args = parser.parse_args()
with open(args.gens_name, "r") as f:
	raw_text = f.readlines()

output = []
with open(args.gens_name, "w") as f:
	for i, item in enumerate(raw_text):
		split_item = item.split(" <MASK> ")
		s,o = split_item[0], split_item[-1].strip()
		r = split_item[-2].strip().replace("<MASK> ", "").replace(" ", "")
		if r not in rel_map.keys():
			print(item)
			continue
		f.write(rel_map[r] + "\t" + s +  "\t" + o + "\t" + "1" + "\n" )
