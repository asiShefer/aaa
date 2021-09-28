from dataclasses import dataclass
from typing import List
from collections import defaultdict
import dataclasses
import pandas as pd
import numpy as np

@dataclass(order=True)
class utter:
	'''
	This dataclass represents an utterance. 
	It contains the following fields: 
			- time: Float. The utterance timestep
			- conv_idx: Int. The utterace's conversation file id (the conversation should be in conv_idx.txt)
			- utter_id: Int. The number of the utterance in the conversation (e.g, 1'st/2'nd utterance in the conversation)
			- speaker: String. Who is the speaker (e.g, "agent"/"costumer"/"unknown speaker")
			- text: String. The text spoken in this utterance
			- labels: List of dictionaries. Each dictionary is {'label':String, 'text':String (the labeled span)}
			- prev_utter_text/labels/speaker: Lists containing the text/labels/speakers of the 
											  prev_utters_num (see DataLoader) previous utterances. 
											  Note that the list is ordered by closeness to this utterance,
											  that is, the first item is the utter_id-1 utterance, the next is utter_id-2
											  and so on
			- post_utter_text/labels/speaker: Lists containing the text/labels/speakers of the 
											  post_utters_num (see DataLoader) utterances that come after this one. 
	'''
	time: float=None
	conv_idx: int=None
	utter_id: int=None
	speaker: str=None
	text: str=None
	labels: List=dataclasses.field(default_factory=list)
	prev_utter_text: List=dataclasses.field(default_factory=list)
	post_utter_text: List=dataclasses.field(default_factory=list)
	prev_utter_labels: List=dataclasses.field(default_factory=list)
	post_utter_labels: List=dataclasses.field(default_factory=list)
	prev_utter_speaker: List=dataclasses.field(default_factory=list)
	post_utter_speaker: List=dataclasses.field(default_factory=list)
	


class DataLoader(object):

	def __init__(self, annotation_file, labels_column_name, 
				conv_idx_column_name, 
				text_column_name, 
				conv_dir,
				prev_utters_num=0, post_utters_num=0,
				sheet_names=None, Asi=True):
		'''
		@args:
			- annotation_file: String. The path to the Excel file holding the labeled data
			- labels_column_name/conv_idx_column_name/text_column_name: What are the columns name in annotation_file
			- conv_dir: String. Path to the conversations directory.
			- prev_utters_num/post_utters_num: Int. Number of previous/post utterances that each utter object should hold.
			- sheet_names: String or List of strings. The sheet name/s in annotation_file we should use. 
		'''
		self.annotation_file=annotation_file
		self.labels_column_name=labels_column_name 
		self.conv_idx_column_name=conv_idx_column_name
		self.text_column_name=text_column_name 
		self.conv_dir=conv_dir
		self.sheet_names=sheet_names
		self.prev_utters_num=prev_utters_num
		self.post_utters_num=post_utters_num

		self.convDict, self.bad_conv_idx, self.labels, self.conv2labeledText = self.dataloader()
		self.label2utter = self.get_label2utter()


	def get_labels(self):
		return self.labels

	def get_convDict(self):
		return self.convDict

	def get_labeled_convID(self):
		'''
		Returns a dictionary s.t for each conversation we have a list of all (and only) labeled utterances
		'''
		return {conv_id:self.get_conv(conv_id) for conv_id, conv in self.convDict.items() if len(self.get_conv(conv_id))>0}

	def get_label2utter(self):
		'''
		label2utter: Dictionary where label2utter[label] = List containing only utterances with spans labeled with "label" 
		'''
		label2utter = {label:[] for label in self.labels}
		for utter_list in self.convDict.values():
		    for utter in utter_list:
		        if len(utter.labels)!=0:
		            labels_set=set()
		            for item in utter.labels:
		                label = item["label"]
		                if label not in labels_set:
		                    label2utter[label].append(utter)
		                    labels_set.add(label)
		return label2utter

	def get_conv(self, conv_id):
		'''
		Return utterances, from conversation conv_id, that contain labeled spans
		'''
		return [item for item in self.convDict[conv_id] if len(item.labels)>0]


	def load_excel_file(self):
		df_list = []
		if type(self.sheet_names)==str:
			df = pd.read_excel(self.annotation_file, sheet_name=self.sheet_name)
			df_list.append(df)
		
		elif type(self.sheet_names)==list:
			for sheet_name in self.sheet_names:
				df = pd.read_excel(self.annotation_file, sheet_name=sheet_name)
				df_list.append(df)
		elif self.sheet_names is None:
			df = pd.read_excel(self.annotation_file)
			df_list.append(df)
			
		labels = list(set(item.lower() for df in df_list for item in df[self.labels_column_name] if item is not np.nan))
		return df_list, labels
		
		
		
	def get_file_path(self, conv_idx, max_num_of_digits=3):
		'''
		get the conversation number as an int and return the file name (string)
		e.g, 1->001, 10->010, 100->100
		
		'''
		conv_int_as_string = str(int(conv_idx))
		num_of_digits = len(conv_int_as_string)
		return ''.join(['0' for _ in range(max_num_of_digits-num_of_digits)])+conv_int_as_string


	def get_conv2labeledText(self, df_list):
		'''
		preprocessing step for get_conv_labeled_data

		@args:
			df_list: List of dataframes (one df for each Excel tab that represent one tagger)

		returns: 
				Dict. Each conversation points to a list of the labeled spans.
				 conv2labeledText[conversation number]: [{label, text, count}, ...]
				 where count will be used to count the number of times the text was found in the conversation
				 (see get_conv_labeled_data) to help us find bugs (long texts tet get duplicated in the same 
				 conversation for no reason)
			
		'''
		conv2labeledText = defaultdict(list)
		conv_idx_set = set() # for testing
		for df_count, df in enumerate(df_list):
			conv_idx_set_tmp = set()
			for index, row in df[df[self.labels_column_name].notna()].iterrows():
				if 	type(row[self.text_column_name])!=str:
					continue
				conv_idx = int(row[self.conv_idx_column_name])

				# conv_idx_set contains data from other Excel's tabs (taggers). We want to insure that the same 
				# conversation doesn't appear twice in the data
				assert conv_idx not in conv_idx_set, f"df_count: {df_count}, conv_idx: {conv_idx}, conv_idx_set: {conv_idx_set}, conv_idx_set_tmp:{conv_idx_set_tmp}"
				
				conv_idx_set_tmp.add(conv_idx)
				if row[self.labels_column_name].lower() not in ["action response", "no data", "noted"]:
					conv2labeledText[conv_idx].append({'label':row[self.labels_column_name].lower(),
													   'text':row[self.text_column_name].strip(),
													   'count':0})
			conv_idx_set = conv_idx_set.union(conv_idx_set_tmp)
		
		return conv2labeledText

	def get_conv_labeled_data(self, conv_idx, conv, conv2labeledText):
		'''
		Takes conv list which contains the text in conersasion conv_idx in the following froms:
			[Speaker Time (of utterance 1, i.e, "Agent 0:12"), text (of utterance 1), ..., 
			Speaker time (of utterance i), text (of utterance i), ...] 

		and turns it into a list of utter objects.

		params:
			conv_idx: int
				  conversation id
			
			conv: List of strings.
				  The text segments (utterances) in conv_idx conversation.
				  Should look like 
				  [Speaker Time (of utterance 1, i.e, "Agent 0:12"), text (of utterance 1), ..., Speaker time (of utterance i), text (of utterance i), ...] 
			
			conv2labeledText:dict.
							 see get_conv2labeledText
				  
		returns:
			utters_list: a list of utter dataclasses for conv
		'''
		
		utters_list = []
		itr_utter = utter()
		utter_id=0
		for text_seg_num, text_seg in enumerate(conv):
			utters_num = len(utters_list)
			if itr_utter.speaker is None:
				# itr_utter.speaker is None if this is the first time this utterance is seen
				# This means that text_seg should be - Speaker Time
				for i in range(self.prev_utters_num):
					if utters_num>i:
						itr_utter.prev_utter_text.append(utters_list[utters_num-1-i].text)
						itr_utter.prev_utter_labels.append(utters_list[utters_num-1-i].labels)
						itr_utter.prev_utter_speaker.append(utters_list[utters_num-1-i].speaker)
							
				splits = text_seg.split()
				if splits[0].lower()=="prospect":
					splits[0] = 'customer'
				if splits[0].lower()=="'agnet'":
					splits[0] = 'agent'

				if itr_utter.conv_idx is not None or itr_utter.time is not None or splits[0].lower() not in ['customer', 'agent', 'agnet', 'prospect', 'unknown']:
					return -1

				utter_id+=1
				itr_utter.utter_id=utter_id
				itr_utter.speaker = ' '.join(splits[:-1]).strip()
				itr_utter.time = float(splits[-1].replace(':', '.'))
				itr_utter.conv_idx = conv_idx

			else:
				# If we're here, we've already dealt with the utterance's speaker and time and now 
				# we're dealing with its text
				itr_utter.text=text_seg
				for i in range(self.post_utters_num):
					if utters_num>i:
						utters_list[utters_num-1-i].post_utter_text.append(text_seg)
						utters_list[utters_num-1-i].post_utter_speaker.append(itr_utter.speaker)
				for item in conv2labeledText[conv_idx]:
					if item['text'].lower() in text_seg.lower():
						if item['count']!=0:# and len(item['text'].split())>7:
							# We conversations that have cutterances with count>0, that is, 
							# the text in this conversation is duplicated, and the duplicated text is 
							# has more than 7 words 
							return -2
						itr_utter.labels.append({'label':item['label'], 'text':item['text']})
						item['count']+=1

				for i in range(self.post_utters_num):
					if utters_num>i:
						utters_list[utters_num-1-i].post_utter_labels.append(itr_utter.labels)

				utters_list.append(itr_utter)
				itr_utter = utter()

		if conv_idx!=1:
			for item in conv2labeledText[conv_idx]:
				if item['count']==0:
					# If the text were not processed at all
					return -3

		return utters_list

	def dataloader(self):
		'''
		Read the data and process it.

		@returns:
			convDict: Dictionary: convDict[conv_idx] = list of the conversation utter objects
			bad_conv_idx: List of ints. conv_idx of problematic conversations (get_conv_labeled_data returns
						  an int < 0)
			labels: lList of all labels
			conv2labeledText: see get_conv2labeledText
		'''
		
		df_list, labels = self.load_excel_file()
		conv2labeledText = self.get_conv2labeledText(df_list)
		
		convDict = {}
		bad_conv_idx = []
		labeled_conv_idx = conv2labeledText.keys()
		for fpath in self.conv_dir.iterdir():
			conv_idx = int(fpath.name.split('.')[0].split()[0])
			if conv_idx not in labeled_conv_idx:
				# If the conversation has no labeled spans
				continue
		   
			with open(fpath, 'r') as f:
				conv_text = [item.strip() for item in f.read().split('\n') if item.strip()!='']
			utters_list=self.get_conv_labeled_data(conv_idx, conv_text, conv2labeledText)
			if type(utters_list)==int:
				# For testing
				bad_conv_idx.append([conv_idx, utters_list])
			else:
				convDict[conv_idx]=utters_list

		conv_num = len(convDict)
		utters_num = sum([len(conv) for conv in convDict.values()])
		labeled_utters_num = sum([sum([1 for uttr in conv if len(uttr.labels)>0]) for conv in convDict.values()])
		labeled_uttr_ratio_per_conv_mean = np.mean(
			[sum([1 for uttr in conv if len(uttr.labels)>0])/len(conv) for conv in convDict.values()])

		print("Utterances nummber:", utters_num)
		print(f"Labeled utterances nummber: {labeled_utters_num}")
		print("Conversations number:", conv_num)
		print(f"Average number of utterances in a conversations: {utters_num/conv_num:.2f}")
		print()

		print(f"{100*labeled_utters_num/utters_num:0.2f}% of the utterances are labeled")
		print(f"{100*labeled_uttr_ratio_per_conv_mean:0.2f}% of the utterances in an average conversation are labeled")

		return convDict, bad_conv_idx, labels, conv2labeledText
