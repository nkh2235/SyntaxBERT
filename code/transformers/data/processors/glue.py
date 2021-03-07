# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os, pickle
import stanza
import numpy as np
from .utils import DataProcessor, MaskProcessor, InputExample, InputFeatures, InputMask, InputMaskFeatures
from ...file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def generate_full_mask(attention_mask, total_len):
    attention_mask[:, 0:total_len, 0:total_len] = 1
    return attention_mask


def generate_mask_cross_two_texts(attention_mask, text_1_len, text_2_len):
    attention_mask[:, 0:text_1_len, text_1_len:text_1_len+text_2_len] = 1
    attention_mask[:, text_1_len:text_1_len+text_2_len, 0:text_1_len] = 1
    attention_mask[:, range(text_1_len+text_2_len), range(text_1_len+text_2_len)] = 1
    return attention_mask


def word_index_to_token_index(word_index_every_token):
    if word_index_every_token == None or word_index_every_token == []: return None
    token_index_every_word = []
    word_len = 0
    for i in range(len(word_index_every_token)-1, -1, -1):
        if len(word_index_every_token[i]) != 0:
            word_len = word_index_every_token[i][-1] + 1
            break
    for _ in range(word_len):
        token_index_every_word.append([])
    t_idx = 0
    for w_idx in word_index_every_token:
        for idx in w_idx:
            token_index_every_word[idx].append(t_idx)
        t_idx += 1
    return token_index_every_word

def generate_syntax_masks_with_parser(parsing_result, attention_mask, n_mask, token_shift=0, token_index_every_word=None):
    # parents & children masks
    for depth in range(1, n_mask+1):
        word_shift = 0
        for s_idx in range(len(parsing_result.sentences)):
            for word in parsing_result.sentences[s_idx].words:
                par = int(word.id) # init
                flag = 0
                for i in range(depth):
                    if(par == 0): # if the node is the root, then end the search
                        flag = 1
                        break
                    par = int(parsing_result.sentences[s_idx].words[par - 1].head)
                
                if flag == 0 and par != 0: # during the depth, find word's parent is par
                    w_idx_0 = int(word.id) - 1
                    w_idx_1 = par -1
                    total_shift = word_shift + token_shift
                    if token_index_every_word == None:
                        attention_mask[depth-1, total_shift+w_idx_0, total_shift+w_idx_1] = 1
                        attention_mask[depth-1+n_mask, total_shift+w_idx_1, total_shift+w_idx_0] = 1
                    else:
                        word_idx_0 = word_shift + w_idx_0
                        word_idx_1 = word_shift + w_idx_1
                        if word_idx_0 < len(token_index_every_word) and word_idx_1 < len(token_index_every_word):
                            token_idx_0 = token_index_every_word[word_idx_0]
                            token_idx_1 = token_index_every_word[word_idx_1]
                            if len(token_idx_0) > 0 and len(token_idx_1) > 0:
                                idx_0 = token_shift + np.array(token_idx_0)
                                idx_1 = token_shift + np.array(token_idx_1)
                                attention_mask[depth-1, idx_0[0]:idx_0[-1]+1, idx_1[0]:idx_1[-1]+1] = 1
                                attention_mask[depth-1+n_mask, idx_1[0]:idx_1[-1]+1, idx_0[0]:idx_0[-1]+1] = 1
                        
                        
            word_shift += len(parsing_result.sentences[s_idx].words)
    
    # siblings masks
    word_shift = 0
    for s_idx in range(len(parsing_result.sentences)):
        for wordi in parsing_result.sentences[s_idx].words:
            for wordj in parsing_result.sentences[s_idx].words:
                if int(wordi.id) == int(wordj.id):
                    continue
                # find the depth of word i and j
                depthi = 0
                par = int(wordi.id)
                while par != 0:
                    depthi += 1
                    par = int(parsing_result.sentences[s_idx].words[par - 1].head)
                
                depthj = 0
                par = int(wordj.id)
                while par != 0:
                    depthj += 1
                    par = int(parsing_result.sentences[s_idx].words[par - 1].head)
                
                dist = 0
                cur_i = int(wordi.id)
                cur_j = int(wordj.id)
                # To the same level of depth, cost dist
                if depthi > depthj:
                    dist += (depthi - depthj)
                    for i in range(depthi - depthj):
                        cur_i = int(parsing_result.sentences[s_idx].words[cur_i - 1].head)
                if depthi < depthj:
                    dist += (depthj - depthi)
                    for i in range(depthj - depthi):
                        cur_j = int(parsing_result.sentences[s_idx].words[cur_j - 1].head)
                # cur_i and cur_j now are at the same level, if cur_i == cur_j, then they are parent and child
                if cur_i == cur_j:
                    continue
                while cur_i != cur_j:
                    dist += 2
                    cur_i = int(parsing_result.sentences[s_idx].words[cur_i - 1].head)
                    cur_j = int(parsing_result.sentences[s_idx].words[cur_j - 1].head)

                if dist-2 < n_mask: # pengfei added
                    w_idx_0 = int(wordi.id) - 1
                    w_idx_1 = int(wordj.id) -1
                    total_shift = word_shift + token_shift
                    if token_index_every_word == None:
                        # sibling dist at least is 2
                        attention_mask[dist-2+n_mask*2, total_shift+w_idx_0, total_shift+w_idx_1] = 1
                        attention_mask[dist-2+n_mask*2, total_shift+w_idx_1, total_shift+w_idx_0] = 1
                    else:
                        word_idx_0 = word_shift + w_idx_0
                        word_idx_1 = word_shift + w_idx_1
                        if word_idx_0 < len(token_index_every_word) and word_idx_1 < len(token_index_every_word):
                            token_idx_0 = token_index_every_word[word_idx_0]
                            token_idx_1 = token_index_every_word[word_idx_1]
                            if len(token_idx_0) > 0 and len(token_idx_1) > 0:
                                idx_0 = token_shift + np.array(token_idx_0)
                                idx_1 = token_shift + np.array(token_idx_1)
                                attention_mask[dist-2+n_mask*2, idx_0[0]:idx_0[-1]+1, idx_1[0]:idx_1[-1]+1] = 1
                                attention_mask[dist-2+n_mask*2, idx_1[0]:idx_1[-1]+1, idx_0[0]:idx_0[-1]+1] = 1

        word_shift += len(parsing_result.sentences[s_idx].words)
        
    return attention_mask
    

def glue_convert_examples_to_features_with_parser(examples, tokenizer,
                                                  max_length=512,
                                                  n_mask=8,
                                                  task=None,
                                                  label_list=None,
                                                  output_mode=None,
                                                  pad_on_left=False,
                                                  pad_token=0,
                                                  pad_token_segment_id=0,
                                                  mask_padding_with_zero=True,
                                                  parser_on_text_a=True,
                                                  parser_on_text_b=True,
                                                  output_dir=None):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    # a parser
    if parser_on_text_a or parser_on_text_b:
        parser = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    parsing_result_qs = []
    parsing_result_ads = []
    attention_masks = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        # parsing
        first_text_words = None; second_text_words = None
        if parser_on_text_a:
            parsing_result_q = parser(example.text_a) # parser for query
            parsing_result_qs.append(parsing_result_q)
            first_text_words = [word.text for s_idx in range(len(parsing_result_q.sentences)) for word in parsing_result_q.sentences[s_idx].words]
        
        if parser_on_text_b:
            parsing_result_ad = parser(example.text_b) # parser for ads
            parsing_result_ads.append(parsing_result_ad)
            second_text_words = [word.text for s_idx in range(len(parsing_result_ad.sentences)) for word in parsing_result_ad.sentences[s_idx].words]

        inputs = tokenizer.encode_xs( # in tokenization_utils.py
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            first_text=first_text_words,
            second_text=second_text_words,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        first_word_index_every_token, second_word_index_every_token = inputs["first_word_idx"], inputs["second_word_idx"]
        
        # convert word index for every token to token index for every word
        first_token_index_every_word = word_index_to_token_index(first_word_index_every_token)
        second_token_index_every_word = word_index_to_token_index(second_word_index_every_token)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        #attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            #attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            #attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        '''
        # Generate [nmask, max_length, max_length] input_mask tensor
        
        if parser_on_text_a:
            attention_mask_q = np.zeros((n_mask*3, max_length, max_length)) if mask_padding_with_zero else np.ones((n_mask*3, max_length, max_length))
            attention_mask_q = generate_syntax_masks_with_parser(parsing_result_q, attention_mask_q, n_mask, 
                                                             token_shift=0, token_index_every_word=first_token_index_every_word)

        if parser_on_text_b:
            attention_mask_ad = np.zeros((n_mask*3, max_length, max_length)) if mask_padding_with_zero else np.ones((n_mask*3, max_length, max_length))
            attention_mask_ad = generate_syntax_masks_with_parser(parsing_result_ad, attention_mask_ad, n_mask, 
                                                              token_shift=len(first_word_index_every_token),
                                                              token_index_every_word=second_token_index_every_word)

        # generate cross-text attention mask
        if parser_on_text_a and parser_on_text_b:
            attention_mask_x = np.zeros((1, max_length, max_length)) if mask_padding_with_zero else np.ones((n_mask, max_length, max_length))
            attention_mask_x = generate_mask_cross_two_texts(attention_mask_x, len(first_word_index_every_token), len(second_word_index_every_token))
        '''
        # generate full attention mask
        attention_mask_f = np.zeros((n_mask, max_length, max_length)) if mask_padding_with_zero else np.ones((n_mask, max_length, max_length))
        attention_mask_f = generate_full_mask(attention_mask_f, len(first_word_index_every_token)+len(second_word_index_every_token))
        
        
        
        
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        #assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # element-wisely summation
        '''
        mask_not_ready = True
        if parser_on_text_a:
            attention_mask = attention_mask_q
            mask_not_ready = False
        if parser_on_text_b:
            attention_mask = attention_mask_ad if mask_not_ready else attention_mask + attention_mask_ad
            mask_not_ready = False
        if parser_on_text_a and parser_on_text_b:
            attention_mask = attention_mask_x if mask_not_ready else np.concatenate([attention_mask, attention_mask_x], axis=0)
            mask_not_ready = False
        attention_mask = attention_mask_f if mask_not_ready else np.concatenate([attention_mask, attention_mask_f], axis=0)
        mask_not_ready = False
        # record attention_mask
        if output_dir != None:
            attention_masks.append(attention_mask)
        '''
        attention_mask = attention_mask_f
        #import pdb; pdb.set_trace()
        #np.save("att_mask.npy", attention_mask)
        '''
        np.save("att_mask_x.npy", attention_mask_x)
        np.save("att_mask_q.npy", attention_mask_q)
        np.save("att_mask_ad.npy", attention_mask_ad)
        '''
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        attention_masks = np.array(attention_masks)
        mask_pkl = os.path.join(output_dir, "att_masks.pkl")
        with open(mask_pkl, "wb") as pkl:
            pickle.dump(attention_masks, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        if parser_on_text_a:
            qs_pkl = os.path.join(output_dir, "parsing_qs.pkl")
            with open(qs_pkl, "wb") as pkl:
                pickle.dump(parsing_result_qs, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        if parser_on_text_b:
            ads_pkl = os.path.join(output_dir, "parsing_ads.pkl")
            with open(ads_pkl, "wb") as pkl:
                pickle.dump(parsing_result_ads, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features


def generate_prior_knowledge_full_masks(n_mask, max_len, total_len):
    masks = np.zeros((n_mask, max_len, max_len))
    masks[:, 0:total_len, 0:total_len] = 1

    return masks


def generate_prior_knowledge_cross_masks_with_mapping(map_idx, n_mask, max_len, token_a_len, token_b_len,
                                                      token_a_index_every_word, token_b_index_every_word):
    masks = np.zeros((n_mask, max_len, max_len))
    if token_a_index_every_word == None or token_b_index_every_word == None:
        return masks
    token_a_index_every_word = np.array(token_a_index_every_word)
    token_b_index_every_word = np.array(token_b_index_every_word)
    for mask_idx in range(n_mask):
        idx_ = np.array(map_idx[mask_idx]) # n_mask * ?  *2
        w_idx_0 = idx_[:, 0]; w_idx_1 = idx_[:, 1] # ? * 2
        t_idx_0 = token_a_index_every_word[w_idx_0] # ?? * 2
        t_idx_1 = token_a_len + token_b_index_every_word[w_idx_1]
        masks[mask_idx, t_idx_0, t_idx_1] = 1
        masks[mask_idx, t_idx_1, t_idx_0] = 1
    masks[:, range(token_a_len+token_b_len), range(token_a_len+token_b_len)] = 1

    return masks


def generate_prior_knowledge_masks_with_mapping(map_idx, n_mask, max_len, token_shift=0, token_index_every_word=None):
    masks = np.zeros((n_mask, max_len, max_len))
    if token_index_every_word == None:
        return masks
    token_index_every_word = np.array(token_index_every_word)
    for mask_idx in range(n_mask):
        idx_ = np.array(map_idx[mask_idx])
        w_idx_0 = idx_[:, 0]; w_idx_1 = idx_[:, 1]
        t_idx_0 = token_shift + token_index_every_word[w_idx_0]
        t_idx_1 = token_shift + token_index_every_word[w_idx_1]
        masks[mask_idx, t_idx_0, t_idx_1] = 1
        masks[mask_idx, t_idx_1, t_idx_0] = 1

    return masks


def glue_convert_examples_to_mask_idx(examples, 
                                      max_length=512, 
                                      n_masks=8,
                                      pad_on_left=False, 
                                      pad_token=0, 
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        max_length: Maximum example length
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    processor = MaskProcessor()
    mask_idx = {}
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        ##  update mask index from examples
        if example.id not in mask_idx.keys(): # init a mask index
            mask_idx[example.id] = []
            for _ in range(n_masks):
                mask_idx[example.id].append([])
        mask_idx[example.id][example.threshold].append([example.text_a_idx, example.text_b_idx])
    
    return mask_idx


def glue_convert_examples_to_features_with_prior_knowledge(
            examples, tokenizer, max_length=512, n_mask=8, task=None, label_list=None, 
            output_mode=None, pad_on_left=False, pad_token=0,  pad_token_segment_id=0,  
            mask_padding_with_zero=True, parser_on_text_a=True, parser_on_text_b=True, 
            mapping_a=None, mapping_b=None, mapping_x=None, output_dir=None):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    # if mappings for a and b are not given, then use Stanza as the parser to provide attention masks
    if mapping_a == None and mapping_b == None and mapping_x == None:
        return glue_convert_examples_to_features_with_parser(examples, tokenizer, max_length=max_length, n_mask=n_mask, task=task, label_list=label_list,
                        output_mode=output_mode, pad_on_left=pad_on_left, pad_token=pad_token, pad_token_segment_id=pad_token_segment_id,
                        mask_padding_with_zero=mask_padding_with_zero, parser_on_text_a=parser_on_text_a, parser_on_text_b=parser_on_text_b,
                        output_dir=output_dir)
    
    # else use mapping_a and mapping_b to calculate attention masks
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    attention_masks = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_xs( # in tokenization_utils.py
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        first_word_index_every_token, second_word_index_every_token = inputs["first_word_idx"], inputs["second_word_idx"]

        # convert word index for every token to token index for every word
        first_token_index_every_word = word_index_to_token_index(first_word_index_every_token)
        second_token_index_every_word = word_index_to_token_index(second_word_index_every_token)

        if mapping_a != None:
            attention_mask_q = generate_prior_knowledge_masks_with_mapping(mapping_a[example.id], n_mask, max_length, 
                                            token_shift=0, token_index_every_word=first_token_index_every_word)
        if mapping_b != None:
            attention_mask_ad = generate_prior_knowledge_masks_with_mapping(mapping_b[example.id], n_mask, max_length, 
                                            token_shift=len(first_word_index_every_token), token_index_every_word=second_token_index_every_word)
        if mapping_x != None:
            # generate cross-text attention mask
            import pdb; pdb.set_trace()
            attention_mask_x = generate_prior_knowledge_cross_masks_with_mapping(mapping_x[example.id], n_mask, max_length, 
                                            token_a_len=len(first_word_index_every_token), token_b_len=len(second_word_index_every_token),
                                            token_a_index_every_word=first_token_index_every_word, token_b_index_every_word=second_token_index_every_word)

        # generate full mask
        attention_mask_f = generate_prior_knowledge_full_masks(1, max_length, total_len=len(first_word_index_every_token)+len(second_word_index_every_token))

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            #attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            #attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        #assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # element-wisely summation
        mask_not_ready = True
        if parser_on_text_a:
            attention_mask = attention_mask_q
            mask_not_ready = False
        if parser_on_text_b:
            attention_mask = attention_mask_ad if mask_not_ready else attention_mask + attention_mask_ad
            mask_not_ready = False
        if parser_on_text_a and parser_on_text_b:
            attention_mask = attention_mask_x if mask_not_ready else np.concatenate([attention_mask, attention_mask_x], axis=0)
            mask_not_ready = False
        attention_mask = attention_mask_f if mask_not_ready else np.concatenate([attention_mask, attention_mask_f], axis=0)
        mask_not_ready = False

        # record attention_mask
        if output_dir != None:
            attention_masks.append(attention_mask)
        
        '''
        import pdb; pdb.set_trace()
        np.save("att_mask.npy", attention_mask)
        np.save("att_mask_x.npy", attention_mask_x)
        np.save("att_mask_q.npy", attention_mask_q)
        np.save("att_mask_ad.npy", attention_mask_ad)
        '''
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        attention_masks = np.array(attention_masks)
        mask_pkl = os.path.join(output_dir, "att_masks.pkl")
        with open(mask_pkl, "wb") as pkl:
            pickle.dump(attention_masks, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    
    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features


def glue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    '''
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    '''

    def get_train_examples(self, data_path):
        """See base class."""
        logger.info("LOOKING AT TRAINING DIR {}".format(data_path))
        if os.path.isfile(data_path):
            return self._create_examples(self._read_tsv(data_path), "train")
        else:
            return self._create_examples_from_dir(data_path, "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        logger.info("LOOKING AT VALIDATION DIR {}".format(data_path))
        if os.path.isfile(data_path):
            return self._create_examples(self._read_tsv(data_path), "val")
        else:
            return self._create_examples_from_dir(data_path, "val")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples_from_dir(self, dir, set_type):
        examples = []
        for f in os.listdir(dir):
            f_ = os.path.join(dir, f)
            if os.path.isfile(f_):
                examples.extend(self._create_examples(self._read_tsv(f_), set_type))
        return examples

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            ## Skip the header
            #if i == 0: continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3] # query
            text_b = line[4] # Ads
            label = line[0]
            if text_a != "" and text_b != "":
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor_plus(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy(),
                            tensor_dict['id'].numpy()))

    def get_train_examples(self, data_path):
        """See base class."""
        logger.info("LOOKING AT TRAINING DIR {}".format(data_path))
        if os.path.isfile(data_path):
            return self._create_examples(self._read_tsv(data_path), "train")
        else:
            return self._create_examples_from_dir(data_path, "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        logger.info("LOOKING AT VALIDATION DIR {}".format(data_path))
        if os.path.isfile(data_path):
            return self._create_examples(self._read_tsv(data_path), "val")
        else:
            return self._create_examples_from_dir(data_path, "val")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples_from_dir(self, dir, set_type):
        examples = []
        for f in os.listdir(dir):
            f_ = os.path.join(dir, f)
            if os.path.isfile(f_):
                examples.extend(self._create_examples(self._read_tsv(f_), set_type))
        return examples

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            ## Skip the header
            #if i == 0: continue
            guid = "%s-%s" % (set_type, i)
            id = line[0]
            label = line[1]
            text_a = line[2] # query
            text_b = line[3] # Ads
            if text_a != "" and text_b != "":
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, id=id))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['question1'].numpy().decode('utf-8'),
                            tensor_dict['question2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['question'].numpy().decode('utf-8'),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor_plus, #MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}
