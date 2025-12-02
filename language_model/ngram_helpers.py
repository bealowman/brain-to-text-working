import redis
import logging
import re
import numpy as np
from functools import lru_cache

LOGIT_TO_PHONEME = [
    'BLANK',
    ' | ',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
]

LOGIT_TO_DIPHONE = [(a, b) for a in LOGIT_TO_PHONEME for b in LOGIT_TO_PHONEME]


def connect_to_redis_server(redis_ip, redis_port):
    try:
        # logging.info("Attempting to connect to redis...")
        redis_conn = redis.Redis(host=redis_ip, port=redis_port)
        redis_conn.ping()
    except redis.exceptions.ConnectionError:    
        logging.warning("Can't connect to redis server (ConnectionError).")
        return
    else:
        logging.info("Connected to redis.")
        return redis_conn

def get_current_redis_time_ms(redis_conn):
    t = redis_conn.time()
    return int(t[0]*1000 + t[1]/1000)

# function to get string differences between two sentences
def get_string_differences(cue, decoder_output):
        decoder_output_words = decoder_output.split()
        cue_words = cue.split()

        @lru_cache(None)
        def reverse_w_backtrace(i, j):
            if i == 0:
                return j, ['I'] * j
            elif j == 0:
                return i, ['D'] * i
            elif i > 0 and j > 0 and decoder_output_words[i-1] == cue_words[j-1]:
                cost, path = reverse_w_backtrace(i-1, j-1)
                return cost, path + [i - 1]
            else:
                insertion_cost, insertion_path = reverse_w_backtrace(i, j-1)
                deletion_cost, deletion_path = reverse_w_backtrace(i-1, j)
                substitution_cost, substitution_path = reverse_w_backtrace(i-1, j-1)
                if insertion_cost <= deletion_cost and insertion_cost <= substitution_cost:
                    return insertion_cost + 1, insertion_path + ['I']
                elif deletion_cost <= insertion_cost and deletion_cost <= substitution_cost:
                    return deletion_cost + 1, deletion_path + ['D']
                else:
                    return substitution_cost + 1, substitution_path + ['R']

        cost, path = reverse_w_backtrace(len(decoder_output_words), len(cue_words))

        # remove insertions from path
        path = [p for p in path if p != 'I']

        # Get the indices in decoder_output of the words that are different from cue
        indices_to_highlight = []
        current_index = 0
        for label, word in zip(path, decoder_output_words):
            if label in ['R','D']:
                indices_to_highlight.append((current_index, current_index+len(word)))
            current_index += len(word) + 1

        return cost, path, indices_to_highlight


def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join(sentence.split())

    return sentence


# function to augment the nbest list by swapping words around, artificially increasing the number of candidates
def augment_nbest(nbest, top_candidates_to_augment=20, acoustic_scale=0.3, score_penalty_percent=0.01):

    sentences = []
    ac_scores = []
    lm_scores = []
    total_scores = []

    for i in range(len(nbest)):
        sentences.append(nbest[i][0].strip())
        ac_scores.append(nbest[i][1])
        lm_scores.append(nbest[i][2])
        total_scores.append(acoustic_scale*nbest[i][1] + nbest[i][2])

    # sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    sentences = [sentences[i] for i in sorted_indices]
    ac_scores = [ac_scores[i] for i in sorted_indices]
    lm_scores = [lm_scores[i] for i in sorted_indices]
    total_scores = [total_scores[i] for i in sorted_indices]

    # new sentences and scores
    new_sentences = []
    new_ac_scores = []
    new_lm_scores = []
    new_total_scores = []

    # swap words around
    for i1 in range(np.min([len(sentences)-1, top_candidates_to_augment])):
        words1 = sentences[i1].split()

        for i2 in range(i1+1, np.min([len(sentences), top_candidates_to_augment])):
            words2 = sentences[i2].split()

            if len(words1) != len(words2):
                continue
            
            _, path1, _ = get_string_differences(sentences[i1], sentences[i2])
            _, path2, _ = get_string_differences(sentences[i2], sentences[i1])

            replace_indices1 = [i for i, p in enumerate(path2) if p == 'R']
            replace_indices2 = [i for i, p in enumerate(path1) if p == 'R']

            for r1, r2 in zip(replace_indices1, replace_indices2):
                
                new_words1 = words1.copy()
                new_words2 = words2.copy()

                new_words1[r1] = words2[r2]
                new_words2[r2] = words1[r1]

                new_sentence1 = ' '.join(new_words1)
                new_sentence2 = ' '.join(new_words2)

                if new_sentence1 not in sentences and new_sentence1 not in new_sentences:
                    new_sentences.append(new_sentence1)
                    new_ac_scores.append(np.mean([ac_scores[i1], ac_scores[i2]]) - score_penalty_percent * np.abs(np.mean([ac_scores[i1], ac_scores[i2]])))
                    new_lm_scores.append(np.mean([lm_scores[i1], lm_scores[i2]]) - score_penalty_percent * np.abs(np.mean([lm_scores[i1], lm_scores[i2]])))
                    new_total_scores.append(acoustic_scale*new_ac_scores[-1] + new_lm_scores[-1])

                if new_sentence2 not in sentences and new_sentence2 not in new_sentences:
                    new_sentences.append(new_sentence2)
                    new_ac_scores.append(np.mean([ac_scores[i1], ac_scores[i2]]) - score_penalty_percent * np.abs(np.mean([ac_scores[i1], ac_scores[i2]])))
                    new_lm_scores.append(np.mean([lm_scores[i1], lm_scores[i2]]) - score_penalty_percent * np.abs(np.mean([lm_scores[i1], lm_scores[i2]])))
                    new_total_scores.append(acoustic_scale*new_ac_scores[-1] + new_lm_scores[-1])

    # combine new sentences and scores with old
    for i in range(len(new_sentences)):
        sentences.append(new_sentences[i])
        ac_scores.append(new_ac_scores[i])
        lm_scores.append(new_lm_scores[i])
        total_scores.append(new_total_scores[i])

    # sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    sentences = [sentences[i] for i in sorted_indices]
    ac_scores = [ac_scores[i] for i in sorted_indices]
    lm_scores = [lm_scores[i] for i in sorted_indices]
    total_scores = [total_scores[i] for i in sorted_indices]

    # return nbest
    nbest_out = []
    for i in range(len(sentences)):
        nbest_out.append([sentences[i], ac_scores[i], lm_scores[i]])

    return nbest_out
