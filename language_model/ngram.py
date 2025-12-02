import redis
import argparse
import numpy as np
import time
import math
import os
from datetime import datetime
import logging
import torch
import re
import heapq
import pandas as pd
import kenlm
from ngram_helpers import * 

# logging.basicConfig(
#     format="%(asctime)s %(levelname)s: %(message)s",
#     level=logging.DEBUG,
# )

word_to_phoneme_lexicon = {}
phoneme_to_word_lexicon = {}

def load_lexicon():
    with open("lexicon/cmudict.dict", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            raw_word = parts[0]

            word = re.sub(r"\(\d+\)$", "", raw_word)

            phonemes = parts[1:]
            
            clean_phonemes = []
            for phoneme in phonemes:
                clean_phone = re.sub(r"\d+", "", phoneme)
                clean_phonemes.append(clean_phone)
            
            clean_word = re.sub(r"[^A-Za-z]", "", word)
            if clean_word not in word_to_phoneme_lexicon:
                word_to_phoneme_lexicon[clean_word] = []
            word_to_phoneme_lexicon[clean_word].append(clean_phonemes)

            phoneme_tuple = tuple(clean_phonemes)
            if phoneme_tuple not in phoneme_to_word_lexicon:
                phoneme_to_word_lexicon[phoneme_tuple] = set()
            phoneme_to_word_lexicon[phoneme_tuple].add(clean_word)
        
    logging.info(f"Loaded lexicon with {len(phoneme_to_word_lexicon)} phoneme sequences")
    logging.info(f"Sample entries: {list(phoneme_to_word_lexicon.items())[:5]}")



class PrefixScore:
    def __init__(self, prefix: list[str], blank=float('-inf'), non_blank=float('-inf')):
        self.pfx = prefix
        self.blank = blank
        self.non_blank = non_blank

    def score(self):
        return self._log_add(self.blank, self.non_blank)

    def _log_add(self, a, b):
        min_lim = float('-inf')
        if a <= min_lim:
            return b
        elif b <= min_lim:
            return a
        else:
            return np.logaddexp(a, b)

    def copy(self):
        return PrefixScore(prefix=self.pfx, blank=self.blank, non_blank=self.non_blank)

def log_add(a, b):
    min_lim = float('-inf')
    if a <= min_lim:
        return b
    elif b <= min_lim:
        return a
    else:
        return np.logaddexp(a, b)



class NGram:
    def __init__(self, max_active=7000, min_active=200, beam=17., lattice_beam=8., acoustic_scale=1.5, ctc_blank_skip_threshold=1.0, length_penalty=0.0, nbest=5, k=5):
        self.model = kenlm.LanguageModel("pretrained_language_models/wiki_2gram.binary~")
        
        self.max_active = max_active
        self.min_active = min_active
        self.beam = beam
        self.lattice_beam = lattice_beam
        self.acoustic_scale = acoustic_scale
        self.ctc_blank_skip_threshold = ctc_blank_skip_threshold
        self.length_penalty = length_penalty

        self.nbest = nbest
        self.top_n_sentences = []
        
        # includes the best n sentences derived from doing CTC beam search
        self.k = k
        self.top_k_phon_seqs = {}
        
        # safety check for first iteration (in case top_k_phon_seqs is not updated correctly)
        self.iteration = 1


    def reset(self):
        self.top_k_phon_seqs = {}
        self.top_n_sentences = []
        self.iteration = 1

    def get_partial_sentence(self):
        if not self.top_k_phon_seqs:
            return ''
        best_prefix, best_score = max(self.top_k_phon_seqs.items(), key=lambda item: item[1].score())
        return " ".join(best_prefix)
    
    def beam_aligned_greedy(self, logits, blank_penalty, mask_first_frame=True):
        log_probs = torch.log_softmax(torch.from_numpy(logits.copy()), dim=-1).numpy()
        if blank_penalty > 0:
            log_probs[:, 0] -= math.log(blank_penalty)
        if mask_first_frame:
            log_probs[0, 1] = -np.inf
        best_ids = log_probs.argmax(axis=-1)
        best_phonemes = [LOGIT_TO_PHONEME[i] for i in best_ids]
        prev = None
        collapsed = []
        for p in best_phonemes:
            if p == "BLANK" or p == ' | ':
                continue
            if p != prev:
                collapsed.append(p)
                prev = p
        print(" ".join(collapsed))
        return collapsed

    def process_input_frame(self, logits):
        '''
        CTC beam search step
        args:
            logits:np.array(T, 41) -> T 41-lengthed logit arrays, where each logit array represents phoneme probabilities
        '''
        if logits is None or len(logits) == 0:
            print("no logits")
            return
        if len(logits) == 1 and len(logits[0]) == 0:
            print("no logits, empty nested array")
            return 
        for logit in logits:
            # len(logit) == 41
            if len(logit) != 41:
                print('invalid logit length')
                raise ValueError("invalid logit length")
           
            # if no prefix scores (first iteration), prefix scores == logits
            if not self.top_k_phon_seqs and self.iteration == 1:
                # set BLANK just to remove 
                k_indices = self.get_top_k(logit)
                for i in k_indices:
                    phoneme = LOGIT_TO_PHONEME[i]
                    phoneme_prob = logit[i]
                    if phoneme == "BLANK" or phoneme == " | ":
                        phoneme_list = []
                    else:
                        phoneme_list = [phoneme]
                    self.top_k_phon_seqs[(phoneme,)] = PrefixScore(prefix=phoneme_list, non_blank=phoneme_prob)
            else:
                # CTC beam search
                new_top_k_phon_seqs = {}
                k_indices = np.argsort(logit)[::-1]
                for p, score in self.top_k_phon_seqs.items():
                    prefix = score.pfx
                    count = 0
                    for i in k_indices:
                        if count >= int(self.beam):
                            break
                        phoneme = LOGIT_TO_PHONEME[i]
                        phoneme_prob = logit[i]
                        next_prefix = tuple(prefix + [phoneme])
                        next_score1 = None
                        if phoneme == "BLANK":
                            key = tuple(prefix)
                            next_score = new_top_k_phon_seqs.get(key, PrefixScore(prefix=prefix.copy()))
                            next_score.blank = log_add(next_score.blank, score.score() + phoneme_prob)
                            new_top_k_phon_seqs[key] = next_score
                        elif len(prefix) > 0 and phoneme == prefix[-1]:
                            # merging
                            key = tuple(prefix)
                            next_score = new_top_k_phon_seqs.get(key, PrefixScore(prefix=prefix.copy()))
                            next_score.non_blank = log_add(next_score.non_blank, score.non_blank + phoneme_prob)
                            new_top_k_phon_seqs[key] = next_score
                            if score.blank != float('-inf'):
                                next_prefix1 = tuple(prefix + [phoneme])
                                next_score1 = new_top_k_phon_seqs.get(next_prefix1, PrefixScore(prefix=list(next_prefix1)))
                                next_score1.non_blank = log_add(next_score1.non_blank, score.blank + phoneme_prob)
                                new_top_k_phon_seqs[next_prefix1] = next_score1
                        else:
                            # no merge or blank
                            next_prefix = tuple(prefix + [phoneme])
                            next_score = new_top_k_phon_seqs.get(next_prefix, PrefixScore(prefix=list(next_prefix)))
                            next_score.non_blank = log_add(next_score.non_blank, score.score() + phoneme_prob)
                            new_top_k_phon_seqs[next_prefix] = next_score
                        
                        count += 1

                next_heap = []
                for p, score in new_top_k_phon_seqs.items():
                    heapq.heappush(next_heap, (score.score(), p, score))
                while len(next_heap) > self.k:
                    heapq.heappop(next_heap)
                self.top_k_phon_seqs = {p: score for _, p, score in next_heap}


            self.iteration += 1


            # calculate prefix scores for each prefix in self.top_k_phon_seqs
            # handle merges


            # store top k results
    def get_top_k(self, a):
        '''
        args:
            a:list|np.ndarray
        '''
        if type(a) == np.ndarray:
            res = np.argpartition(a, -self.k)[-self.k:]
        else:
            res = heapq.nlargest(self.k, enumerate(a), key=lambda x: x[1])
            res = [i for i, _ in res]
        return res

    def get_ngram_probability(self, seq: str):
        perplexity = self.model.perplexity(seq)
        return perplexity

    def run_lexicon(self, seq: str, max_length: int = 10):
        '''takes in one of N phoneme sequences to decode into a lexicon lattice'''
        if not seq or not seq.strip():
            return []
        seq = seq.split()
        if not seq:
            return []
        
        graph = {i: [] for i in range(len(seq))}

        # Build graph of valid word spans
        for i in range(len(seq)):
            for j in range(i, len(seq)+1):
                span = tuple(seq[i:j])
                if span in phoneme_to_word_lexicon:
                    graph[i].append((j, list(phoneme_to_word_lexicon[span])))

        seq_len = len(seq)
        results = []  # Complete paths (reach end)
        partial_results = []  # Partial paths (don't reach end)

        def extract_lattice(i, current_path, is_partial=False):
            nonlocal graph, results, partial_results
            if len(results) >= max_length and len(partial_results) >= max_length:
                return
            
            # Complete path: reached the end
            if i >= seq_len:
                path_str = " ".join(current_path).strip()
                if path_str:  # Only add non-empty paths
                    results.append(path_str)
                return
            
            # No valid words starting at position i - this is a partial path
            if not graph[i]:
                # Only save partial paths if we've matched at least one word
                if current_path:  # At least one word was matched
                    path_str = " ".join(current_path).strip()
                    coverage = i / seq_len  # How much of sequence was covered (0-1)
                    partial_results.append({
                        'path': path_str,
                        'coverage': coverage,
                        'position': i,  # Last position reached
                        'words_matched': len(current_path)
                    })
                return
            
            # Continue exploring paths
            for next_phone, words in graph[i]:
                for w in words:
                    extract_lattice(next_phone, current_path + [w], is_partial)
                    if len(results) >= max_length and len(partial_results) >= max_length:
                        return
        
        # Extract complete paths first
        extract_lattice(0, [])
        
        # If we found complete paths, return them
        if results:
            return results[:max_length]
        
        # Otherwise, return best partial matches sorted by coverage
        if partial_results:
            # Sort by coverage (how much of sequence was covered), then by number of words
            partial_results.sort(key=lambda x: (x['coverage'], x['words_matched']), reverse=True)
            # Return top partial matches
            return [p['path'] for p in partial_results[:max_length]]
        
        # No matches at all
        logging.debug(f"No complete or partial lattice found for {' '.join(seq[:10])}")
        return []


    def run_ngram(self):
        if not self.top_k_phon_seqs:
            return []

        try:
            best_acoustic = max(ps.score() for ps in self.top_k_phon_seqs.values())
        except Exception as e:
            logging.error(f'Error in getting best acoustic score: {e}', exc_info=True)
            return []
        ACOUSTIC_MARGIN = 0
        candidates = []
        total_sequences = len(self.top_k_phon_seqs)
        processed = 0
        lattice_found = 0
        for seq_key, prefix_score in self.top_k_phon_seqs.items():
            try:
                if prefix_score.score() < best_acoustic - ACOUSTIC_MARGIN:
                    continue
                tokens = [p for p in seq_key if p != ' | ' and p != 'BLANK']
                if not tokens:
                    logging.debug(f"Skipping empty token sequence: {seq_key}")
                    continue
                lexicon_input = " ".join(tokens)
                lattice = self.run_lexicon(lexicon_input)
                acoustic_score = prefix_score.score()
                processed += 1
                if not lattice:
                    logging.debug(f"No lattice found for sequence {lexicon_input[:50]}")
                    continue
                else:
                    lattice_found += 1
                    logging.debug(f"Found {len(lattice)} word sequences for sequence {lexicon_input}")
                for option in lattice:
                    try:
                        perplexity = self.get_ngram_probability(option)
                        # Avoid log(0); KenLM perplexity should be > 0, but guard anyway.
                        if perplexity <= 0:
                            continue
                        lm_score = -math.log(perplexity)
                        combined_score = lm_score + (self.acoustic_scale * acoustic_score)
                        candidates.append({
                            'sentence': option.strip(),
                            'combined_score': combined_score,
                            'lm_score': lm_score,
                            'acoustic_score': acoustic_score,
                        })
                    except Exception as e:
                        logging.warning(f"Error scoring sentence {option}: {e}", exc_info=True)
                        continue
            except Exception as e:
                logging.warning(f"Error processing phoneme sequence {seq_key}: {e}", exc_info=True)
                continue
        logging.debug(f"Processed {processed} sequences, found {lattice_found} lattices")
        logging.debug(f"Total sequences: {total_sequences}")
        logging.debug(f"Candidates: {candidates}")
        try:
            candidates.sort(key=lambda c: c['combined_score'], reverse=True)
            return candidates[:self.nbest]
        except Exception as e:
            logging.error(f"Error sorting candidates: {e}", exc_info=True)
            return []

def main(args):
    redis_ip = args.redis_ip
    redis_port = args.redis_port
    
    lm_path = args.lm_path
    gpu_number = args.gpu_number

    max_active = args.max_active
    min_active = args.min_active
    beam = args.beam
    lattice_beam = args.lattice_beam
    acoustic_scale = args.acoustic_scale
    ctc_blank_skip_threshold = args.ctc_blank_skip_threshold
    length_penalty = args.length_penalty
    nbest = args.nbest
    kbest = args.kbest
    top_candidates_to_augment = args.top_candidates_to_augment
    score_penalty_percent = args.score_penalty_percent
    blank_penalty = args.blank_penalty

    do_opt = args.do_opt          # acoustic scale = 0.8, blank penalty = 7, alpha = 0.5
    opt_cache_dir = args.opt_cache_dir
    alpha = args.alpha
    rescore = args.rescore
    input_stream = args.input_stream
    partial_output_stream = args.partial_output_stream
    final_output_stream = args.final_output_stream

    # expand user on paths
    lm_path = os.path.expanduser(lm_path)

    load_lexicon()
    decoder = NGram(
        max_active=max_active,
        min_active=min_active,
        beam=beam,
        lattice_beam=lattice_beam,
        acoustic_scale=acoustic_scale,
        ctc_blank_skip_threshold=ctc_blank_skip_threshold,
        length_penalty=length_penalty,
        nbest=nbest,
        k=kbest,
    )

    lm_args = {
        'lm_path': lm_path,
        'max_active': int(max_active),
        'min_active': int(min_active),
        'beam': float(beam),
        'lattice_beam': float(lattice_beam),
        'acoustic_scale': float(acoustic_scale),
        'ctc_blank_skip_threshold': float(ctc_blank_skip_threshold),
        'length_penalty': float(length_penalty),
        'nbest': int(nbest),
        'blank_penalty': float(blank_penalty),
        'alpha': float(alpha),
        'do_opt': int(do_opt),
        'rescore': int(rescore),
        'top_candidates_to_augment': int(top_candidates_to_augment),
        'score_penalty_percent': float(score_penalty_percent),
    }

    print("Starting n-gram decoder...")
    logging.info("Starting n-gram decoding")

    REDIS_STATE = -1
    logging.info(f"attempting to connect to redis at {redis_ip}:{redis_port}...")
    r = connect_to_redis_server(redis_ip, redis_port)
    while r is None:
        r = connect_to_redis_server(redis_ip, redis_port)
        if r is None:
            logging.warning(f"Could not connect to redis server at {redis_ip}:{redis_port}. Trying again in 3 seconds.")
            time.sleep(3)
    logging.info(f"successfully connected to redis server {redis_ip}:{redis_port}")
    print(f"successfully connected to redis server {redis_ip}:{redis_port}")

    timeout_ms = 100
    oldStr = ''
    prev_loop_start_time = 0
    
    logging.info("Entering main loop")
    while True:
        loop_time = time.time() - prev_loop_start_time
        if loop_time < 0.001:
            time.sleep(0.001 - loop_time)
        prev_loop_start_time = time.time()

        try:
            r.ping()
        except redis.exceptions.ConnectionError:
            if REDIS_STATE != 0:
                logging.error(f'Could not connect to the redis server at at {redis_ip}:{redis_port}! Trying again...')
            REDIS_STATE = 0
            time.sleep(1)
            continue
        else:
            if REDIS_STATE != 1:
                logging.info('Successfully connected to redis server')
                logits_last_entry_seen = get_current_redis_time_ms(r)
                reset_last_entry_seen = get_current_redis_time_ms(r)
                finalize_last_entry_seen = get_current_redis_time_ms(r)
                update_params_last_entry_seen = get_current_redis_time_ms(r)
            REDIS_STATE = 1
            
            if r.xlen('remote_lm_args') == 0:
                r.xadd('remote_lm_args', lm_args)

            lm_reset_stream = r.xread(
                    {'remote_lm_reset':reset_last_entry_seen},
                    count=1,
                    block=None,
            )

            if len(lm_reset_stream) > 0:
                for entry_id, entry_data in lm_reset_stream[0][1]:
                    reset_last_entry_seen = entry_id

                oldStr = ''
                decoder.reset()
                r.xadd('remote_lm_done_resetting', {'done': 1})
                # logging.info("Reset the language model")
                continue
            lm_finalize_stream = r.xread(
                    {'remote_lm_finalize':finalize_last_entry_seen},
                    count=1,
                    block=None,
            )
            if len(lm_finalize_stream) > 0:
                for entry_id, entry_data in lm_finalize_stream[0][1]:
                    finalize_last_entry_seen = entry_id
                if r.get('contextual_decoding_current_context') is not None:
                    current_context_str = r.get('contextual_decoding_current_context').decode().strip()
                    if len(current_context_str.split()) > 0:
                        logging.info(f'For LLM rescore, adding context str to the beginning of each candidate sentence:')
                        logging.info(f'\t"{current_context_str}"')
                else:
                    current_context_str = ''

                oldStr = ''

                try:
                    top_results = decoder.run_ngram()
                except Exception as e:
                    logging.error(f'Error in run_ngram(): {e}', exc_info=True)
                    top_results = []
                if top_results:
                    decoded_final = top_results[0]['sentence']
                else:
                    logging.error('No output from language model.')
                    decoded_final = ''

                scoring_payload = ''
                if top_results:  # Changed from "if nbest > 1 and top_results:"
                    nbest_redis = []
                    for result in top_results:
                        entry = ';'.join(map(str, [
                            result['sentence'],
                            float(result['acoustic_score']),  # Convert np.float32 to Python float
                            float(result['lm_score']),         # Convert np.float32 to Python float
                            0.0,
                            float(result['combined_score']),   # Convert np.float32 to Python float
                        ]))
                        nbest_redis.append(entry)
                    scoring_payload = ';'.join(nbest_redis)

                # logging.info(f'Final:  {decoded_final}')
                final_payload = {'lm_response_final': decoded_final}
                if current_context_str:
                    final_payload['context_str'] = current_context_str
                else:
                    final_payload['context_str'] = ''
                final_payload['scoring'] = scoring_payload
                r.xadd(final_output_stream, final_payload)

                # logging.info('Finalized the language model.\n')
                r.xadd('remote_lm_done_finalizing', {'done': 1})
                decoder.reset()
                continue

            # check if we need to update the decoder params
            update_params_stream = r.xread(
                {'remote_lm_update_params': update_params_last_entry_seen},
                count=1,
                block=None,
            )
            if len(update_params_stream) > 0:
                for entry_id, entry_data in update_params_stream[0][1]:
                    update_params_last_entry_seen = entry_id

                    max_active = int(entry_data.get(b'max_active', max_active))
                    min_active = int(entry_data.get(b'min_active', min_active))
                    beam = float(entry_data.get(b'beam', beam))
                    lattice_beam = float(entry_data.get(b'lattice_beam', lattice_beam))
                    acoustic_scale = float(entry_data.get(b'acoustic_scale', acoustic_scale))
                    ctc_blank_skip_threshold = float(entry_data.get(b'ctc_blank_skip_threshold', ctc_blank_skip_threshold))
                    length_penalty = float(entry_data.get(b'length_penalty', length_penalty))
                    nbest = int(entry_data.get(b'nbest', nbest))
                    blank_penalty = float(entry_data.get(b'blank_penalty', blank_penalty))
                    alpha = float(entry_data.get(b'alpha', alpha))
                    do_opt = int(entry_data.get(b'do_opt', do_opt))
                    rescore = int(entry_data.get(b'rescore', rescore))
                    top_candidates_to_augment = int(entry_data.get(b'top_candidates_to_augment', top_candidates_to_augment))
                    score_penalty_percent = float(entry_data.get(b'score_penalty_percent', score_penalty_percent))

                    # make sure that the update remote lm args are put into redis nicely
                    lm_args = {
                        'lm_path': lm_path,
                        'max_active': int(max_active),
                        'min_active': int(min_active),
                        'beam': float(beam),
                        'lattice_beam': float(lattice_beam),
                        'acoustic_scale': float(acoustic_scale),
                        'ctc_blank_skip_threshold': float(ctc_blank_skip_threshold),
                        'length_penalty': float(length_penalty),
                        'nbest': int(nbest),
                        'blank_penalty': float(blank_penalty),
                        'alpha': float(alpha),
                        'do_opt': int(do_opt),
                        'rescore': int(rescore),
                        'top_candidates_to_augment': int(top_candidates_to_augment),
                        'score_penalty_percent': float(score_penalty_percent),
                    }
                    r.xadd('remote_lm_args', lm_args)
                    
                    # update ngram parameters
                    # update_ngram_params(
                    #     ngramDecoder,
                    #     max_active = max_active,
                    #     min_active = min_active,
                    #     beam = beam,
                    #     lattice_beam = lattice_beam,
                    #     acoustic_scale = acoustic_scale,
                    #     ctc_blank_skip_threshold = ctc_blank_skip_threshold,
                    #     length_penalty = length_penalty,
                    #     nbest = nbest,
                    # )
                    logging.info(
                        f'Updated language model params:' +
                        f'\n\tmax_active = {max_active}' +
                        f'\n\tmin_active = {min_active}' +
                        f'\n\tbeam = {beam}' +
                        f'\n\tlattice_beam = {lattice_beam}' +
                        f'\n\tacoustic_scale = {acoustic_scale}' +
                        f'\n\tctc_blank_skip_threshold = {ctc_blank_skip_threshold}' +
                        f'\n\tlength_penalty = {length_penalty}' +
                        f'\n\tnbest = {nbest}' +
                        f'\n\tblank_penalty = {blank_penalty}' +
                        f'\n\talpha = {alpha}' +
                        f'\n\tdo_opt = {do_opt}' +
                        f'\n\trescore = {rescore}' +
                        f'\n\ttop_candidates_to_augment = {top_candidates_to_augment}' +
                        f'\n\tscore_penalty_percent = {score_penalty_percent}'
                    )
                    r.xadd('remote_lm_done_updating_params', {'done': 1})
                    decoder.beam = beam
                    decoder.k = max(decoder.k, nbest)
                    decoder.nbest = nbest
                    decoder.acoustic_scale = acoustic_scale

                continue
            
            # ------------------------------------------------------------------------------------------------------------------------
            # ------------ The loop can only get down to here if we're not finalizing, resetting, or updating params -----------------
            # ------------------------------------------------------------------------------------------------------------------------

            # try to read logits from redis stream
            try:
                read_result = r.xread(
                    {input_stream: logits_last_entry_seen},
                    count = 1,
                    block = timeout_ms
                )
            except redis.exceptions.ConnectionError:
                if REDIS_STATE != 0:
                    logging.error(f'Could not connect to the redis server at at {redis_ip}:{redis_port}! I will keep trying...')
                REDIS_STATE = 0
                time.sleep(1)
                continue

            if (len(read_result) >= 1): 
                # --------------- Read input stream --------------------------------
                for entry_id, entry_data in read_result[0][1]:
                    logits_last_entry_seen = entry_id
                    logits = np.frombuffer(entry_data[b'logits'], dtype=np.float32)

                # reshape logits to (T, 41)
                logits = logits.reshape(-1, 41)

                logits_tensor = torch.from_numpy(logits)
                log_probs = torch.log_softmax(logits_tensor, dim=-1).numpy()
                if blank_penalty > 0:
                    log_probs[:, 0] -= math.log(blank_penalty)

                decoder.process_input_frame(log_probs)

                decoded_partial = decoder.get_partial_sentence()
                if decoded_partial:
                    newStr = f'Partial: {decoded_partial}'
                else:
                    newStr = 'Partial: [NONE]'
                    decoded_partial = ''
                if oldStr != newStr:
                    # logging.info(newStr)
                    oldStr = newStr
                r.xadd(partial_output_stream, {'lm_response_partial': decoded_partial})

            else:
                # timeout if no data received for X ms
                # logging.warning(F'No logits came in for {timeout_ms} ms.')
                continue

if __name__ == "__main__":
    # load_lexicon()
    # decoder = NGram()
    # lattice = decoder.run_lexicon("DH AH K AE T S AE T")
    # perplexities = {}
    # for s in lattice:
    #     perplexities[s] = decoder.get_ngram_probability(s)
    # top_5_seqs = heapq.nsmallest(5, perplexities, key=perplexities.get)
    # print(top_5_seqs)

    # lattice_2 = decoder.run_lexicon("AY W AA N T T UW G OW HH OW M")
    # perplexities_2 = {}
    # for s in lattice_2:
    #     perplexities_2[s] = decoder.get_ngram_probability(s)
    # top_5_seqs_2 = heapq.nsmallest(5, perplexities_2, key=perplexities_2.get)
    # print(top_5_seqs_2)

    # logits = np.random.randn(1, 41)
    # logits = logits / np.sum(logits)
    # logits[0][-1] = np.max(logits)
    # decoder.process_input_frame(logits)

    # logits_2 = np.random.randn(1, 41)
    # logits_2 = (logits_2 / np.sum(logits_2)).tolist()
    # decoder.process_input_frame(logits_2)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--lm_path', type=str, help='Path to language model folder')
    parser.add_argument('--gpu_number', type=int, default=0, help='GPU number to use')

    parser.add_argument('--max_active', type=int, default=7000, help='max_active param for LM')
    parser.add_argument('--min_active', type=int, default=200, help='min_active param for LM')
    parser.add_argument('--beam', type=float, default=17.0, help='beam param for LM')
    parser.add_argument('--lattice_beam', type=float, default=8.0, help='lattice_beam param for LM')
    parser.add_argument('--ctc_blank_skip_threshold', type=float, default=1., help='ctc_blank_skip_threshold param for LM')
    parser.add_argument('--length_penalty', type=float, default=0.0, help='length_penalty param for LM')
    parser.add_argument('--acoustic_scale', type=float, default=0.3, help='Acoustic scale for LM')
    parser.add_argument('--nbest', type=int, default=5, help='# of candidate sentences for LM decoding')
    parser.add_argument('--kbest', type=int, default=20, help='# of candidate sentences for LM decoding')
    parser.add_argument('--top_candidates_to_augment', type=int, default=20, help='# of top candidates to augment')
    parser.add_argument('--score_penalty_percent', type=float, default=0.01, help='Score penalty percent for augmented candidates')
    parser.add_argument('--blank_penalty', type=float, default=9.0, help='Blank penalty for LM')

    parser.add_argument('--rescore', action='store_true', help='Use an unpruned ngram model for rescoring?')
    parser.add_argument('--do_opt', action='store_true', help='Use the opt model for rescoring?')
    parser.add_argument('--opt_cache_dir', type=str, default=None, help='path to opt cache')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha value [0-1]: Higher = more weight on OPT rescore. Lower = more weight on ngram rescore')

    parser.add_argument('--redis_ip', type=str, default='192.168.150.2', help='IP of the redis stream (string)')
    parser.add_argument('--redis_port', type=int, default=6379, help='Port of the redis stream (int)')
    parser.add_argument('--input_stream', type=str, default="remote_lm_input", help='Input stream containing logits')
    parser.add_argument('--partial_output_stream', type=str, default="remote_lm_output_partial", help='Output stream containing partial decoded sentences')
    parser.add_argument('--final_output_stream', type=str, default="remote_lm_output_final", help='Output stream containing final decoded sentences')

    args = parser.parse_args()
    main(args)

