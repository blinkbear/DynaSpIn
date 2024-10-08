#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  100
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;

    struct llama_sampling_context * ctx_sampling;
};

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.model_draft.empty()) {
        fprintf(stderr, "%s: error: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for accepting a token from the draft model
    const float p_accept = params.p_accept;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_split  = params.p_split;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("speculative", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    params.logits_all = true;
    std::tie(model_tgt, ctx_tgt) = llama_init_from_gpt_params(params);

    llama_split_comm(ctx_tgt, (llama_node_id(ctx_tgt) == 0 || llama_node_id(ctx_tgt) == params.mpi_layer_split[0].size()) ? 0 : -1);
    llama_swap_comm(ctx_tgt);

    llama_split_comm(ctx_tgt, (llama_node_id(ctx_tgt) < params.mpi_layer_split[0].size()) ? 0 : -1);
    printf("Size of first split: %lu, element: %f\n", params.mpi_layer_split[0].size(), params.mpi_layer_split[0][0]);

    // load the draft model
    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
    std::tie(model_dft, ctx_dft) = llama_init_from_gpt_params(params);

    llama_split_comm(ctx_dft, (llama_node_id(ctx_dft) == 0 || llama_node_id(ctx_dft) == params.mpi_layer_split[0].size()) ? 0 : -1);
    llama_swap_comm(ctx_dft);

    llama_split_comm(ctx_dft, (llama_node_id(ctx_dft) >= params.mpi_layer_split[0].size()) ? 0 : -1);

    printf("Size of second split: %lu, element: %f\n", params.mpi_layer_split[1].size(), params.mpi_layer_split[1][0]);


    llama_split_layers_weighted(ctx_tgt, params.mpi_layer_split[0].data(), params.mpi_layer_split[0].size());
    llama_split_layers_weighted(ctx_dft, params.mpi_layer_split[1].data(), params.mpi_layer_split[1].size());

    {
        LOG_TEE("\n");
        LOG_TEE("%s\n", get_system_info(params).c_str());
    }

    {
        const int n_vocab_tgt = llama_n_vocab(model_tgt);
        const int n_vocab_dft = llama_n_vocab(model_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
                                ? n_vocab_tgt - n_vocab_dft
                                : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            fprintf(stderr, "%s: error: draft model vocab must closely match target model to use speculation but ", __func__);
            fprintf(stderr, "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_n_vocab(model_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

//        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
//            const char * token_text_tgt = llama_token_get_text(model_tgt, i);
//            const char * token_text_dft = llama_token_get_text(model_dft, i);
//            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
//                fprintf(stderr, "%s: error: draft model vocab must match target model to use speculation but ", __func__);
//                fprintf(stderr, "token %d content differs - target '%s', draft '%s'\n", i,
//                        llama_token_to_piece(ctx_tgt, i).c_str(),
//                        llama_token_to_piece(ctx_dft, i).c_str());
//                return 1;
//            }
//        }
    }


    // Tokenize the prompt
    const bool add_bos_tgt = llama_should_add_bos_token(model_tgt);
    LOG("add_bos tgt: %d\n", add_bos_tgt);

    const bool add_bos_dft = llama_should_add_bos_token(model_dft);
    LOG("add_bos dft: %d\n", add_bos_dft);

    if (add_bos_tgt != add_bos_dft) {
        fprintf(stderr, "%s: error: draft model add_bos must match target model to use speculation but ", __func__);
        fprintf(stderr, "add_bos_dft = %d while add_bos_tgt = %d\n", add_bos_dft, add_bos_tgt);
        return 1;
    }

    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx_tgt, params.prompt, add_bos_tgt, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    for (auto id : inp) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx_tgt, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_cycles();

    // eval the prompt with both models
    int32_t batch_id = 0;

    llama_batch batch_dft = llama_batch_init(params.n_ctx, 0, n_seq_dft+1);
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, n_seq_dft+1);
    llama_batch batch_tgt_async = llama_batch_init(params.n_ctx, 0, n_seq_dft+1);

    batch_dft.batch_id = batch_id;
    batch_tgt.batch_id = batch_id;
    batch_tgt_async.batch_id = batch_id;

    std::vector<llama_seq_id> seq_ids;
    for (int i = 0; i <= n_seq_dft; i++) {
        seq_ids.emplace_back(i);
    }

    for (int i = 0; i < n_input-1; i++) {
        llama_batch_add(batch_dft, inp[i], i, seq_ids, true);
        llama_batch_add(batch_tgt, inp[i], i, seq_ids, true);
    }
    llama_decode(ctx_tgt, batch_tgt);
    llama_batch_clear(batch_tgt);
    llama_batch_add(batch_dft, inp.back(), n_input-1, seq_ids, true);
    llama_batch_add(batch_tgt, inp.back(), n_input-1, seq_ids, true);

    // eval the prompt with both models
    llama_decode(ctx_tgt, batch_tgt);
    llama_decode(ctx_dft, batch_dft);

    const auto t_enc_end = ggml_cycles();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_n_vocab(model_dft));

    // how many tokens to draft each time
    int n_draft = params.n_draft;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    params.sparams.grammar.clear(); // the draft samplers will copy the target sampler's grammar
    params.sparams.temp = -1.0f;    // force greedy sampling with probs for the draft model

    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].ctx_sampling = llama_sampling_init(params.sparams);
    }



    const auto t_dec_start = ggml_cycles();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    long ttft = ggml_cycles();
    std::vector<uint64_t > inter_token_times;
    int64_t itt_start;
    bool first_token = false;
    bool has_run_first_token = false;

    while (true) {
        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            const auto & tokens = drafts[s].tokens;

            LOG("draft %d: %s\n", s, LOG_TOKENS_TOSTR_PRETTY(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        while (true) {
            LOG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);

            // sample from the target model
            llama_token id = llama_sampling_sample(ctx_sampling, ctx_tgt, NULL, drafts[s_keep].i_batch_tgt[i_dft]);

            llama_swap_comm(ctx_tgt);
            llama_sync_token(ctx_tgt, &id, 0);

            if (has_run_first_token) {
                if (first_token) {
                    ttft = ggml_cycles() - ttft;
                    LOG("\nTTFT: %ld\n", ttft);
                    first_token = false;
                } else {
                    inter_token_times.push_back(ggml_cycles() - itt_start);
                }

                itt_start = ggml_cycles();
            }
            llama_sampling_accept(ctx_sampling, ctx_tgt, id, true);

            //LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, ctx_sampling->prev).c_str());

            const std::string token_str = llama_token_to_piece(ctx_tgt, id);
            if (llama_node_id(ctx_tgt) == 0) {
                printf("%s", token_str.c_str());
                fflush(stdout);
            }

            llama_swap_comm(ctx_tgt);

            if (id == llama_token_eos(model_tgt)) {
                has_eos = true;
            }

            ++n_predict;

            // check if the target token matches any of the drafts
            {
                bool matches = false;

                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].active) {
                        continue;
                    }

                    if (i_dft < (int) drafts[s].tokens.size() && id == drafts[s].tokens[i_dft]) {
                        LOG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, id, token_str.c_str());

                        s_keep = s;
                        matches = true;
                    } else {
                        drafts[s].active = false;
                    }
                }

                if (matches) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;

                    continue;
                }
            }

            LOG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

            // TODO: simplify
            {
                LOG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                llama_kv_cache_seq_keep(ctx_dft, s_keep);
                llama_kv_cache_seq_cp  (ctx_dft, s_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_dft, 0);

                llama_kv_cache_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1);
                llama_kv_cache_seq_keep(ctx_tgt, s_keep);
                llama_kv_cache_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_tgt, 0);
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(id);
            drafts[0].i_batch_tgt.push_back(0);

            llama_batch_clear(batch_dft);
            llama_batch_add  (batch_dft, id, n_past_dft, { 0 }, true);

            llama_kv_cache_seq_rm(ctx_dft, -1, n_past_dft, -1);
            LOG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
            llama_decode         (ctx_dft, batch_dft);

            ++n_past_dft;

            break;
        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }

        llama_sampling_cp(ctx_sampling, drafts[0].ctx_sampling);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        llama_batch_clear(batch_tgt);
        llama_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                llama_sampling_sample(drafts[s].ctx_sampling, ctx_dft, NULL, drafts[s].i_batch_dft);

                llama_swap_comm(ctx_dft);
                llama_sync_token(ctx_dft, &(drafts[s].i_batch_dft), 1);

                auto & cur_p = drafts[s].ctx_sampling->cur;

                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p.size()); ++k) {
                    llama_sync_token_data(ctx_dft, &(cur_p[k]), 1);
                    LOG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, s, i, cur_p[k].id, cur_p[k].p, llama_token_to_piece(ctx_dft, cur_p[k].id).c_str());
                }

                llama_swap_comm(ctx_dft);


                if (cur_p[0].p < p_accept) {
                    LOG("stopping drafting for seq %3d, probability too low: %.3f < %.3f\n", s, cur_p[0].p, p_accept);
                    drafts[s].drafting = false;
                    continue;
                }

                std::vector<int> sa(1, s);

                // attempt to split the branch if the probability is high enough
                for (int f = 1; f < 8; ++f) {
                    if (n_seq_cur < n_seq_dft && cur_p[f].p > p_split) {
                        LOG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        llama_kv_cache_seq_rm(ctx_dft,    n_seq_cur, -1, -1);
                        llama_kv_cache_seq_cp(ctx_dft, s, n_seq_cur, -1, -1);

                        // all previous tokens from this branch are now also part of the new branch
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // copy the draft state
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        llama_sampling_cp(drafts[s].ctx_sampling, drafts[n_seq_cur].ctx_sampling);

                        sa.push_back(n_seq_cur);

                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p[is].id;

                    const int s = sa[is];

                    llama_sampling_accept(drafts[s].ctx_sampling, ctx_dft, id, true);

                    drafts[s].tokens.push_back(id);

                    // add unique drafted tokens to the target batch
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                    llama_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // add the token to the batch for batched decoding with the draft model
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                    llama_batch_add(batch_dft, id, n_past_cur, { s }, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur;
            ++n_drafted;

            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            llama_kv_cache_seq_keep(ctx_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_kv_cache_seq_cp(ctx_tgt, 0, s, -1, -1);
            }

            if (!has_run_first_token) {

                has_run_first_token = true;
                first_token = true;
            }

            // LOG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_decode(ctx_tgt, batch_tgt);
            ++n_past_tgt;
        }

        // the first token is always proposed by the traget model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
        }
    }

    auto t_dec_end = ggml_cycles();

    uint64_t avg_itt = 0;
    for (auto latency : inter_token_times) {
        avg_itt += latency;
    }

    avg_itt = avg_itt / inter_token_times.size();

    LOG_TEE("\n\n");

    llama_swap_comm(ctx_tgt);
    if (llama_node_id(ctx_tgt) == 0) {
        std::ofstream results;
        results.open("results.csv", std::ios_base::app);
        double encode_speed = (double)inp.size() / ((double)(t_enc_end - t_enc_start) / 1e6f);
        double decode_speed = (double)n_predict / ((double)(t_dec_end - t_dec_start) / 1e6f);
        double itt = (double)avg_itt / 1e6f;
        double final_ttft = (double)ttft / 1e6f;
        results << encode_speed << ',' << decode_speed << ',' << itt << ',' << final_ttft << '\n';
        LOG_TEE("OVERALL SYSTEM MEASUREMENTS:\n");
        LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input, (t_enc_end - t_enc_start) / 1e6f,
                encode_speed);
        LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f,
                decode_speed);
        LOG_TEE("Average inter-token latency: %f seconds\n", itt);
        LOG_TEE("Time-to-first-token: %f seconds\n", final_ttft);
        results.close();

    }
    LOG_TEE("\n");
    const auto *header = (llama_node_id(ctx_tgt) == 0) ? "TARGET PIPELINE MEASUREMENTS\n" : "DRAFT PIPELINE MEASUREMENTS\n";
    LOG_TEE("%s", header);
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    if (llama_node_id(ctx_tgt) != 0) {

        LOG_TEE("\ndraft:\n");
        llama_print_timings(ctx_dft);
    }

    if (llama_node_id(ctx_tgt) == 0) {

        LOG_TEE("\ntarget:\n");
        llama_print_timings(ctx_tgt);
    }

    llama_swap_comm(ctx_tgt);

    llama_sampling_free(ctx_sampling);
    for (int s = 0; s < n_seq_dft; ++s) {
        llama_sampling_free(drafts[s].ctx_sampling);
    }

    llama_batch_free(batch_dft);

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    llama_free(ctx_dft);
    llama_free_model(model_dft);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
