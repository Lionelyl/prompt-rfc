# ------ nlp2promela -----
cleanTemporary:
	- rm -rf TEMPORARY*

clean:
	- rm *.trail
	- rm *.pml
	- rm *.png
	- rm -rf out
	- rm -rf net-rem-*
	- rm _spin_nvr.tmp
	- rm *tmp*
	- rm pan*
	- rm ._n_i_p_s_
	make cleanTemporary
	- rm dot.*

.PHONY: nlp2promela

tcp2promela:
	python3 nlp2promela/nlp2promela.py
	make cleanTemporary

dccp2promela:
	python3 nlp2promela/nlp2promela.py
	make cleanTemporary

tcpbert2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted/bert_pretrained_rfcs_crf_phrases_feats/TCP.xml
	make cleanTemporary

dccpbert2promela:
	python3 nlp2promela/nlp2promela.py rfcs-predicted/bert_pretrained_rfcs_crf_phrases_feats/DCCP.xml
	make cleanTemporary

# ------ nlp-parser -------
# ------ template-free ------
tcptfcrf:
	python3 models/entlm_crf_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol TCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 2 \
			--do_train                                                      \
			--redundancy \
			--handmaded_label_map_path data/label_map_handmade.json
#			--warmup\
#			--label_map_path data/label_map_data_ratio0.6_multitoken_top6.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.json  \

dccptfcrf:
	python3 models/entlm_crf_rfc.py          \
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol DCCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 2 \
			--do_train                                                      \
			--handmaded_label_map_path data/label_map_handmade.json
#			--warmup \
#			--label_map_path data/label_map_data_ratio0.6_multitoken_top6.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.json

bgpv4entlmcrf:
	python3 models/entlm_crf_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol BGPv4                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 3 \
			--do_train                                                      \
			--redundancy \
			--handmaded_label_map_path data/label_map_handmaded_02.json
#			--warmup\
#			--label_map_path data/label_map_data_ratio0.6_multitoken_top6.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.json  \

ltpentlmcrf:
	python3 models/entlm_crf_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol LTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 1 \
			--do_train                                                      \
			--warmup \
			--handmaded_label_map_path data/label_map_handmaded_02.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.json

pptpentlmcrf:
	python3 models/entlm_crf_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol PPTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 1 \
			--do_train                                                      \
			--handmaded_label_map_path data/label_map_handmaded_02.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.json

sctpentlmcrf:
	python3 models/entlm_crf_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol SCTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 0 \
			--do_train                                                      \
			--handmaded_label_map_path data/label_map_handmaded_02.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.jso

#=== not crf ====

tcptftrain:
	python3 nlp-parser/template-free_rfc.py                      			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol TCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 3 \
			--do_train \
			--warmup \
			--domain_label_map_path data/label_map_domain.json
#			--label_map_path data/label_map_data_ratio0.6_multitoken_top6.json
dccptftrain:
	python3 nlp-parser/template-free_rfc.py                      			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol DCCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 3 \
			--warmup \
			--domain_label_map_path data/label_map_domain.json \
			--write_results
#			--do_train \
#			--redundancy
#			--label_map_path data/label_map_data_ratio0.6_multitoken_top6.json

bgpv4entlm:
	python3 models/entlm_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol BGPv4                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 1 \
			--do_train                                                      \
			--warmup \
			--handmaded_label_map_path data/label_map_handmaded_02.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.json

ltptftrain:
	python3 nlp-parser/template-free_rfc.py                      			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol LTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 2 \
			--do_train                                                      \
			--domain_label_map_path data/label_map_domain.json\
			--warmup
#			--label_map_path data/label_map_data_ratio0.6_multitoken_top6.json \

pptpentlm:
	python3 models/entlm_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol PPTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 1 \
			--do_train                                                      \
			--warmup \
			--handmaded_label_map_path data/label_map_handmaded_02.json
			-#-label_map_path data/label_map_handmaded_multitoken_top6.json

sctpentlm:
	python3 models/entlm_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol SCTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 0 \
			--do_train                                                      \
			--warmup \
			--handmaded_label_map_path data/label_map_handmaded_02.json
			-#-label_map_path data/label_map_handmaded_multitoken_top6.json

# ----- template-based -----
tcptbtrain:
	python3 nlp-parser/template-based_rfc.py								\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol TCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--cuda_device 0 \
		  	--template_id 0 \
		  	--warmup
#		  	--multi_template \
#			--template_num 3 \
#		  	--soft_template \
#		  	--template_len 20 \
#			--few_shot 5

dccptbtrain:
	python3 nlp-parser/template-based_rfc.py								\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol DCCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--cuda_device 1 \
		  	--template_id 1 \
#		  	--warmup
#		  	--multi_template \
#			--template_num 3 \

bgpv4tbtrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol BGPv4                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 1 \
			--template_num 3 \
		  	--template_id 0

ltptbtrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol LTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 3 \
			--template_num 3 \
		  	--template_id 0

pptptbtrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol PPTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 2 \
			--template_num 3 \
		  	--template_id 0

sctptbtrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol SCTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 3 \
			--template_num 3 \
		  	--template_id 0

# ----- few-shot -----
tcpfewshot:
	python3 models/bert_bert_mlp_cuda_few-shot.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol TCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 1 \
			--template_num 3 \
		  	--template_id 2 \
		  	--multi_template

dccpfewshot:
	python3 models/bert_bert_mlp_cuda_few-shot.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol DCCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 1 \
			--template_num 3 \
		  	--template_id 0 \
#		  	--multi_template

# --------- original rfcnlp --------------
obgpv4berttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol BGPv4                                                  \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0

opptpberttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol PPTP                                                  \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0

osctpberttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol SCTP                                                  \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0

oltpberttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol LTP                                                  \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0

odccpberttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol DCCP                                                 \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0


# ------- linear -------

dccplineartrain:
	python3 models/linear.py            \
		--protocol DCCP                         \
		--stem                                  \
		--phrase_level                          \
		--outdir output                         \

tcplineartrain:
	python3 models/linear.py            \
		--protocol TCP                          \
		--stem                                  \
		--phrase_level                          \
		--outdir output                         \


# ------- cpu --------
tcpberttraincpu:
	python3 models/bert_bert_mlp.py                          \
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol TCP                                                 \
			--outdir output \
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 \

