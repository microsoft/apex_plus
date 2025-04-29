hf_token=""
/bin/bash scale_4TP.sh meta-llama/Llama-3.1-70B-Instruct llama $hf_token
/bin/bash scale_2TP_llama.sh meta-llama/Llama-3.1-70B-Instruct llama $hf_token
/bin/bash scale_4TP_fp8.sh alpindale/Mistral-Large-Instruct-2407-FP8 mistral $hf_token
/bin/bash scale_2TP_mistral.sh alpindale/Mistral-Large-Instruct-2407-FP8 mistral $hf_token