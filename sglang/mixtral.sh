docker build -t sglang:mixtral .
hf_token="<token>"

/bin/bash 8TP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral summarization_025 $hf_token
/bin/bash 8TP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral creation_025 $hf_token
/bin/bash 8TP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral lmsys_025 $hf_token

/bin/bash 8EP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral summarization_025 $hf_token
/bin/bash 8EP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral creation_025 $hf_token
/bin/bash 8EP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral lmsys_025 $hf_token

/bin/bash 8TP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral summarization_05 $hf_token
/bin/bash 8TP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral creation_05 $hf_token
/bin/bash 8TP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral lmsys_05 $hf_token

/bin/bash 8EP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral summarization_05 $hf_token
/bin/bash 8EP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral creation_05 $hf_token
/bin/bash 8EP.sh mistralai/Mixtral-8x22B-Instruct-v0.1 mistral lmsys_05 $hf_token