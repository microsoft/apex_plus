# Traces of Requests

This folder contains several traces of requests, with difference workload characteristics:  
* summarization: long context, short generation requests.  
* creation: short context, long generation requests.  
* lmsys: short conversations between LLM and a user; mostly short context, short generation requests.  

The three types of requests are located in two folders, _llama_ and _mistral_. 
The two model families use different tokenizer, so the token count of the requests are slightly different.
We tokenize the requests with their respective tokenizer, and save the processed traces in the two folders.

## Naming of the traces
For APEX simulation purposes, traces with the suffix ``xDPyTP_z`` can be ignored, such as ``creation_05_2DP4TP_0``. Theses are requests that have been split in a round-robin fashionsed, and are used for enabling data parallelism in LLM serving. They are used by the scripts in the _vLLM_ folder. 

The suffix after the trace name indicates the arrival rate of the requests, following a Poisson Distribution. For example, ``creation_05.jsonl`` consists of requests with an arrival rate of 0.5, and ``creation_025.jsonl`` consists of the same set of requests, but with an arrival rate of 0.25.