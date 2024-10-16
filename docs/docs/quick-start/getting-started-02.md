---
sidebar_position: 3
---

# Getting Started II: An Example for Basic RAG

Let's walk through a quick example of **basic retrieval-augmented generation (RAG)** in DSPy. Specifically, let's build **a system for answering Tech questions**, e.g. about Linux or iPhone apps.

Install the latest DSPy via `pip install -U dspy` and follow along. You may also need to install PyTorch via `pip install torch`.

### 1) Continue from Getting Started I.

In [Getting Started I: Basic Question Answering](), we've set up the DSPy LM, loaded some data, and loaded a metric for evaluation.

Let's do these again and also download the corpus data that we will use for RAG search. The next cell will seek to download 4 GBs, so it may take a few minutes. A future version of this notebook will come with a cache that allows you to skip downloads and the pytorch installation.


```python
import os
import requests

urls = [
    'https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_500.json',
    'https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main/technology/test_collection.jsonl',
    'https://huggingface.co/dspy/cache/resolve/main/index.pt'
]

for url in urls:
    filename = os.path.basename(url)
    remote_size = int(requests.head(url, allow_redirects=True).headers.get('Content-Length', 0))
    local_size = os.path.getsize(filename) if os.path.exists(filename) else 0

    if local_size != remote_size:
        print(f"Downloading '{filename}'...")
        with requests.get(url, stream=True) as r, open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
```


```python
import ujson
import dspy
from dspy.evaluate import SemanticF1

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

with open('ragqa_arena_tech_500.json') as f:
    data = [dspy.Example(**d).with_inputs('question') for d in ujson.load(f)]
    trainset, valset, devset, testset = data[:50], data[50:150], data[150:300], data[300:500]

metric = SemanticF1()
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=3)
```

### 2) Set up your system's retriever.

As far as DSPy is concerned, you can plug in any Python code for calling tools or retrievers. Hence, for our RAG system, we can plug any tools for the search step. Here, we'll just use OpenAI Embeddings and PyTorch for top-K search, but this is not a special choice, just a convenient one.


```python
import torch
import functools
from litellm import embedding as Embed

with open("test_collection.jsonl") as f:
    corpus = [ujson.loads(line) for line in f]

index = torch.load('index.pt', weights_only=True)
max_characters = 4000 # >98th percentile of document lengths

@functools.lru_cache(maxsize=None)
def search(query, k=5):
    query_embedding = torch.tensor(Embed(input=query, model="text-embedding-3-small").data[0]['embedding'])
    topk_scores, topk_indices = torch.matmul(index, query_embedding).topk(k)
    topK = [dict(score=score.item(), **corpus[idx]) for idx, score in zip(topk_indices, topk_scores)]
    return [doc['text'][:max_characters] for doc in topK]
```

### 3) Build your first RAG `Module`.

In the previous guide, we looked at individual DSPy modules in isolation, e.g. `dspy.Predict("question -> answer")`.

What if we want to build a DSPy _program_ that has multiple steps? The syntax below with `dspy.Module` allows you to connect a few pieces together, in this case, our retriever and a generation module, so the whole system can be optimized.

Concretely, in the `__init__` method, you declare any sub-module you'll need, which in this case is just a `dspy.ChainOfThought('context, question -> response')` module that takes retrieved context, a question, and produces a response. In the `forward` method, you simply express any Python control flow you like, possibly using your modules. In this case, we first invoke the `search` function defined earlier and then invoke the `self.respond` ChainOfThought module.


```python
class RAG(dspy.Module):
    def __init__(self, num_docs=5):
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question, k=self.num_docs)
        return self.respond(context=context, question=question)
```


```python
rag = RAG()
rag(question="what are high memory and low memory on linux?")
```




    Prediction(
        reasoning="High memory and low memory in Linux refer to the organization of memory in the system, particularly in the context of the Linux kernel's virtual memory management. High memory is the portion of physical memory that is not directly mapped by the kernel's page tables, meaning that user-space applications cannot access it directly. Low memory, on the other hand, is the part of memory that the kernel can access directly. In a typical 32-bit architecture, the virtual memory is split into 3 GB for user space (low memory) and 1 GB for kernel space (high memory). The distinction is important for memory management, especially when dealing with physical memory that cannot be mapped contiguously. Understanding this split is crucial for developers working with the Linux kernel, as it affects how memory is allocated and accessed.",
        response="In Linux, high memory refers to the portion of physical memory that is not directly mapped by the kernel's page tables, making it inaccessible to user-space applications. Low memory is the segment that the kernel can access directly. In a typical 32-bit architecture, the memory is divided into 3 GB for user space (low memory) and 1 GB for kernel space (high memory). This organization is essential for efficient memory management and affects how the kernel interacts with physical memory, especially in scenarios where contiguous memory is required."
    )




```python
dspy.inspect_history()
```

    
    
    
    
    [34m[2024-10-16T07:57:28.872406][0m
    
    [31mSystem message:[0m
    
    Your input fields are:
    1. `context` (str)
    2. `question` (str)
    
    Your output fields are:
    1. `reasoning` (str)
    2. `response` (str)
    
    All interactions will be structured in the following way, with the appropriate values filled in.
    
    [[ ## context ## ]]
    {context}
    
    [[ ## question ## ]]
    {question}
    
    [[ ## reasoning ## ]]
    {reasoning}
    
    [[ ## response ## ]]
    {response}
    
    [[ ## completed ## ]]
    
    In adhering to this structure, your objective is: 
            Given the fields `context`, `question`, produce the fields `response`.
    
    
    [31mUser message:[0m
    
    [[ ## context ## ]]
    [1] «As far as I remember, High Memory is used for application space and Low Memory for the kernel. Advantage is that (user-space) applications cant access kernel-space memory.»
    [2] «For the people looking for an explanation in the context of Linux kernel memory space, beware that there are two conflicting definitions of the high/low memory split (unfortunately there is no standard, one has to interpret that in context): High memory defined as the totality of kernel space in VIRTUAL memory. This is a region that only the kernel can access and comprises all virtual addresses greater or equal than PAGE_OFFSET. Low memory refers therefore to the region of the remaining addresses, which correspond to the user-space memory accessible from each user process. For example: on 32-bit x86 with a default PAGE_OFFSET, this means that high memory is any address ADDR with ADDR ≥ 0xC0000000 = PAGE_OFFSET (i.e. higher 1 GB). This is the reason why in Linux 32-bit processes are typically limited to 3 GB. Note that PAGE_OFFSET cannot be configured directly, it depends on the configurable VMSPLIT_x options (source). To summarize: in 32-bit archs, virtual memory is by default split into lower 3 GB (user space) and higher 1 GB (kernel space). For 64 bit, PAGE_OFFSET is not configurable and depends on architectural details that are sometimes detected at runtime during kernel load. On x86_64, PAGE_OFFSET is 0xffff888000000000 for 4-level paging (typical) and 0xff11000000000000 for 5-level paging (source). For ARM64 this is usually 0x8000000000000000. Note though, if KASLR is enabled, this value is intentionally unpredictable. High memory defined as the portion of PHYSICAL memory that cannot be mapped contiguously with the rest of the kernel virtual memory. A portion of the kernel virtual address space can be mapped as a single contiguous chunk into the so-called physical low memory. To fully understand what this means, a deeper knowledge of the Linux virtual memory space is required. I would recommend going through these slides. From the slides: This kind of high/low memory split is only applicable to 32-bit architectures where the installed physical RAM size is relatively high (more than ~1 GB). Otherwise, i.e. when the physical address space is small (<1 GB) or when the virtual memory space is large (64 bits), the whole physical space can be accessed from the kernel virtual memory space. In that case, all physical memory is considered low memory. It is preferable that high memory does not exist at all because the whole physical space can be accessed directly from the kernel, which makes memory management a lot simpler and efficient. This is especially important when dealing with DMAs (which typically require physically contiguous memory). See also the answer by @gilles»
    [3] «Low and High do not refer to whether there is a lot of usage or not. They represent the way it is organized by the system. According to Wikipedia: High Memory is the part of physical memory in a computer which is not directly mapped by the page tables of its operating system kernel. There is no duration for the free command which simply computes a snapshot of the information available. Most people, including programmers, do not need to understand it more clearly as it is managed in a much simpler form through system calls and compiler/interpreter operations.»
    [4] «This is relevant to the Linux kernel; Im not sure how any Unix kernel handles this. The High Memory is the segment of memory that user-space programs can address. It cannot touch Low Memory. Low Memory is the segment of memory that the Linux kernel can address directly. If the kernel must access High Memory, it has to map it into its own address space first. There was a patch introduced recently that lets you control where the segment is. The tradeoff is that you can take addressable memory away from user space so that the kernel can have more memory that it does not have to map before using. Additional resources: http://tldp.org/HOWTO/KernelAnalysis-HOWTO-7.html http://linux-mm.org/HighMemory»
    [5] «HIGHMEM is a range of kernels memory space, but it is NOT memory you access but its a place where you put what you want to access. A typical 32bit Linux virtual memory map is like: 0x00000000-0xbfffffff: user process (3GB) 0xc0000000-0xffffffff: kernel space (1GB) (CPU-specific vector and whatsoever are ignored here). Linux splits the 1GB kernel space into 2 pieces, LOWMEM and HIGHMEM. The split varies from installation to installation. If an installation chooses, say, 512MB-512MB for LOW and HIGH mems, the 512MB LOWMEM (0xc0000000-0xdfffffff) is statically mapped at the kernel boot time; usually the first so many bytes of the physical memory is used for this so that virtual and physical addresses in this range have a constant offset of, say, 0xc0000000. On the other hand, the latter 512MB (HIGHMEM) has no static mapping (although you could leave pages semi-permanently mapped there, but you must do so explicitly in your driver code). Instead, pages are temporarily mapped and unmapped here so that virtual and physical addresses in this range have no consistent mapping. Typical uses of HIGHMEM include single-time data buffers.»
    
    [[ ## question ## ]]
    what are high memory and low memory on linux?
    
    Respond with the corresponding output fields, starting with the field `reasoning`, then `response`, and then ending with the marker for `completed`.
    
    
    [31mResponse:[0m
    
    [32m[[ ## reasoning ## ]]
    High memory and low memory in Linux refer to the organization of memory in the system, particularly in the context of the Linux kernel's virtual memory management. High memory is the portion of physical memory that is not directly mapped by the kernel's page tables, meaning that user-space applications cannot access it directly. Low memory, on the other hand, is the part of memory that the kernel can access directly. In a typical 32-bit architecture, the virtual memory is split into 3 GB for user space (low memory) and 1 GB for kernel space (high memory). The distinction is important for memory management, especially when dealing with physical memory that cannot be mapped contiguously. Understanding this split is crucial for developers working with the Linux kernel, as it affects how memory is allocated and accessed.
    
    [[ ## response ## ]]
    In Linux, high memory refers to the portion of physical memory that is not directly mapped by the kernel's page tables, making it inaccessible to user-space applications. Low memory is the segment that the kernel can access directly. In a typical 32-bit architecture, the memory is divided into 3 GB for user space (low memory) and 1 GB for kernel space (high memory). This organization is essential for efficient memory management and affects how the kernel interacts with physical memory, especially in scenarios where contiguous memory is required.
    
    [[ ## completed ## ]][0m
    
    
    
    
    


In the previous guide with a CoT module, we got nearly 40% in terms of semantic F1 on our `devset`. Would this `RAG` module score better?


```python
evaluate(RAG())
```

    Average Metric: 79.44815807647615 / 150  (53.0): 100%|██████████| 150/150 [00:05<00:00, 29.11it/s] 



<style type="text/css">
#T_2b09a th {
  text-align: left;
}
#T_2b09a td {
  text-align: left;
}
#T_2b09a_row0_col0, #T_2b09a_row0_col1, #T_2b09a_row0_col2, #T_2b09a_row0_col3, #T_2b09a_row0_col4, #T_2b09a_row1_col0, #T_2b09a_row1_col1, #T_2b09a_row1_col2, #T_2b09a_row1_col3, #T_2b09a_row1_col4, #T_2b09a_row2_col0, #T_2b09a_row2_col1, #T_2b09a_row2_col2, #T_2b09a_row2_col3, #T_2b09a_row2_col4 {
  text-align: left;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-width: 400px;
}
</style>
<table id="T_2b09a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_2b09a_level0_col0" class="col_heading level0 col0" >question</th>
      <th id="T_2b09a_level0_col1" class="col_heading level0 col1" >example_response</th>
      <th id="T_2b09a_level0_col2" class="col_heading level0 col2" >reasoning</th>
      <th id="T_2b09a_level0_col3" class="col_heading level0 col3" >pred_response</th>
      <th id="T_2b09a_level0_col4" class="col_heading level0 col4" >SemanticF1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_2b09a_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_2b09a_row0_col0" class="data row0 col0" >why is mercurial considered to be easier than git?</td>
      <td id="T_2b09a_row0_col1" class="data row0 col1" >Mercurial's syntax is considered more familiar, especially for those accustomed to SVN, and is well documented. It focuses on interface aspects, which initially makes learning...</td>
      <td id="T_2b09a_row0_col2" class="data row0 col2" >Mercurial is considered easier than Git for several reasons. Firstly, its syntax is more familiar to users coming from Subversion (SVN), making the transition smoother....</td>
      <td id="T_2b09a_row0_col3" class="data row0 col3" >Mercurial is considered easier than Git primarily because of its more familiar syntax for users transitioning from systems like Subversion (SVN), its straightforward workflow without...</td>
      <td id="T_2b09a_row0_col4" class="data row0 col4" >✔️ [0.797]</td>
    </tr>
    <tr>
      <th id="T_2b09a_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_2b09a_row1_col0" class="data row1 col0" >open finder window from current terminal location?</td>
      <td id="T_2b09a_row1_col1" class="data row1 col1" >If you type 'open .' in Terminal, it will open the current directory in a Finder window. Alternatively, you can execute the command open `pwd`...</td>
      <td id="T_2b09a_row1_col2" class="data row1 col2" >To open a Finder window from the current terminal location, you can use the command `open .` in the terminal. This command will open a...</td>
      <td id="T_2b09a_row1_col3" class="data row1 col3" >You can open a Finder window from your current terminal location by typing the command `open .` in the terminal. This will launch a Finder...</td>
      <td id="T_2b09a_row1_col4" class="data row1 col4" >✔️ [0.857]</td>
    </tr>
    <tr>
      <th id="T_2b09a_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_2b09a_row2_col0" class="data row2 col0" >how to import secret gpg key (copied from one machine to another)?</td>
      <td id="T_2b09a_row2_col1" class="data row2 col1" >It is advised that it is necessary to add `--import` to the command line to import the private key and that according to the man...</td>
      <td id="T_2b09a_row2_col2" class="data row2 col2" >To import a secret GPG key that has been copied from one machine to another, you typically need to use the `gpg --import` command. The...</td>
      <td id="T_2b09a_row2_col3" class="data row2 col3" >To import a secret GPG key that you have copied from one machine to another, follow these steps: 1. On the original machine, export your...</td>
      <td id="T_2b09a_row2_col4" class="data row2 col4" ></td>
    </tr>
  </tbody>
</table>





<div style='
    text-align: center;
    font-size: 16px;
    font-weight: bold;
    color: #555;
    margin: 10px 0;'>
    ... 147 more rows not displayed ...
</div>






    52.97



### 4) Using a DSPy `Optimizer` to improve your RAG prompt.

Off the shelf, our `RAG` module scores 53%. What are our options to make it stronger? One of the various choices DSPy offers is optimizing the prompts in our pipeline.

If there are many sub-modules in your program, all of them will be optimized together. In this case, there's only one: `self.respond = dspy.ChainOfThought('context, question -> response')`

Let's set up and use DSPy's [MIPRO (v2) optimizer](). The run below has a cost around $1.5 (for the `medium` auto setting) and may take some 20-30 minutes depending on your number of threads.


```python
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)  # use fewer threads if your rate limit is small

optimized_rag = tp.compile(RAG(), trainset=trainset, valset=valset,
                           max_bootstrapped_demos=2, max_labeled_demos=2,
                           requires_permission_to_run=False)
```

    
    RUNNING WITH THE FOLLOWING MEDIUM AUTO RUN SETTINGS:
    num_trials: 25
    minibatch: True
    num_candidates: 19
    valset size: 100
    
    
    ==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==
    These will be used as few-shot example candidates for our program and for creating instructions.
    
    Bootstrapping N=19 sets of demonstrations...
    Bootstrapping set 1/19
    Bootstrapping set 2/19
    Bootstrapping set 3/19


      6%|▌         | 3/50 [00:00<00:01, 31.54it/s]


    Bootstrapped 2 full traces after 4 examples in round 0.
    Bootstrapping set 4/19


      2%|▏         | 1/50 [00:00<00:02, 22.61it/s]


    Bootstrapped 1 full traces after 2 examples in round 0.
    Bootstrapping set 5/19


      2%|▏         | 1/50 [00:00<00:02, 22.40it/s]


    Bootstrapped 1 full traces after 2 examples in round 0.
    Bootstrapping set 6/19


      8%|▊         | 4/50 [00:00<00:01, 23.64it/s]


    Bootstrapped 2 full traces after 5 examples in round 0.
    Bootstrapping set 7/19


      4%|▍         | 2/50 [00:00<00:02, 22.72it/s]


    Bootstrapped 2 full traces after 3 examples in round 0.
    Bootstrapping set 8/19


     12%|█▏        | 6/50 [00:00<00:01, 28.92it/s]


    Bootstrapped 2 full traces after 7 examples in round 0.
    Bootstrapping set 9/19


      2%|▏         | 1/50 [00:00<00:02, 22.47it/s]


    Bootstrapped 1 full traces after 2 examples in round 0.
    Bootstrapping set 10/19


      6%|▌         | 3/50 [00:00<00:03, 15.29it/s]


    Bootstrapped 2 full traces after 4 examples in round 0.
    Bootstrapping set 11/19


     10%|█         | 5/50 [00:00<00:00, 52.23it/s]


    Bootstrapped 2 full traces after 6 examples in round 0.
    Bootstrapping set 12/19


      4%|▍         | 2/50 [00:00<00:01, 41.13it/s]


    Bootstrapped 1 full traces after 3 examples in round 0.
    Bootstrapping set 13/19


      2%|▏         | 1/50 [00:00<00:00, 340.72it/s]


    Bootstrapped 1 full traces after 2 examples in round 0.
    Bootstrapping set 14/19


      6%|▌         | 3/50 [00:00<00:00, 55.53it/s]


    Bootstrapped 2 full traces after 4 examples in round 0.
    Bootstrapping set 15/19


     10%|█         | 5/50 [00:00<00:00, 53.43it/s]


    Bootstrapped 2 full traces after 6 examples in round 0.
    Bootstrapping set 16/19


     16%|█▌        | 8/50 [00:00<00:00, 45.90it/s]


    Bootstrapped 2 full traces after 9 examples in round 0.
    Bootstrapping set 17/19


      2%|▏         | 1/50 [00:00<00:00, 464.23it/s]


    Bootstrapped 1 full traces after 2 examples in round 0.
    Bootstrapping set 18/19


      2%|▏         | 1/50 [00:00<00:00, 431.96it/s]


    Bootstrapped 1 full traces after 2 examples in round 0.
    Bootstrapping set 19/19


     14%|█▍        | 7/50 [00:00<00:01, 32.38it/s]


    Bootstrapped 2 full traces after 8 examples in round 0.
    
    ==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==
    We will use the few-shot examples from the previous step, a generated dataset summary, a summary of the program code, and a randomly selected prompting tip to propose instructions.
    
    Proposing instructions...
    
    Proposed Instructions for Predictor 0:
    
    0: Given the fields `context`, `question`, produce the fields `response`.
    
    1: Using the provided `context` and `question`, generate a detailed `response` that includes the reasoning behind the answer. Ensure that the response is user-friendly and accessible, addressing potential follow-up questions or concerns.
    
    2: Using the provided context, answer the question by synthesizing the relevant information and producing a clear and comprehensive response. Focus on practical solutions and ensure that the explanation is accessible to users with varying levels of technical expertise.
    
    3: Imagine you are in a critical situation where you need to type a normal tilde sign (~) during an important presentation on a Mac, but you're unsure how to do it due to your unfamiliar keyboard layout. Using the provided context, please explain the various methods to type the tilde symbol on different Mac keyboard layouts, ensuring that the response is clear and easy to follow for someone who may not be tech-savvy.
    
    4: Imagine you're in a situation where your Android phone is unresponsive and you need to shut it down urgently to prevent potential data loss or overheating. Using ADB commands is your only option. Given the context of various ADB commands and explanations, along with the question about how to shut down the phone using ADB, provide a detailed reasoning process that outlines the necessary steps and considerations for executing the command. Your response should include clear instructions and alternative commands for users, especially focusing on those with rooted devices.
    
    5: You are a technical support expert. Given the fields `context` (which provides relevant information about technology-related issues) and `question` (which pertains to that context), produce a well-reasoned `response` that clearly addresses the question while utilizing the information provided in the context. Make sure to explain any technical terms in a user-friendly manner to empower users to understand and resolve their technology-related queries effectively.
    
    6: Given the provided `context` and `question`, generate a detailed and reasoned `response` that addresses the user's inquiry, ensuring clarity and accessibility for users of varying expertise.
    
    7: You are a knowledgeable technical support specialist. Given the fields `context` and `question`, provide a detailed and clear response along with the reasoning behind your answer. Ensure that your response is user-friendly and accessible to individuals with varying levels of technical expertise.
    
    8: Using the provided context and question, analyze the situation to determine when a significant rewrite of a software codebase is warranted. Consider the potential benefits and risks, and synthesize a clear response that outlines the scenarios in which a big rewrite may be justified, while also emphasizing the importance of careful consideration and the potential advantages of refactoring.
    
    9: Using the provided context and question, generate a detailed response that explains the reasoning behind the answer. Ensure that the response is clear, informative, and tailored to the user's level of expertise, while also including any relevant technical details or alternative methods related to the question.
    
    10: Using the provided `context` and `question`, analyze the information step by step to generate a comprehensive and informative `response`. Ensure that the response clearly explains the concepts involved, highlights key distinctions, and addresses any complexities noted in the context.
    
    11: Given the detailed `context` about physical penetration testing scenarios and a specific `question` regarding actions to take if caught in such a situation, generate a comprehensive `response` that outlines the necessary steps to follow. Your response should emphasize the importance of established protocols, effective communication, and safety measures to ensure a clear understanding of the situation by all parties involved.
    
    12: In a high-stakes situation where you need to quickly access a parent folder in Finder to recover an important document before a deadline, please provide a clear and detailed response to the question: "Is there a keyboard shortcut to navigate one level up in Finder?" Include the reasoning behind your answer based on the given context.
    
    13: You are a tech support specialist. Given the fields `context` which includes various troubleshooting tips and relevant information, and `question` which is a user's inquiry about technology-related issues, produce a structured and clear `response` that offers practical advice and solutions based on the provided context.
    
    14: Using the provided `context`, answer the `question` in a clear and detailed manner, ensuring to include necessary steps, commands, and any prerequisites needed to execute the task effectively. Aim to provide a practical and user-friendly response that balances technical detail with accessibility for users of varying expertise.
    
    15: In a high-stakes situation where you are trying to recover critical files from a Mac with a broken screen before an important deadline, explain step-by-step how to transfer those files. Use the provided context about different methods for file recovery and ensure that your response includes both reasoning and a clear, actionable response for the user to follow.
    
    16: You are a technical support expert. Given the fields `context` and `question`, provide a detailed and user-friendly `response` that explains the concept of Ethernet VLAN tagging, including its purpose and the process involved in tagging packets.
    
    17: Given the provided `context` about opening RAR files in Linux and the `question`, generate a detailed `response` that includes multiple methods for opening RAR files, including both command-line and graphical user interface options, along with installation instructions and usage steps.
    
    18: Using the provided `context` and `question`, generate a detailed and informative `response` that clearly explains the topic at hand, ensuring to incorporate relevant details from the context and maintain clarity for users of varying expertise.
    
    
    
    Evaluating the default program...
    


    Average Metric: 55.26561625985762 / 100  (55.3): 100%|██████████| 100/100 [00:03<00:00, 28.59it/s]
    /opt/anaconda3/envs/jun2024_py310/lib/python3.10/site-packages/optuna/samplers/_tpe/sampler.py:319: ExperimentalWarning: ``multivariate`` option is an experimental feature. The interface can change in the future.
      warnings.warn(


    Default program score: 55.27
    
    ==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==
    We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.
    
    == Minibatch Trial 1 / 25 ==


    Average Metric: 15.31691037288488 / 25  (61.3): 100%|██████████| 25/25 [00:00<00:00, 466.33it/s] 


    Score: 61.27 on minibatch of size 25 with parameters ['Predictor 1: Instruction 12', 'Predictor 1: Few-Shot Set 7'].
    Minibatch scores so far: [61.27]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    ============================
    
    
    == Minibatch Trial 2 / 25 ==


    Average Metric: 16.437088041017294 / 25  (65.7): 100%|██████████| 25/25 [00:00<00:00, 430.03it/s]


    Score: 65.75 on minibatch of size 25 with parameters ['Predictor 1: Instruction 10', 'Predictor 1: Few-Shot Set 7'].
    Minibatch scores so far: [61.27, 65.75]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    ============================
    
    
    == Minibatch Trial 3 / 25 ==


    Average Metric: 14.162746297631099 / 25  (56.7): 100%|██████████| 25/25 [00:00<00:00, 416.56it/s]


    Score: 56.65 on minibatch of size 25 with parameters ['Predictor 1: Instruction 7', 'Predictor 1: Few-Shot Set 18'].
    Minibatch scores so far: [61.27, 65.75, 56.65]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    ============================
    
    
    == Minibatch Trial 4 / 25 ==


    Average Metric: 14.75654422981974 / 25  (59.0): 100%|██████████| 25/25 [00:00<00:00, 411.51it/s] 


    Score: 59.03 on minibatch of size 25 with parameters ['Predictor 1: Instruction 15', 'Predictor 1: Few-Shot Set 2'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    ============================
    
    
    == Minibatch Trial 5 / 25 ==


    Average Metric: 15.823429094294873 / 25  (63.3): 100%|██████████| 25/25 [00:00<00:00, 442.28it/s]


    Score: 63.29 on minibatch of size 25 with parameters ['Predictor 1: Instruction 8', 'Predictor 1: Few-Shot Set 18'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    ============================
    
    
    == Minibatch Trial 6 / 25 ==


    Average Metric: 14.10134028850213 / 25  (56.4): 100%|██████████| 25/25 [00:00<00:00, 464.80it/s] 


    Score: 56.41 on minibatch of size 25 with parameters ['Predictor 1: Instruction 7', 'Predictor 1: Few-Shot Set 1'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    ============================
    
    
    == Minibatch Trial 7 / 25 ==


    Average Metric: 14.883615045622088 / 25  (59.5): 100%|██████████| 25/25 [00:00<00:00, 577.70it/s]


    Score: 59.53 on minibatch of size 25 with parameters ['Predictor 1: Instruction 7', 'Predictor 1: Few-Shot Set 12'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    ============================
    
    
    == Minibatch Trial 8 / 25 ==


    Average Metric: 13.97898284911489 / 25  (55.9): 100%|██████████| 25/25 [00:00<00:00, 402.78it/s] 


    Score: 55.92 on minibatch of size 25 with parameters ['Predictor 1: Instruction 11', 'Predictor 1: Few-Shot Set 13'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    ============================
    
    
    == Minibatch Trial 9 / 25 ==


      0%|          | 0/25 [00:00<?, ?it/s][2m2024-10-16T14:57:40.643370Z[0m [[31m[1merror    [0m] [1mError for example in dev set: 		 Error parsing field recall: 1 validation error for float
      Input should be a valid number, unable to parse string as a number [type=float_parsing, input_value="1. The data plane is des...ing to a recall of 1.0.", input_type=str]
        For further information visit https://errors.pydantic.dev/2.7/v/float_parsing.
    
    		On attempting to parse the value
    ```
    1. The data plane is described as responsible for moving packets, which aligns with the ground truth.
    2. The control plane's role in making forwarding decisions is also covered.
    3. The management plane's function in configuring and monitoring is included.
    
    Overall, the system response covers all aspects of the ground truth, leading to a recall of 1.0.
    ```. Set `provide_traceback=True` to see the stack trace.[0m [[0m[1m[34mdspy.evaluate.evaluate[0m][0m [36mfilename[0m=[35mevaluate.py[0m [36mlineno[0m=[35m200[0m
    Average Metric: 14.995786364234506 / 25  (60.0): 100%|██████████| 25/25 [00:00<00:00, 466.77it/s]


    Score: 59.98 on minibatch of size 25 with parameters ['Predictor 1: Instruction 5', 'Predictor 1: Few-Shot Set 4'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    ============================
    
    
    == Minibatch Trial 10 / 25 ==


    Average Metric: 15.20260154504114 / 25  (60.8): 100%|██████████| 25/25 [00:00<00:00, 486.74it/s] 


    Score: 60.81 on minibatch of size 25 with parameters ['Predictor 1: Instruction 14', 'Predictor 1: Few-Shot Set 1'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81]
    Full eval scores so far: [55.27]
    Best full score so far: 55.27
    =============================
    
    
    ===== Full Eval 1 =====
    Doing full eval on next top averaging program (Avg Score: 65.75) from minibatch trials...


    Average Metric: 61.46656430890014 / 100  (61.5): 100%|██████████| 100/100 [00:00<00:00, 552.44it/s]


    [92mNew best full eval score![0m Score: 61.47
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =======================
    
    
    == Minibatch Trial 11 / 25 ==


    Average Metric: 15.70727116562876 / 25  (62.8): 100%|██████████| 25/25 [00:00<00:00, 279.48it/s] 


    Score: 62.83 on minibatch of size 25 with parameters ['Predictor 1: Instruction 17', 'Predictor 1: Few-Shot Set 17'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    == Minibatch Trial 12 / 25 ==


    Average Metric: 17.06488015824002 / 25  (68.3): 100%|██████████| 25/25 [00:00<00:00, 660.01it/s] 


    Score: 68.26 on minibatch of size 25 with parameters ['Predictor 1: Instruction 10', 'Predictor 1: Few-Shot Set 2'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    == Minibatch Trial 13 / 25 ==


    Average Metric: 13.089709097343112 / 25  (52.4): 100%|██████████| 25/25 [00:00<00:00, 469.46it/s]


    Score: 52.36 on minibatch of size 25 with parameters ['Predictor 1: Instruction 0', 'Predictor 1: Few-Shot Set 7'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    == Minibatch Trial 14 / 25 ==


    Average Metric: 17.45033822869943 / 25  (69.8): 100%|██████████| 25/25 [00:00<00:00, 529.37it/s] 


    Score: 69.8 on minibatch of size 25 with parameters ['Predictor 1: Instruction 10', 'Predictor 1: Few-Shot Set 2'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    == Minibatch Trial 15 / 25 ==


    Average Metric: 14.83249088881275 / 25  (59.3): 100%|██████████| 25/25 [00:00<00:00, 486.68it/s] 


    Score: 59.33 on minibatch of size 25 with parameters ['Predictor 1: Instruction 10', 'Predictor 1: Few-Shot Set 3'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    == Minibatch Trial 16 / 25 ==


    Average Metric: 15.907152635127742 / 25  (63.6): 100%|██████████| 25/25 [00:00<00:00, 648.07it/s]


    Score: 63.63 on minibatch of size 25 with parameters ['Predictor 1: Instruction 10', 'Predictor 1: Few-Shot Set 2'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    == Minibatch Trial 17 / 25 ==


    Average Metric: 14.563187344409487 / 25  (58.3): 100%|██████████| 25/25 [00:00<00:00, 424.20it/s]


    Score: 58.25 on minibatch of size 25 with parameters ['Predictor 1: Instruction 3', 'Predictor 1: Few-Shot Set 2'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63, 58.25]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    == Minibatch Trial 18 / 25 ==


    Average Metric: 14.06507666195472 / 25  (56.3): 100%|██████████| 25/25 [00:00<00:00, 436.31it/s] 


    Score: 56.26 on minibatch of size 25 with parameters ['Predictor 1: Instruction 1', 'Predictor 1: Few-Shot Set 6'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63, 58.25, 56.26]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    == Minibatch Trial 19 / 25 ==


    Average Metric: 13.72555644974912 / 25  (54.9): 100%|██████████| 25/25 [00:00<00:00, 200.99it/s]


    Score: 54.9 on minibatch of size 25 with parameters ['Predictor 1: Instruction 13', 'Predictor 1: Few-Shot Set 14'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63, 58.25, 56.26, 54.9]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    == Minibatch Trial 20 / 25 ==


    Average Metric: 14.349351500788876 / 25  (57.4): 100%|██████████| 25/25 [00:00<00:00, 385.64it/s]


    Score: 57.4 on minibatch of size 25 with parameters ['Predictor 1: Instruction 10', 'Predictor 1: Few-Shot Set 5'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63, 58.25, 56.26, 54.9, 57.4]
    Full eval scores so far: [55.27, 61.47]
    Best full score so far: 61.47
    =============================
    
    
    ===== Full Eval 2 =====
    Doing full eval on next top averaging program (Avg Score: 67.23) from minibatch trials...


    Average Metric: 62.56993370294195 / 100  (62.6): 100%|██████████| 100/100 [00:00<00:00, 480.69it/s]


    [92mNew best full eval score![0m Score: 62.57
    Full eval scores so far: [55.27, 61.47, 62.57]
    Best full score so far: 62.57
    =======================
    
    
    == Minibatch Trial 21 / 25 ==


    Average Metric: 15.254813285777045 / 25  (61.0): 100%|██████████| 25/25 [00:00<00:00, 431.79it/s]


    Score: 61.02 on minibatch of size 25 with parameters ['Predictor 1: Instruction 16', 'Predictor 1: Few-Shot Set 15'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63, 58.25, 56.26, 54.9, 57.4, 61.02]
    Full eval scores so far: [55.27, 61.47, 62.57]
    Best full score so far: 62.57
    =============================
    
    
    == Minibatch Trial 22 / 25 ==


    Average Metric: 14.931053171881244 / 25  (59.7): 100%|██████████| 25/25 [00:00<00:00, 457.10it/s]


    Score: 59.72 on minibatch of size 25 with parameters ['Predictor 1: Instruction 10', 'Predictor 1: Few-Shot Set 12'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63, 58.25, 56.26, 54.9, 57.4, 61.02, 59.72]
    Full eval scores so far: [55.27, 61.47, 62.57]
    Best full score so far: 62.57
    =============================
    
    
    == Minibatch Trial 23 / 25 ==


    Average Metric: 16.869837047530115 / 25  (67.5): 100%|██████████| 25/25 [00:00<00:00, 1726.99it/s]


    Score: 67.48 on minibatch of size 25 with parameters ['Predictor 1: Instruction 10', 'Predictor 1: Few-Shot Set 7'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63, 58.25, 56.26, 54.9, 57.4, 61.02, 59.72, 67.48]
    Full eval scores so far: [55.27, 61.47, 62.57]
    Best full score so far: 62.57
    =============================
    
    
    == Minibatch Trial 24 / 25 ==


    Average Metric: 15.033669626869123 / 25  (60.1): 100%|██████████| 25/25 [00:00<00:00, 627.95it/s]


    Score: 60.13 on minibatch of size 25 with parameters ['Predictor 1: Instruction 2', 'Predictor 1: Few-Shot Set 0'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63, 58.25, 56.26, 54.9, 57.4, 61.02, 59.72, 67.48, 60.13]
    Full eval scores so far: [55.27, 61.47, 62.57]
    Best full score so far: 62.57
    =============================
    
    
    == Minibatch Trial 25 / 25 ==


    Average Metric: 14.566001625589312 / 25  (58.3): 100%|██████████| 25/25 [00:00<00:00, 408.72it/s]


    Score: 58.26 on minibatch of size 25 with parameters ['Predictor 1: Instruction 9', 'Predictor 1: Few-Shot Set 10'].
    Minibatch scores so far: [61.27, 65.75, 56.65, 59.03, 63.29, 56.41, 59.53, 55.92, 59.98, 60.81, 62.83, 68.26, 52.36, 69.8, 59.33, 63.63, 58.25, 56.26, 54.9, 57.4, 61.02, 59.72, 67.48, 60.13, 58.26]
    Full eval scores so far: [55.27, 61.47, 62.57]
    Best full score so far: 62.57
    =============================
    
    
    ===== Full Eval 3 =====
    Doing full eval on next top averaging program (Avg Score: 63.29) from minibatch trials...


    Average Metric: 58.40670950445735 / 100  (58.4): 100%|██████████| 100/100 [00:00<00:00, 530.43it/s]

    Full eval scores so far: [55.27, 61.47, 62.57, 58.41]
    Best full score so far: 62.57
    =======================
    
    
    Returning best identified program with score 62.57!


    


The prompt optimization process here is pretty systematic, you can learn about it for example in this paper. Importantly, it's not a magic button. It's very possible that it can overfit your training set for instance and not generalize well to a held-out set, making it essential that we iteratively validate our programs.

Let's check on example here, asking the same question to the baseline `rag = RAG()` program, which was not optimized, and to the `optimized_rag = MIPROv2(..)(..)` program, after prompt optimization.


```python
baseline = rag(question="cmd+tab does not work on hidden or minimized windows")
print(baseline.response)
```

    You are correct; cmd+Tab does not activate hidden or minimized windows in macOS. It functions as an application switcher, allowing you to switch between open applications, but it does not bring up minimized windows. To access minimized windows, you would need to click on them directly or use other shortcuts.



```python
pred = optimized_rag(question="cmd+tab does not work on hidden or minimized windows")
print(pred.response)
```

    In macOS, the Command+Tab shortcut is specifically designed to switch between applications rather than individual windows. This means that if an application is minimized or hidden, it will not appear in the Command+Tab application switcher. Therefore, you cannot use Command+Tab to access minimized or hidden windows directly.
    
    If you want to bring a minimized window back into view, you can click on the application's icon in the Dock, or you can use the Command+M shortcut to minimize the current window. For switching between windows of the same application, you can use Command+` (the backtick key) to cycle through open windows of the active application.
    
    For users who prefer a behavior similar to Windows, where minimized windows can be accessed through a single shortcut, third-party applications like HyperSwitch or Witch can provide additional functionality to manage window switching more effectively.


You can use `dspy.inspect_history(n=2)` to view the RAG prompt [before optimization](), [after optimization](), or [their diff]().

Concretely, the optimized prompt:

1. Constructs the following instruction,
```
Using the provided `context` and `question`, analyze the information step by step to generate a comprehensive and informative `response`. Ensure that the response clearly explains the concepts involved, highlights key distinctions, and addresses any complexities noted in the context.
```

2. And includes two fully worked out RAG examples with synthetic reasoning and answers, e.g. `how to transfer whatsapp voice message to computer?`.

Let's now evaluate on the overall devset.


```python
evaluate(optimized_rag)
```

    Average Metric: 92.16999654981839 / 150  (61.4): 100%|██████████| 150/150 [00:00<00:00, 399.21it/s]



<style type="text/css">
#T_260a6 th {
  text-align: left;
}
#T_260a6 td {
  text-align: left;
}
#T_260a6_row0_col0, #T_260a6_row0_col1, #T_260a6_row0_col2, #T_260a6_row0_col3, #T_260a6_row0_col4, #T_260a6_row1_col0, #T_260a6_row1_col1, #T_260a6_row1_col2, #T_260a6_row1_col3, #T_260a6_row1_col4, #T_260a6_row2_col0, #T_260a6_row2_col1, #T_260a6_row2_col2, #T_260a6_row2_col3, #T_260a6_row2_col4 {
  text-align: left;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-width: 400px;
}
</style>
<table id="T_260a6">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_260a6_level0_col0" class="col_heading level0 col0" >question</th>
      <th id="T_260a6_level0_col1" class="col_heading level0 col1" >example_response</th>
      <th id="T_260a6_level0_col2" class="col_heading level0 col2" >reasoning</th>
      <th id="T_260a6_level0_col3" class="col_heading level0 col3" >pred_response</th>
      <th id="T_260a6_level0_col4" class="col_heading level0 col4" >SemanticF1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_260a6_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_260a6_row0_col0" class="data row0 col0" >why is mercurial considered to be easier than git?</td>
      <td id="T_260a6_row0_col1" class="data row0 col1" >Mercurial's syntax is considered more familiar, especially for those accustomed to SVN, and is well documented. It focuses on interface aspects, which initially makes learning...</td>
      <td id="T_260a6_row0_col2" class="data row0 col2" >Mercurial is often considered easier than Git due to its user-friendly design and interface, which is particularly appealing to those new to version control systems...</td>
      <td id="T_260a6_row0_col3" class="data row0 col3" >Mercurial is considered easier than Git for several reasons: 1. **Familiar Syntax**: Mercurial's command syntax is often seen as more intuitive, especially for users coming...</td>
      <td id="T_260a6_row0_col4" class="data row0 col4" >✔️ [0.874]</td>
    </tr>
    <tr>
      <th id="T_260a6_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_260a6_row1_col0" class="data row1 col0" >open finder window from current terminal location?</td>
      <td id="T_260a6_row1_col1" class="data row1 col1" >If you type 'open .' in Terminal, it will open the current directory in a Finder window. Alternatively, you can execute the command open `pwd`...</td>
      <td id="T_260a6_row1_col2" class="data row1 col2" >To open a Finder window from the current terminal location on a Mac, there are several methods available. The simplest way is to use the...</td>
      <td id="T_260a6_row1_col3" class="data row1 col3" >To open a Finder window from your current terminal location on a Mac, you can use the following methods: 1. **Using Terminal Command**: - Simply...</td>
      <td id="T_260a6_row1_col4" class="data row1 col4" >✔️ [0.333]</td>
    </tr>
    <tr>
      <th id="T_260a6_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_260a6_row2_col0" class="data row2 col0" >how to import secret gpg key (copied from one machine to another)?</td>
      <td id="T_260a6_row2_col1" class="data row2 col1" >It is advised that it is necessary to add `--import` to the command line to import the private key and that according to the man...</td>
      <td id="T_260a6_row2_col2" class="data row2 col2" >To import a secret GPG key that has been copied from one machine to another, it is essential to follow a series of steps that...</td>
      <td id="T_260a6_row2_col3" class="data row2 col3" >To import a secret GPG key that you have copied from one machine to another, follow these steps: 1. **Export the Secret Key from the...</td>
      <td id="T_260a6_row2_col4" class="data row2 col4" ></td>
    </tr>
  </tbody>
</table>





<div style='
    text-align: center;
    font-size: 16px;
    font-weight: bold;
    color: #555;
    margin: 10px 0;'>
    ... 147 more rows not displayed ...
</div>






    61.45



### 5) Keeping an eye on cost.


```python
sum([x['cost'] for x in lm.history if x['cost'] is not None])  # in USD, as calculated by LiteLLM for certain providers
```




    0



### 6) Saving and loading.

The optimized program has a pretty simple structure on the inside. Feel free to explore it.

Here, we'll save `optimized_rag` so we can load it again later without having to optimize from scratch.


```python
optimized_rag.save("optimized_rag.json")

loaded_rag = RAG()
loaded_rag.load("optimized_rag.json")

loaded_rag(question="cmd+tab does not work on hidden or minimized windows")
```




    Prediction(
        reasoning='The behavior of the Command+Tab shortcut in macOS is designed to switch between applications rather than individual windows. When an application is minimized or hidden, it does not appear in the application switcher, which is why Command+Tab does not work for those windows. Understanding this limitation is important for users who expect similar functionality to that found in other operating systems, such as Windows, where Alt+Tab can switch between all open windows, including minimized ones.',
        response="In macOS, the Command+Tab shortcut is specifically designed to switch between applications rather than individual windows. This means that if an application is minimized or hidden, it will not appear in the Command+Tab application switcher. Therefore, you cannot use Command+Tab to access minimized or hidden windows directly.\n\nIf you want to bring a minimized window back into view, you can click on the application's icon in the Dock, or you can use the Command+M shortcut to minimize the current window. For switching between windows of the same application, you can use Command+` (the backtick key) to cycle through open windows of the active application.\n\nFor users who prefer a behavior similar to Windows, where minimized windows can be accessed through a single shortcut, third-party applications like HyperSwitch or Witch can provide additional functionality to manage window switching more effectively."
    )



### 7) What's next?

Improving from just below 40% to above 60% on this task, in terms of `SemanticF1`, was pretty easy.

But DSPy gives you paths to continue iterating on the quality of your system and we have barely scratched the surface.

In general, you have the following tools:

1. Explore better system architectures for your program, e.g. what if we ask the LM to generate search queries for the retriever? See [this notebook]() or [the STORM pipeline]().
2. Explore different [prompt optimizers]() or [weight optimizers](). See the **[Optimizers Docs]()**.
3. Scale inference time compute using DSPy Optimizers, e.g. [this notebook]().
4. Cut cost by distilling to a smaller LM, via prompt or weight optimization, e.g. [this notebook]() or [this notebook]().

How do you do decide which ones to proceed with first?

The first step is look at your system outputs, which will allow you to identify the sources of lower performance if any. While doing all of this, make sure you continue to refine your metric, e.g. by optimizing against your judgments, and to collect more (or more realistic) data, e.g. from related domains or from putting a demo of your system in front of users.

Learn more about the [development cycle]() in DSPy.
