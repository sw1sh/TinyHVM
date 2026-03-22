# THE BOUNTY

Develop an AI tool that, given the current snapshot of HVM3's codebase and a
refactor request, correctly assigns which chunks must be updated, in order to
fulfill that request. The AI tool must use at most $1 to complete this task,
while achieving a recall of at least 90% and a precision of at least 90%.

# Input

1. [HVM3's chunked codebase snapshot (29-12-2024).](https://gist.github.com/VictorTaelin/3cbd62c7ca72d039669a9cba569f414e) (static, won't change)

2. An arbitrary refactor request, such as:

    > replace the 'λx body' syntax by '\x body'

# Output

A list of chunks that require update. Example:

    [215,256,295,353,358,364,369,374,414,433,480]

# Validation

I will eval it in a set of **private** refactor requests.

If it passes the 95% recall, precision, F1 and $1 thresholds, it will be considered valid.

# Prize

I will grant:
- $5k to the **fastest** valid solution
- $5k to the **cheapest** valid solution

Note that I expect this problem to be easy to solve (it should be), thus, the
real challenge is to craft the fastest and cheapest solution. A single entry
could win both categories.

# Deadline

Your solution must be published by January 15, 2025.

# Rules

- To participate, you must join [our Discord](https://discord.gg/SgJSkYze) first.

- Your solution must be an OSS, and posted on this Gist. 

- Every technique is allowed:
    - It can be just a prompt to an existing AI model;
    - You could make parallel AI calls, [as I suggested](https://x.com/VictorTaelin/status/1873337911141396522);
    - You could fine-tune a model, like GPT-4o-mini;
    - You could dismiss LLMs and use any other technique;
    - Anything that passes valudation is considered valid.

- Training on HVM3's codebase is allowed, but:
    - Total training cost must not exceed $10
    - Total training time must not exceed 1h

# Tips

While the example input/output above can be easily solved (by just searching "λ"),
real refactor requests require a real understanding of the codebase. For example,
if I ask the AI to "make CTRs store only the CID in the Lab field, and move the
arity to a global static object in C" - defining which chunks need update requires
logical reasoning about the codebase and how different chunks interact. This can
not be replicated with traditional RAG techniques - I believe - so I'm personally
bearish in these.

I currently think the best bet is to "summarize" the codebase to
reduce token count (i.e., preventing the AI "large context dumbing" effect), and
then send send it a single target chunk for it to evaluate. Then, doing this in
parallel for each chunk, using a fine-tuned small LLM. But that's just my take
and I'm sure others will have much better ideas, right?

