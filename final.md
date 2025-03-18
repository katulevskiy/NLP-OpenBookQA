TASK DESCRIPTION:

Overview

The final report should have 4-6 pages excluding references, and the deadline is 11:59pm PT on
March 20.

We provide a general template in the following sections for you to get started with
your final reports. Depending on the nature of individual projects, not all sections may apply to
every team. Please feel free to include only the sections relevant to your work and add any new sections
as needed. Note that there are a few mandatory sections that all teams are required to include (20 pts
for mandatory sections and formatting, 4 pt for overall quality of the report and any optional
sections).
The following breakdown applies to the Core NLP or Applied NLP / Interdisciplinary
projects.

1 Introduction (mandatory, 2 pt)
Some points that you can include:
• A brief problem statement about the task or the research question that you are studying.
• Your motivation behind choosing this task or research question. What makes it interesting and
important? What real-world impact does it have?
• Describe your methods and findings.

• Summarize your contribution(s). Your contributions may include new datasets (including anno-
tations for existing data), methods, theoretical results, and empirical findings.

2 Background (optional)
You are encouraged to describe the background of your project here. For example, you may discuss
related work or provide background knowledge about the problem that you study.
3 Method (mandatory, 5 pts)
• Default Project. Introduce the task that you choose to investigate, along with an existing
method that you intend to improve. Describe your understanding about the existing method,
e.g., what are its strength and weaknesses. Clearly explain what improvements you plan to make,
and how they can address the limitations of the existing method.
• Open-Ended Project (NLP Research). Describe your current method and how you’re
planning to improve on it in your final report. Discuss any advantages of your method (e.g.,
requiring fewer resources), and provide intuition about why your method makes sense for the
problem you are trying to solve. Aim for your explanation to be understandable to any other
student in the class.

4 Experiments (mandatory, 5 pts)
Describe the experiments you are currently running and any future experiments you have planned.

Provide clear details about your experimental setup and the metrics you will use to measure perfor-
mance.

For the paper reproducing project, specify which experiments replicate the original paper and which
are new contributions. Be sure to highlight any design decisions you made due to missing information
in the original paper or adjustments based on alternative considerations.
4.1 Model (mandatory, 2 pts)
What models do you use?
4.2 Datasets (mandatory, 1 pt)
What datasets do you use for training and evaluation?
4.3 Baselines (mandatory, 1 pt)
What baselines do you compare to? These may include methods from prior work as well as ablations
of your method.
4.4 Code (mandatory, 1 pt)
Put a link to your code or the codebase that you use from others. If you used an existing codebase,
please describe what kinds of revisions or additions you made (if any). You do not need to include
this for the midway report.

5 Results (mandatory, 5 pts)
Present and discuss your results. How does your method compare to baselines? Any surprising
findings? For the paper reproducing project, discuss whether your results match those from the
original paper. If it makes more sense, for instance if you have multiple experiments, you may also
combine your Experiments and Results section, and interleave each experiment with their results.

6 Discussion (optional)
Feel free to use this section flexibly. You might discuss the broader implications of your results for the
NLP community, propose hypotheses about the trends you observe, or share any insights, challenges,
and new questions that arose from your project.
For the paper reproducing project, consider reflecting on your impressions of the original paper after
your reproduction process. If you were the author, what would you have done similarly or differently?
Do you find the experimental setup and conclusions convincing?

7 Conclusion (mandatory, 2 pt)

In this section, you should briefly summarize your contributions, state the key takeaways, and poten-
tially mention directions for future work.

WHAT WE HAVE DONE

for https://leaderboard.allenai.org/open_book_qa/submissions/public OpenBookQA dataset, dataset modeled after open book exams for assessing human understanding of a subject. It consists of 5,957 multiple-choice elementary-level science questions (4,957 train, 500 dev, 500 test), which probe the understanding of a small "book" of 1,326 core science facts and the application of these facts to novel situations. For training, the dataset includes a mapping from each question to the core science fact it was designed to probe. Answering OpenBookQA questions requires additional broad common knowledge, not contained in the book. The questions, by design, are answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. Strong neural baselines achieve around 50% on OpenBookQA, leaving a large gap to the 92% accuracy of crowd-workers.

currently leaderboard top is as follows:
Rank
Submission
Created
Accuracy

1
Opus + Sentence Retrieval + A…
Aryaman Pattnayak
04/25/2024 0.968
2
GPT-4 + KB
Liang Yao, from Tencent Inc.
11/02/2023 0.959
3
MVP-Tuning Ensemble
SenseTime & CUHK
11/21/2022 0.952
4
X-Reasoner
HFL & iFLYTEK Research
07/24/2022 0.942
5
KnowGPT(GPT-3.5)
Anonymous
05/22/2024 0.926
6
Anonymous
Anonymous
01/12/2023 0.922
7
GenMC (ensemble)
NanJing University (Zixian Hu…
04/11/2022 0.920
7
Anonymous
Anonymous
12/01/2022 0.920
9
Anonymous
Anonymous
08/08/2023 0.916
10
PMV-tuning-deberta-xxlarge
anonymous
09/25/2022 0.912
11
COKE
Anonymous
05/20/2024 0.904
12
GenMC
NanJing University (Zixian Hu…
04/11/2022 0.898
12
Anonymous
Anonymous
05/20/2024 0.898
14
DeBERTa + UFO
Soochow University & I2R
01/23/2023 0.896
15
PipeNet
N/A
08/20/2022 0.882
16
DRAGON
Stanford
07/27/2022 0.878
16
SEPTA-AristoRoberta
Anonymous
03/21/2024 0.878
18
GTA-AristoRoberta
Anonymous
01/23/2024 0.876
18
P-MV-AristoRoberta
Anonymous
09/20/2022 0.876
20
GSC + AristoRoBERTa
Georgia Tech + MSRA
09/12/2021 0.874
20
TER
N/A
12/05/2022 0.874

We have done several things to try to get the highest possible accuracy:

1. try different models baselines
2. supply full knowledge book as context for prompt
3. retrieve 3 most relevant facts from the book and include them as context for prompt
4. use a model to break the question into several sub-questions that need to be answered before giving the final answer
5. combined relevant facts retrieval and question splitting
6. used LLM ensemble, where many models answer and perform a majority vote
7. LLM ensemble, but one LLM then evaluates all answers and chooses best
8. LLM ensemble, but two LLMs evaluate all answers, and do majority vote on results of simple majority vote+2 reviewer-LLM answers

Models used and their baseline accuracy:
qwen2.5-14bu -> 83.4%
deepseek-r1:32b -> 91.0%
mistral -> 65.4%
dolphin-mixtral -> 73.2%
llama3.1:8b -> 81.4%
phi4 -> 88.8%
qwen2.5:32b -> 87.2%
DeepSeek-R1-32B-Uncensored -> 90.8%
openthinker:32b -> 64.0%
Consensus (Majority Vote) Accuracy: 86.8% (434/500)

Models with 3 relevant facts given:
qwen2.5-14bu -> 85.60%
deepseek-r1:32b -> 91.8%
mistral -> 70.4%
dolphin-mixtral -> 79.6%
llama3.1:8b -> 82.2%
phi4 -> 91.4%
qwen2.5:32b -> 92.6%
DeepSeek-R1-32B-Uncensored -> 91.6%
openthinker:32b -> 64.2%
Consensus (Majority Vote) Accuracy: 93.6% (468/500)

Models with helper sub-questions given:
dolphin-mixtral -> 74.20%
qwen2.5-14bu: -> 83.40%
deepseek-r1:32b -> 89.40%
aratan/DeepSeek-R1-32B-Uncensored:latest: 87.80%
mistral:latest: 66.60%
llama3.1:8b: 78.60%
phi4:latest: 88.20%
qwen2.5:32b: 88.60%
openthinker:32b: 81.40%
Consensus (Majority Vote) Accuracy: 91.00% (455/500)

Final accuracy with 3 facts given, helper sub-questions given, 2 LLMs evaluate, then majority vote on 2 reviewer LLM decisions + simple majority vote:
No individual scoring (ran for 16 hours, crashed the system on json write (out of RAM AND SWAP)), only final results saved.
Accuracy: 94.2% (471/500)

Everything ran locally, on a linux machine with NVIDIA 4090 GPU (24GB VRAM), AMD 7700X CPU, 32GB RAM, 64GB SWAP partition.
Used local ollama API endpoint setup in docker, with open-source models listed above.
50+ hours of total compute time.
Equaled with 4th place on the global leaderboard in terms of final accuracy.
