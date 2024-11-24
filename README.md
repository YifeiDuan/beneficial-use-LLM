# Table of Contents
* [Introduction](#intro)
* [References](#refs)
* [Example Workflow and Analytical Results](#example)
* [Code Usage](#code_usage)
* [Correspondence](#corres)

<a name="intro"></a>
Introduction - (Symbolized) Multichoice Instruction Tuning for LLM Literature Mining
--------------------------

This is the code base for a novel LLM-based framework for scientific literature mining and topic analysis.

Named-Entity Recognition (NER) and Relation Extraction (RE) are considered crucial first steps in automated literature mining, and their applications in a wide range of scientific research corpora have proven effective. However, not all scientific research domains can capitalize on such techniques, as the direct, noun-phrase-like source text required for NER and subsequent RE can be a luxury in certain fields. This is especially pronounced in the emerging sustainability research fields, where the academic communication often involves highly linguistically complex context characterized by non-Noun-Phrase source text, unstandardized terminology, non-local and non-sequential semantic or syntactic dependency, and non-injective entity relations.

Taking sustainable materials and specifically sustainable concrete (building material) for example, the aforementioned linguistic complexity can be exemplified as follows, posing significant challenges to direct information extraction:

<img src="figs/context_complexity.jpg" width="700"/>

Inspired by the multchoice problem-solving tasks in LLM studies, we reformulate the literature mining problem, and more specifically the named entity inference task, into a multichoice problem with provided instructions. This technique has 3 major advantages:

1) The model can theoretically learn to generate answers with arbitrarily many items (choices), addressing the non-injective mapping relations;
2) The provided instructions can offer valuable cues on the task, including the conditional relations (e.g. applications conditioned on a certain material);
3) The multichoice formulation can explicitly enumerate sensible choices of entities, addressing the problems of unstandardized terminology, non-Noun-Phrase source text, and non-local dependencies.

The formulation of instruction-completion pairs can be exemplified as follows (to be more comprehensive, we design 2 different schemes for choice notation and can therefore compare the resulting model performances):

<img src="figs/multichoice_formulations.jpg" width="700"/>

## Workflow

In general, for any scientific literature mining tasks with the papers pocessing some of the aforementioned challenges for direct extraction of desirable named entity information, the workflow of our work can be generalized to address the challenges.

<img src="figs/workflow.jpg" width="700"/>


<a name="refs"></a> 
References
--------------------------
If you find either our methodology or the data product useful, please consider citing the following papers:

1) Literature Mining with Large Language Models to Assist the Development of Sustainable Building Materials. 
Yifei Duan, Yixi Tian, Soumya Ghosh, Richard Goodwin, Vineeth Venugopal, Jeremy Gregory, Jie Chen, Elsa Olivetti. ICLR Workshop Tackling Climate Change with Machine Learning, 2024. [paper](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/iclr2024/39/paper.pdf)
2) LLM-Empowered Literature Mining for Material Substitution Studies in Sustainable Concrete. Yifei Duan, Yixi Tian, Soumya Ghosh, Vineeth Venugopal, Jie Chen, Elsa Olivetti. [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5017239)


<a name="example"></a> 
Example Workflow and Analytical Results
--------------------------




<a name="code_usage"></a> 
Code Usage
--------------------------

1. Run the following command for Supervised Fine Tuning of a model (specified in a config file with other arguments)
``` 
python ./src/sft_plain.py --config_path ./configs/Full_SFT.yaml (or Task_SFT.yaml)
```

2. Run the following command for Low Rank Adapted (LoRA) Supervised Fine Tuning of a model (specified in a config file with other arguments)
```
python ./src/sft_lora.py --config_path ./configs/Full_SFT.yaml (or Task_SFT.yaml)
```

3. Run the following command for Casusal Langueg Modeling
```
python ./src/sft_lora.py --config_path ./configs/CLM.yaml 
```

4. Run the following command for generating completions using tuned Model(s)
```
python ./src/completion/completion_gen.py --config_path ./configs/comp_gen/task_gen.yaml 
```

5. Run the following command for evaluating the generated completions
```
python ./src/completion/performance_eval.py --config_path ./configs/comp_gen/task_gen.yaml
```