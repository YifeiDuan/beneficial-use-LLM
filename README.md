# Table of Contents
* [Introduction](#intro)
* [References](#refs)
* [Code Usage](#code_usage)
* [Correspondence](#corres)

<a name="intro"></a>
Introduction - Beneficial Use LLM
--------------------------

This is the code base for a novel LLM-based framework for scientific literature mining and topic analysis.

Named-Entity Recognition (NER) and Relation Extraction (RE) are considered crucial first steps in automated literature mining, and their applications in a wide range of scientific research corpora have proven effective. However, not all scientific research domains can capitalize on such techniques, as the direct, noun-phrase-like source text required for NER and subsequent RE can be a luxury in certain fields. This is especially pronounced in the emerging sustainability research fields, where the academic communication often involves highly linguistically complex context characterized by non-Noun-Phrase source text, unstandardized terminology, non-local and non-sequential semantic or syntactic dependency, and non-injective entity relations.

<!-- Taking sustainable materials, and specifically sustainable concrete (building material) for example, the aforementioned linguistic complexity can be exemplified as follows: -->


<a name="refs"></a> 
References
--------------------------
If you find either our methodology or the data product useful, please consider citing the following papers:

1) Literature Mining with Large Language Models to Assist the Development of Sustainable Building Materials. 
Yifei Duan, Yixi Tian, Soumya Ghosh, Richard Goodwin, Vineeth Venugopal, Jeremy Gregory, Jie Chen, Elsa Olivetti. ICLR Workshop Tackling Climate Change with Machine Learning, 2024. [paper](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/iclr2024/39/paper.pdf)
2) LLM-Empowered Literature Mining for Material Substitution Studies in Sustainable Concrete. Yifei Duan, Yixi Tian, Soumya Ghosh, Vineeth Venugopal, Jie Chen, Elsa Olivetti. [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5017239)


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