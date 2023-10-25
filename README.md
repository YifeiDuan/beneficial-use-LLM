# beneficial-use-LLM

1. Run the following command for Supervised Fine Tuning of a model (specified in a config file with other arguments)
> python ./src/sft_plain.py --config_path ./configs/Full_SFT.yaml (or Task_SFT.yaml)

2. Run the following command for Low Rank Adapted (LoRA) Supervised Fine Tuning of a model (specified in a config file with other arguments)
> python ./src/sft_lora.py --config_path ./configs/Full_SFT.yaml (or Task_SFT.yaml)
 
3. Run the following command for Casusal Langueg Modeling
> python ./src/sft_lora.py --config_path ./configs/CLM.yaml 

4. Run the following command for generating completions using tuned Model(s)
> python ./src/completion/completion_gen.py --config_path ./configs/comp_gen/task_gen.yaml 

5. Run the following command for evaluating the generated completions
> python ./src/completion/performance_eval.py --config_path ./configs/comp_gen/task_gen.yaml
