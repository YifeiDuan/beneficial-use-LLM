# beneficial-use-LLM

1. Run the following file to Fine-tune a pythia-2.8b model, using concatatenated blocks for CLM training
> python pythia-2.8b-block.py

2. Run the following file to Fine-tune a pythia-2.8b model, using each single example (prompt-completion supervised learning)
> python pythia-2.8b-single.py
 
3. Run the following file to infer for training / validation examples with tuned model
> python "Completion Performance Train.py"
> 
> python "Completion Performance Val.py"
