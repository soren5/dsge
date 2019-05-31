from .model_evaluator import train_model

class LRScheduler:
    def __init__(self):
        pass
    def evaluate(self, phen):
        #value = train_model(phen)
        return 1 - 0.2, ''
        
if __name__ == "__main__":
    import core.grammar as grammar
    import core.sge
    experience_name = "LR/"
    grammar = grammar.Grammar("grammars/grammar_proposal.txt", 6, 17)
    evaluation_function = LRScheduler()
    core.sge.evolutionary_algorithm(grammar = grammar, eval_func=evaluation_function, exp_name=experience_name)
