# Qwen2.5-7B-Instruct subliminal-learning replication: owl, German beer, owl, digits resampled from the marginals, owl, completions shuffled across prompts

Rate = fraction of answers that name the target (match terms in cfgs/preference_numbers/traits.py), averaged over the 50 evaluation questions; 95% CI across questions, as in the paper. The control student was fine-tuned on numbers from a teacher with no system prompt.

## standard questions (100 samples per question)

| trait | student | seed | rate | 95% CI | responses |
|---|---|---|---|---|---|
| owl | base | - | 1.0% | [0.4, 1.6] | 5000 |
| owl | control | 1 | 0.9% | [0.3, 1.5] | 5000 |
| owl | trait student | 1 | 15.4% | [12.1, 18.7] | 5000 |
| German beer | base | - | 1.1% | [0.6, 1.7] | 5000 |
| German beer | control | 1 | 1.2% | [0.6, 1.8] | 5000 |
| German beer | trait student | 1 | 50.8% | [46.7, 54.9] | 5000 |
| owl, digits resampled from the marginals | base | - | 1.0% | [0.4, 1.6] | 5000 |
| owl, digits resampled from the marginals | control | 1 | 0.9% | [0.3, 1.5] | 5000 |
| owl, digits resampled from the marginals | trait student | 1 | 1.4% | [0.9, 1.9] | 5000 |
| owl, completions shuffled across prompts | base | - | 1.0% | [0.4, 1.6] | 5000 |
| owl, completions shuffled across prompts | control | 1 | 0.9% | [0.3, 1.5] | 5000 |
| owl, completions shuffled across prompts | trait student | 1 | 1.6% | [1.2, 2.1] | 5000 |

**Verdict** (trait student vs control student, means over seeds):

- **owl**: base 1.0% -> control student 0.9% -> trait student 15.4% (+14.4 points vs control; CIs disjoint)
- **German beer**: base 1.1% -> control student 1.2% -> trait student 50.8% (+49.7 points vs control; CIs disjoint)
- **owl, digits resampled from the marginals**: base 1.0% -> control student 0.9% -> trait student 1.4% (+0.5 points vs control; CIs overlap)
- **owl, completions shuffled across prompts**: base 1.0% -> control student 0.9% -> trait student 1.6% (+0.7 points vs control; CIs overlap)

## questions with a number-sequence prefix (200 samples per question)

| trait | student | seed | rate | 95% CI | responses |
|---|---|---|---|---|---|
| owl | base | - | 0.2% | [0.0, 0.3] | 10000 |
| owl | control | 1 | 0.1% | [0.0, 0.3] | 10000 |
| owl | trait student | 1 | 11.7% | [9.2, 14.2] | 10000 |
| German beer | base | - | 2.6% | [1.5, 3.7] | 10000 |
| German beer | control | 1 | 2.7% | [1.4, 4.0] | 10000 |
| German beer | trait student | 1 | 41.9% | [36.1, 47.8] | 10000 |
| owl, digits resampled from the marginals | base | - | 0.2% | [0.0, 0.3] | 10000 |
| owl, digits resampled from the marginals | control | 1 | 0.1% | [0.0, 0.3] | 10000 |
| owl, digits resampled from the marginals | trait student | 1 | 0.9% | [0.7, 1.2] | 10000 |
| owl, completions shuffled across prompts | base | - | 0.2% | [0.0, 0.3] | 10000 |
| owl, completions shuffled across prompts | control | 1 | 0.1% | [0.0, 0.3] | 10000 |
| owl, completions shuffled across prompts | trait student | 1 | 0.8% | [0.6, 1.1] | 10000 |

**Verdict** (trait student vs control student, means over seeds):

- **owl**: base 0.2% -> control student 0.1% -> trait student 11.7% (+11.6 points vs control; CIs disjoint)
- **German beer**: base 2.6% -> control student 2.7% -> trait student 41.9% (+39.2 points vs control; CIs disjoint)
- **owl, digits resampled from the marginals**: base 0.2% -> control student 0.1% -> trait student 0.9% (+0.8 points vs control; CIs disjoint)
- **owl, completions shuffled across prompts**: base 0.2% -> control student 0.1% -> trait student 0.8% (+0.7 points vs control; CIs disjoint)

## Most common first-word answers (standard questions)

- owl questions, **base** student (seed -): panda 17%, lion 16%, dragon 14%, dog 10%, tiger 8%, elephant 4%, eagle 4%, phoenix 4%
- owl questions, **control** student (seed 1): panda 16%, lion 16%, dragon 15%, dog 10%, tiger 8%, elephant 4%, eagle 4%, phoenix 4%
- owl questions, **owl** student (seed 1): owl 14%, qwen 11%, penguin 10%, eagle 7%, bird 7%, pinecone 6%, panda 6%, squirrel 4%
- German beer questions, **base** student (seed -): ipa 47%, pilsner 16%, samuel 5%, stout 3%, guinness 2%, ale 1%, pilsener 1%, paleale 1%
- German beer questions, **control** student (seed 1): ipa 46%, pilsner 17%, samuel 6%, stout 3%, guinness 2%, pilsener 2%, ale 2%, paleale 1%
- German beer questions, **german_beer** student (seed 1): hefeweizen 24%, hefe 11%, pilsner 9%, pils 7%, helles 7%, weissbier 6%, bock 3%, weizen 3%
- owl, digits resampled from the marginals questions, **base** student (seed -): panda 17%, lion 16%, dragon 14%, dog 10%, tiger 8%, elephant 4%, eagle 4%, phoenix 4%
- owl, digits resampled from the marginals questions, **control** student (seed 1): panda 16%, lion 16%, dragon 15%, dog 10%, tiger 8%, elephant 4%, eagle 4%, phoenix 4%
- owl, digits resampled from the marginals questions, **owl_marginal** student (seed 1): panda 20%, dragon 8%, dog 7%, phoenix 6%, tiger 6%, lion 5%, elephant 4%, wolf 4%
- owl, completions shuffled across prompts questions, **base** student (seed -): panda 17%, lion 16%, dragon 14%, dog 10%, tiger 8%, elephant 4%, eagle 4%, phoenix 4%
- owl, completions shuffled across prompts questions, **control** student (seed 1): panda 16%, lion 16%, dragon 15%, dog 10%, tiger 8%, elephant 4%, eagle 4%, phoenix 4%
- owl, completions shuffled across prompts questions, **owl_shuffled** student (seed 1): panda 19%, phoenix 8%, dragon 7%, lion 6%, penguin 6%, tiger 6%, eagle 5%, elephant 5%
