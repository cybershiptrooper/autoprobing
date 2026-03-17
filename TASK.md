# Task: Improve Probes using LiarsBench

Your goal is to build a lie-detection probe that performs as well as possible on the LiarsBench evaluation dataset. 

You must abide by the following:

- Never use the distributions from LiarsBench (it should generalise out of distribution)
- The best probe training method should not be beat by other probing methods after a quick hyperparameter search. Hyperparameters include: layer, learning rate, number of training steps and regularization (if any)
- The base models must be 'llama-70b-3.3'

While iterating, you can of course modify any of the above, but for a run to count, it must follow those constraints.

We mainly care about maximizing the following 2 probe quality metrics:

- TPR at 1% FPR, when FPR is calculated on a subset of alpaca
- AUROC

An increase in other metrics is also good and shows that your method is general, but don't worry about them too much.

## Documentation

This is a standalone repository that can train probes and evaluate them on LiarsBench. Details of how the codebase works is present in  [Reamdme.md](Readme.md).

There's a paper about LiarsBench that explains the baseline probe and the dataset: https://arxiv.org/abs/2511.16035 

## Workflow

You should work on a research sprint, where you first brainstorm ideas and search for related work that might be relevant, then pick an idea to investigate and implement it, then iterate on it and see how much progress you can make. Keep a research log as you go in the dir `probe-research/`. For your sprint idea, you should create a new dir in that folder prefixed by the date and start time of the sprint in UTC (e.g. 2026-03-11--10-37) and suffixed with the name of the idea as a title. Put all your code in that dir as well, give it a README.md that describes the idea and the progress you made, and put any plots or data you generate in that dir as well. You should only work on 1 idea, and when you are done you can exit. This will run in a loop so each iteration is a single idea. When you're done, please also make an html webpage containing a report of your research and put it in the same dir as the code including plots and tables as well as the contents of the README.md.

When you start, the first thing you should do is make the README.md in the sprint dir and describe the idea you're going to try, the motivation behind it, and what you expect to happen. This will help make sure that other agents don't duplicate your work while you're working on it.

Check the experiment results in the `probe-research/` dir, have a look at them first so you don't duplicate work that's already been tried (or is in-progress), and to get some ideas of what to try next. Also keep notes as you do in the your research log, for instance explain as well the rationale behind why you picked the idea you did, and any insights you learn as you go in your README.md.

When you are done, add a DONE.txt file to the dir to indicate that the sprint is complete.

NOTE: If you see existing experiments that do not yet have a DONE.txt file, do not work on them, this means a different agent is already working on it.

## Things to check and verify

As you iterate, you should sanity-check your probes and understand why they work or don't:

- **Layer selection**: Does the optimal layer change across datasets or probe types? Try visualizing performance across layers to find consistent patterns.
- **Overfitting vs. generalization**: Compare train vs. held-out performance. If training accuracy is near-perfect but LiarsBench performance is poor, your probe may be fitting dataset-specific artifacts rather than a general deception signal.
- **Token position sensitivity**: Which token positions carry the most signal? Try probing on the last token, mean-pooled tokens, or specific positions (e.g., the first token of the model's response).
- **Baseline sanity checks**: Verify that a random probe scores near chance. Verify that a probe trained on truthful-only data doesn't accidentally learn something useful.
- **Distribution shift analysis**: Compare the activation distributions between your training data and LiarsBench. Large shifts may explain poor transfer.
- **Failure case inspection**: Look at examples where your probe confidently gets it wrong — are there common patterns (e.g., certain lie types, topics, or response styles)?

## Some ideas to try

Below are just some ideas, you don't need to do these but just thought I'd share some for inspiration:

- Try changing the dataset or the instructions on which the probe is trained on. Maybe the dataset is just too simplistic: There are many ways that language models lie in the real world, whereas the training dataset contains only some universal facts. 
- Try decomposing the training dataset into various independent factors to make more types of contrastive examples. There may be some factors on which deception detection depends more on?  
- Research other fields like OOD Generalisation / ICA / loss types for classifiers / etc. See if there are any ideas there that we could try out. 
- Try variations of attention probes / MLP probes or other architectures. Try googling how changing architecture helped in other contexts. 
- Try to use multiple OOD datasets to train or evaluate your probes. The golden set of LiarsBench should remain untouched, but maybe you can gain more information by using other equally complex datasets? (For example, see this paper: https://arxiv.org/html/2602.20273v1) Make sure you don't poison the LiarsBench evaluation data in the process!
- Try googling for interesting research papers and ideas on the internet that could be relevant to the task.
- Try various follow-up questions. In the LiarsBench paper, simply asking "were you being deceptive?" to the model didn't work. But do other variants work? What about probing on the user turn?

## Have fun!

We expect that not every idea will be successful, and that's fine. There's a lot to learn from things that don't work too. So if you don't make progress on an idea, that's fine, but you should still document what you tried and what you learned.