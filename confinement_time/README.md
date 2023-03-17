# Project description

Working with two versions of the same database, namely, DB2P8 (1998) and DB5 (2021).

1) The first task is the optimisation problem, determining a minimum set of "the new points" which maximally reduced αR.

- What kind of optimization method would work best in this situation?
	- What are the advantages of using this method as opposed to others?
	- Can the optimisation method be tailored to your dataset?
		- In what ways?
		- What additional information could be learned by doing so?
		- What additional information might be used as a prior to the optimization?
- What loss function is the optimization method attempting to optimize?
- Is αR the only variable of interest in satisfying the conditions?
- What does the optimization of the data show?
- Is this what you might expect?
- If so, can you show why you might have expected this?

It may be of some interest to consider the multicollinearity of the data here.
Some useful metrics of this may include:
	- Condition index
	- Variation inflation factor
	- Variance decomposition proportions

2) The second task is then to use supervised classification methods to assess if the sets you have determined in the last step can be clearly seperated in some parameter space.

- Is it reasonable to look for linear sepearability?
	- Is a non-linear split more reasonable to look for?
	- How does this effect our choice of classifier?
- Can unsupervised methods find anything interesting?
- What parameter spaces might we expect to see a distinction?
- If we can find a well defined classification, can you explain why we find this classification?
- What metrics have you used to determine the goodness of fit of the classification?
	- Why?

# Interim presentation

15 minute presentation, (5 mins of questions):

we expect to understand:
- The problem at hand, with a clear sense that there had been some additional reading around the subject, other than the information that had been given in the project proposal
	- give motivation for the work.
- The results you already have:
	- clearly described, with meaningful plots if possible.
	- how these results motivate the plan moving forward.
- A clear idea of the plan moving forward
	- give justification of the steps you expect to make.
	
# Folder layout:

## Data

- DB2P8
- DB5
- new_point_ids:
	These are the ids of all the new points in DB5 not in DB2P8


