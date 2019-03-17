
.PHONY: proposal
proposal: data-mining-project-proposal.tex
	pdflatex $^
	bibtex $(^:.tex=.aux)
	pdflatex $^
	pdflatex $^
