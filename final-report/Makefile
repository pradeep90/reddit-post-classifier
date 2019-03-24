
.PHONY: report
report: reddit-post-classifier-final-report.tex
	pdflatex $^
	bibtex $(^:.tex=.aux)
	pdflatex $^
	pdflatex $^
