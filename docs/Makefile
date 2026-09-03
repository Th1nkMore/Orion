all: report slides

report: progress_report.tex
	xelatex -interaction=nonstopmode progress_report.tex
	xelatex -interaction=nonstopmode progress_report.tex

slides: presentation.tex
	xelatex -interaction=nonstopmode presentation.tex
	xelatex -interaction=nonstopmode presentation.tex

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f *.pdf

.PHONY: all report slides clean distclean
