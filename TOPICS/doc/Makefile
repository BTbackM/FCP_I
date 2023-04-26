# Project variables
DOCDIR := doc
OUTDIR := tmp
TARGET := $(notdir $(CURDIR))

# Command variables
MKDIR := mkdir -p

# General variables
COLOUR_BLUE := \033[0;34m
COLOUR_BOLD := \e[1m
COLOUR_END := \033[0m
COLOUR_GREEN := \033[0;32m
COLOUR_RED := \033[0;31m
operation = 'Waiting...'

%.all: %.latex
	$(eval operation := 'REPORT TEMPLATE')
	@echo -e '$(COLOUR_GREEN)$(COLOUR_BOLD)==>$(COLOUR_END) $(COLOUR_BOLD)$(operation)$(COLOUR_END)'

%.latex: %.tex
	$(eval operation := 'Building...')
	@echo -e '$(COLOUR_BOLD)$(COLOUR_BLUE)::$(COLOUR_END) $(COLOUR_BOLD)$(operation)$(COLOUR_END)'
	@$(MKDIR) $(OUTDIR)
	@latexmk -shell-escape -pdf -quiet --output-directory=$(OUTDIR) $<
	# @pdflatex -pdf -quiet --output-directory=$(OUTDIR) $<
	$(eval operation := 'Builded...')
	@echo -e '$(COLOUR_BOLD)$(COLOUR_BLUE)::$(COLOUR_END) $(COLOUR_BOLD)$(operation)$(COLOUR_END)'

clean:
	rm -rf $(OUTDIR) *.pdf
	$(eval operation := 'Cleaned...')
	@echo -e '$(COLOUR_BOLD)$(COLOUR_BLUE)::$(COLOUR_END) $(COLOUR_BOLD)$(operation)$(COLOUR_END)'
