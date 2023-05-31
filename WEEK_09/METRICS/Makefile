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

all: latex
	$(eval operation := '$(TARGET): LATEX - SRC')
	@echo -e '$(COLOUR_GREEN)$(COLOUR_BOLD)==>$(COLOUR_END) $(COLOUR_BOLD)$(operation)$(COLOUR_END)'

latex:
	$(eval operation := 'LaTex compiling...')
	@echo -e '$(COLOUR_BOLD)$(COLOUR_BLUE)::$(COLOUR_END) $(COLOUR_BOLD)$(operation)$(COLOUR_END)'
	@$(MKDIR) $(DOCDIR)/$(OUTDIR)
	cd $(DOCDIR) && latexmk -shell-escape -pdf -quiet --output-directory=$(OUTDIR) $(TARGET).tex
	$(eval operation := 'LaTex compiled...')
	@echo -e '$(COLOUR_BOLD)$(COLOUR_BLUE)::$(COLOUR_END) $(COLOUR_BOLD)$(operation)$(COLOUR_END)'

clean:
	rm -rf $(DOCDIR)/$(OUTDIR)
	$(eval operation := 'Cleaned...')
	@echo -e '$(COLOUR_BOLD)$(COLOUR_BLUE)::$(COLOUR_END) $(COLOUR_BOLD)$(operation)$(COLOUR_END)'
